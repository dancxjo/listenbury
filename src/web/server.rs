use std::io::{BufRead, BufReader, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};

use crate::live_trace::{SseBroadcaster, read_trace_jsonl, read_trace_session};
use crate::trace::viewer_payload::live_trace_events_to_viewer_payload;

use super::assets;

#[derive(Debug, Clone)]
pub struct ServeConfig {
    pub host: String,
    pub port: u16,
    pub payload: Option<PathBuf>,
    pub trace: Option<PathBuf>,
    pub broadcaster: Option<SseBroadcaster>,
}

#[derive(Debug, Clone)]
struct ServerState {
    payload: Option<PathBuf>,
    trace: Option<PathBuf>,
    broadcaster: Option<SseBroadcaster>,
}

#[derive(Debug)]
pub struct BoundServer {
    listener: TcpListener,
    local_addr: SocketAddr,
    state: Arc<ServerState>,
}

impl BoundServer {
    pub fn local_addr(&self) -> SocketAddr {
        self.local_addr
    }

    pub fn serve(self) -> Result<()> {
        println!(
            "Listenbury web viewer serving on http://{}",
            self.local_addr
        );
        println!("Routes: /, /replay, /screenplay, /assets/*, /fixtures/*, /api/*, /api/trace-session, /api/live-events, /healthz");

        for stream in self.listener.incoming() {
            let mut stream = match stream {
                Ok(stream) => stream,
                Err(error) => {
                    eprintln!("web viewer accept error: {error}");
                    continue;
                }
            };
            let state = Arc::clone(&self.state);
            std::thread::spawn(move || {
                if let Err(error) = handle_connection(&mut stream, &state) {
                    eprintln!("web viewer request error: {error:#}");
                    let _ = write_response(
                        &mut stream,
                        &HttpResponse::internal_error("request handling failed\n"),
                        false,
                    );
                }
            });
        }

        Ok(())
    }
}

#[derive(Debug)]
struct HttpResponse {
    status: u16,
    reason: &'static str,
    content_type: &'static str,
    cache_control: &'static str,
    body: Vec<u8>,
    headers: Vec<(&'static str, String)>,
}

impl HttpResponse {
    fn ok(content_type: &'static str, body: impl Into<Vec<u8>>) -> Self {
        Self {
            status: 200,
            reason: "OK",
            content_type,
            cache_control: "no-store",
            body: body.into(),
            headers: Vec::new(),
        }
    }

    fn static_asset(content_type: &'static str, body: impl Into<Vec<u8>>) -> Self {
        Self {
            status: 200,
            reason: "OK",
            content_type,
            cache_control: "public, max-age=3600",
            body: body.into(),
            headers: Vec::new(),
        }
    }

    fn not_found(message: impl Into<String>) -> Self {
        Self {
            status: 404,
            reason: "Not Found",
            content_type: "text/plain; charset=utf-8",
            cache_control: "no-store",
            body: message.into().into_bytes(),
            headers: Vec::new(),
        }
    }

    fn method_not_allowed(message: impl Into<String>) -> Self {
        Self {
            status: 405,
            reason: "Method Not Allowed",
            content_type: "text/plain; charset=utf-8",
            cache_control: "no-store",
            body: message.into().into_bytes(),
            headers: vec![("Allow", "GET, HEAD".to_string())],
        }
    }

    fn internal_error(message: impl Into<String>) -> Self {
        Self {
            status: 500,
            reason: "Internal Server Error",
            content_type: "text/plain; charset=utf-8",
            cache_control: "no-store",
            body: message.into().into_bytes(),
            headers: Vec::new(),
        }
    }
}

pub fn serve(config: ServeConfig) -> Result<()> {
    bind(config)?.serve()
}

pub fn bind(config: ServeConfig) -> Result<BoundServer> {
    let listener = TcpListener::bind((config.host.as_str(), config.port))
        .with_context(|| format!("bind web viewer server to {}:{}", config.host, config.port))?;
    let local_addr = listener
        .local_addr()
        .context("read bound web viewer local address")?;

    let state = Arc::new(ServerState {
        payload: config.payload,
        trace: config.trace,
        broadcaster: config.broadcaster,
    });

    Ok(BoundServer {
        listener,
        local_addr,
        state,
    })
}

fn handle_connection(stream: &mut TcpStream, state: &Arc<ServerState>) -> Result<()> {
    let mut first_line = String::new();
    {
        let mut reader = BufReader::new(
            stream
                .try_clone()
                .context("clone stream for request line read")?,
        );
        reader
            .read_line(&mut first_line)
            .context("read request line")?;
    }

    if first_line.trim().is_empty() {
        return Ok(());
    }

    let first_line = first_line.trim_end();
    let mut parts = first_line.split_whitespace();
    let method = parts.next().unwrap_or_default();
    let target = parts.next().unwrap_or("/");
    let is_head = method.eq_ignore_ascii_case("HEAD");

    let path = target.split('?').next().unwrap_or("/");

    // SSE endpoint: keep connection alive and stream events.
    if path == "/api/live-events" {
        return handle_sse(stream, method, state);
    }

    let response = route_request(method, target, state);
    write_response(stream, &response, is_head)?;
    Ok(())
}

fn handle_sse(stream: &mut TcpStream, method: &str, state: &Arc<ServerState>) -> Result<()> {
    if !method.eq_ignore_ascii_case("GET") {
        let response = HttpResponse::method_not_allowed("only GET is supported\n");
        return write_response(stream, &response, false);
    }

    let Some(broadcaster) = &state.broadcaster else {
        write_sse_headers(stream)?;
        write!(
            stream,
            "event: live-unavailable\ndata: {{\"message\":\"live events are not available because no active listen session is attached\"}}\n\n"
        )
        .context("write SSE unavailable event")?;
        stream.flush().context("flush SSE unavailable event")?;
        return Ok(());
    };

    let rx = broadcaster.subscribe();

    write_sse_headers(stream)?;
    // Send a keepalive comment so the browser knows the connection is established.
    write!(stream, ": connected\n\n").context("write SSE connected comment")?;
    stream.flush().context("flush SSE headers")?;

    loop {
        match rx.recv() {
            Ok(event) => {
                let json =
                    serde_json::to_string(&event).context("serialize SSE live trace event")?;
                write!(stream, "data: {json}\n\n").context("write SSE data frame")?;
                stream.flush().context("flush SSE data frame")?;
            }
            Err(_) => break, // broadcaster dropped (listen session ended)
        }
    }

    Ok(())
}

fn write_sse_headers(stream: &mut TcpStream) -> Result<()> {
    write!(
        stream,
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: keep-alive\r\nAccess-Control-Allow-Origin: *\r\n\r\n"
    )
    .context("write SSE headers")
}

fn write_response(stream: &mut TcpStream, response: &HttpResponse, is_head: bool) -> Result<()> {
    write!(
        stream,
        "HTTP/1.1 {} {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nCache-Control: {}\r\nConnection: close\r\n",
        response.status,
        response.reason,
        response.content_type,
        response.body.len(),
        response.cache_control,
    )
    .context("write status line")?;

    for (name, value) in &response.headers {
        write!(stream, "{name}: {value}\r\n").context("write header")?;
    }
    write!(stream, "\r\n").context("write header terminator")?;
    if !is_head {
        stream.write_all(&response.body).context("write body")?;
    }
    stream.flush().context("flush response")?;
    Ok(())
}

fn route_request(method: &str, target: &str, state: &Arc<ServerState>) -> HttpResponse {
    if !method.eq_ignore_ascii_case("GET") && !method.eq_ignore_ascii_case("HEAD") {
        return HttpResponse::method_not_allowed("only GET/HEAD are supported\n");
    }

    let path = target.split('?').next().unwrap_or("/");
    match path {
        "/" => HttpResponse::ok("text/html; charset=utf-8", assets::INDEX_HTML),
        "/replay" | "/replay/" => {
            HttpResponse::ok("text/html; charset=utf-8", assets::REPLAY_HTML)
        }
        "/screenplay" | "/screenplay/" => {
            HttpResponse::ok("text/html; charset=utf-8", assets::SCREENPLAY_HTML)
        }
        "/healthz" => HttpResponse::ok("text/plain; charset=utf-8", "ok\n"),

        "/app.js" | "/assets/app.js" => {
            HttpResponse::ok("application/javascript; charset=utf-8", assets::APP_JS)
        }
        "/replay.js" | "/assets/replay.js" => {
            HttpResponse::ok("application/javascript; charset=utf-8", assets::REPLAY_JS)
        }
        "/trace-session.mjs" | "/assets/trace-session.mjs" => HttpResponse::ok(
            "application/javascript; charset=utf-8",
            assets::TRACE_SESSION_MJS,
        ),
        "/screenplay.js" | "/assets/screenplay.js" => HttpResponse::ok(
            "application/javascript; charset=utf-8",
            assets::SCREENPLAY_JS,
        ),
        "/screenplay-model.mjs" | "/assets/screenplay-model.mjs" => HttpResponse::ok(
            "application/javascript; charset=utf-8",
            assets::SCREENPLAY_MODEL_JS,
        ),
        "/scene-heading.mjs" | "/assets/scene-heading.mjs" => HttpResponse::ok(
            "application/javascript; charset=utf-8",
            assets::SCENE_HEADING_MJS,
        ),
        "/shared-span-model.mjs" | "/assets/shared-span-model.mjs" => HttpResponse::ok(
            "application/javascript; charset=utf-8",
            assets::SHARED_SPAN_MODEL_MJS,
        ),
        // Shared live-event model modules
        "/assets/shared/events/schema.mjs" => HttpResponse::ok(
            "application/javascript; charset=utf-8",
            assets::SHARED_EVENTS_SCHEMA_MJS,
        ),
        "/assets/shared/events/reducers.mjs" => HttpResponse::ok(
            "application/javascript; charset=utf-8",
            assets::SHARED_EVENTS_REDUCERS_MJS,
        ),
        "/assets/shared/events/selectors.mjs" => HttpResponse::ok(
            "application/javascript; charset=utf-8",
            assets::SHARED_EVENTS_SELECTORS_MJS,
        ),
        "/styles.css" | "/assets/styles.css" => {
            HttpResponse::ok("text/css; charset=utf-8", assets::STYLES_CSS)
        }
        "/screenplay.css" | "/assets/screenplay.css" => {
            HttpResponse::ok("text/css; charset=utf-8", assets::SCREENPLAY_CSS)
        }
        "/welcome.wav" | "/assets/welcome.wav" => {
            HttpResponse::static_asset("audio/wav", assets::WELCOME_WAV)
        }
        "/assets/index.html" => HttpResponse::ok("text/html; charset=utf-8", assets::INDEX_HTML),

        // Fixture files (organised under /fixtures/*)
        "/fixtures/demo.json" | "/api/demo-payload" => {
            HttpResponse::static_asset("application/json; charset=utf-8", assets::DEMO_JSON)
        }
        "/fixtures/live-trace.sample.jsonl"
        | "/assets/live-trace.sample.jsonl" => HttpResponse::static_asset(
            "application/x-ndjson; charset=utf-8",
            assets::LIVE_TRACE_SAMPLE_JSONL,
        ),
        "/fixtures/live-trace.sample.viewer.json"
        | "/assets/live-trace.sample.viewer.json" => HttpResponse::static_asset(
            "application/json; charset=utf-8",
            assets::LIVE_TRACE_SAMPLE_VIEWER_JSON,
        ),

        "/api/payload" => match load_payload(state) {
            Ok(Some(payload)) => HttpResponse::ok("application/json; charset=utf-8", payload),
            Ok(None) => HttpResponse::not_found("no --payload file was provided\n"),
            Err(error) => HttpResponse::internal_error(format!("{error:#}\n")),
        },
        "/api/trace" => match load_trace(state) {
            Ok(Some(trace)) => {
                HttpResponse::ok("application/x-ndjson; charset=utf-8", trace.into_bytes())
            }
            Ok(None) => HttpResponse::not_found("no --trace file was provided\n"),
            Err(error) => HttpResponse::internal_error(format!("{error:#}\n")),
        },
        "/api/trace-session" => match load_trace_session_payload(state) {
            Ok(Some(payload)) => HttpResponse::ok("application/json; charset=utf-8", payload),
            Ok(None) => HttpResponse::not_found("no --trace file was provided\n"),
            Err(error) => HttpResponse::internal_error(format!("{error:#}\n")),
        },
        "/api/trace-viewer-payload" => match load_trace_viewer_payload(state) {
            Ok(Some(payload)) => HttpResponse::ok("application/json; charset=utf-8", payload),
            Ok(None) => HttpResponse::not_found("no --trace file was provided\n"),
            Err(error) => HttpResponse::internal_error(format!("{error:#}\n")),
        },

        _ => HttpResponse::not_found("not found\n"),
    }
}

fn load_payload(state: &Arc<ServerState>) -> Result<Option<Vec<u8>>> {
    let Some(path) = state.payload.as_ref() else {
        return Ok(None);
    };
    let payload = std::fs::read(path)
        .with_context(|| format!("read viewer payload JSON from {}", path.display()))?;
    Ok(Some(payload))
}

fn load_trace(state: &Arc<ServerState>) -> Result<Option<String>> {
    let Some(path) = state.trace.as_ref() else {
        return Ok(None);
    };
    let trace = read_trace_jsonl(path)?;
    Ok(Some(trace))
}

fn load_trace_session_payload(state: &Arc<ServerState>) -> Result<Option<Vec<u8>>> {
    let Some(path) = state.trace.as_ref() else {
        return Ok(None);
    };
    let session = read_trace_session(path)?;
    let payload = serde_json::to_vec_pretty(&session)
        .with_context(|| format!("serialize trace session payload for {}", path.display()))?;
    Ok(Some(payload))
}

fn load_trace_viewer_payload(state: &Arc<ServerState>) -> Result<Option<Vec<u8>>> {
    let Some(path) = state.trace.as_ref() else {
        return Ok(None);
    };
    let session = read_trace_session(path)?;
    let payload = live_trace_events_to_viewer_payload(&session.events);
    let json = serde_json::to_vec_pretty(&payload)
        .with_context(|| format!("serialize viewer payload for {}", path.display()))?;
    Ok(Some(json))
}

#[cfg(test)]
mod tests {
    use std::io::{Read, Write};
    use std::net::{TcpListener, TcpStream};
    use std::sync::Arc;
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;
    use crate::live_trace::LiveTraceSink;

    fn empty_state() -> Arc<ServerState> {
        Arc::new(ServerState {
            payload: None,
            trace: None,
            broadcaster: None,
        })
    }

    fn live_state() -> Arc<ServerState> {
        Arc::new(ServerState {
            payload: None,
            trace: None,
            broadcaster: Some(SseBroadcaster::new()),
        })
    }

    fn temp_path(label: &str) -> PathBuf {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "listenbury-web-server-{label}-{}-{timestamp}",
            std::process::id()
        ))
    }

    #[test]
    fn serves_healthz() {
        let response = route_request("GET", "/healthz", &empty_state());
        assert_eq!(response.status, 200);
        assert_eq!(response.body, b"ok\n");
    }

    #[test]
    fn serves_demo_payload_api() {
        let response = route_request("GET", "/api/demo-payload", &empty_state());
        assert_eq!(response.status, 200);
        let body = String::from_utf8(response.body).expect("utf8");
        assert!(body.contains("\"Listenbury WaveDeck Demo\""));
    }

    #[test]
    fn serves_fixtures_under_fixtures_path() {
        let demo = route_request("GET", "/fixtures/demo.json", &empty_state());
        assert_eq!(demo.status, 200);
        let body = String::from_utf8(demo.body).expect("utf8");
        assert!(body.contains("\"Listenbury WaveDeck Demo\""));

        let jsonl = route_request("GET", "/fixtures/live-trace.sample.jsonl", &empty_state());
        assert_eq!(jsonl.status, 200);

        let viewer_json =
            route_request("GET", "/fixtures/live-trace.sample.viewer.json", &empty_state());
        assert_eq!(viewer_json.status, 200);
    }

    #[test]
    fn serves_replay_page_and_script() {
        let page = route_request("GET", "/replay", &empty_state());
        assert_eq!(page.status, 200);
        assert_eq!(page.content_type, "text/html; charset=utf-8");
        let body = String::from_utf8(page.body).expect("utf8 replay page");
        assert!(body.contains("WaveDeck") && body.contains("replay"));

        let script = route_request("GET", "/assets/replay.js", &empty_state());
        assert_eq!(script.status, 200);
        assert_eq!(script.content_type, "application/javascript; charset=utf-8");

        let trace_session = route_request("GET", "/assets/trace-session.mjs", &empty_state());
        assert_eq!(trace_session.status, 200);
        assert_eq!(
            trace_session.content_type,
            "application/javascript; charset=utf-8"
        );
    }

    #[test]
    fn payload_endpoint_requires_flag() {
        let response = route_request("GET", "/api/payload", &empty_state());
        assert_eq!(response.status, 404);
    }

    #[test]
    fn serves_provided_payload_and_trace_routes() {
        let payload_path = temp_path("payload");
        let trace_path = temp_path("trace");
        std::fs::write(&payload_path, r#"{"title":"custom"}"#).expect("write payload");
        std::fs::write(
            &trace_path,
            include_str!("../../examples/browser-transcript-player/fixtures/live-trace.sample.jsonl"),
        )
        .expect("write trace");

        let state = Arc::new(ServerState {
            payload: Some(payload_path.clone()),
            trace: Some(trace_path.clone()),
            broadcaster: None,
        });

        let payload_response = route_request("GET", "/api/payload", &state);
        assert_eq!(payload_response.status, 200);
        assert_eq!(payload_response.body, br#"{"title":"custom"}"#);

        let trace_response = route_request("GET", "/api/trace", &state);
        assert_eq!(trace_response.status, 200);
        let trace_body = String::from_utf8(trace_response.body).expect("utf8 trace");
        assert!(trace_body.contains("\"kind\":\"transcript\""));

        let trace_session_response = route_request("GET", "/api/trace-session", &state);
        assert_eq!(trace_session_response.status, 200);
        let trace_session_body =
            String::from_utf8(trace_session_response.body).expect("utf8 trace session");
        assert!(trace_session_body.contains("\"metadata\""));
        assert!(trace_session_body.contains("\"events\""));

        let viewer_payload_response = route_request("GET", "/api/trace-viewer-payload", &state);
        assert_eq!(viewer_payload_response.status, 200);
        let viewer_payload_body =
            String::from_utf8(viewer_payload_response.body).expect("utf8 viewer payload");
        assert!(viewer_payload_body.contains("\"streams\""));
        assert!(viewer_payload_body.contains("\"events\""));

        let _ = std::fs::remove_file(payload_path);
        let _ = std::fs::remove_file(trace_path);
    }

    #[test]
    fn trace_routes_accept_structured_trace_session_directory() {
        let session_root = temp_path("trace-session");
        let metadata = crate::live_trace::TraceSessionMetadata::new(
            crate::speech_timeline::SessionId::new(),
            crate::time::ExactTimestamp {
                unix_nanos: 1_000_000_000,
            },
            crate::live_trace::TraceRuntimeMetadata::new("listenbury listen"),
        );
        let mut writer =
            crate::live_trace::TraceSessionWriter::create(&session_root, metadata.clone())
                .expect("create trace session writer");
        writer
            .emit(crate::live_trace::LiveTraceEvent::new(
                metadata.session_id,
                1,
                "transcript",
                crate::time::ExactTimestamp {
                    unix_nanos: 1_250_000_000,
                },
                crate::time::ExactTimestamp {
                    unix_nanos: 1_000_000_000,
                },
            ))
            .expect("write trace event");

        let state = Arc::new(ServerState {
            payload: None,
            trace: Some(session_root.clone()),
            broadcaster: None,
        });

        let trace_response = route_request("GET", "/api/trace", &state);
        assert_eq!(trace_response.status, 200);
        let trace_body = String::from_utf8(trace_response.body).expect("utf8 trace");
        assert!(trace_body.contains("\"kind\":\"transcript\""));

        let trace_session_response = route_request("GET", "/api/trace-session", &state);
        assert_eq!(trace_session_response.status, 200);
        let trace_session_body =
            String::from_utf8(trace_session_response.body).expect("utf8 trace session");
        assert!(trace_session_body.contains("\"listenbury.live-session.v1\""));
        assert!(trace_session_body.contains("\"listenbury listen\""));

        let viewer_payload_response = route_request("GET", "/api/trace-viewer-payload", &state);
        assert_eq!(viewer_payload_response.status, 200);
        let viewer_payload_body =
            String::from_utf8(viewer_payload_response.body).expect("utf8 viewer payload");
        assert!(viewer_payload_body.contains("\"streams\""));

        let _ = std::fs::remove_dir_all(session_root);
    }

    #[test]
    fn old_demo_routes_return_404() {
        // demo.json is now served under /fixtures/demo.json, not /assets/demo.json or /demo
        let response = route_request("GET", "/assets/demo.json", &empty_state());
        assert_eq!(response.status, 404);
        let response = route_request("GET", "/demo", &empty_state());
        assert_eq!(response.status, 404);
    }

    #[test]
    fn live_server_root_serves_index_without_query() {
        let state = live_state();

        let response = route_request("GET", "/", &state);
        assert_eq!(response.status, 200);
        assert_eq!(response.content_type, "text/html; charset=utf-8");

        let app = route_request("GET", "/assets/app.js", &state);
        assert_eq!(app.status, 200);
        assert_eq!(app.content_type, "application/javascript; charset=utf-8");

        let shared_span_model = route_request("GET", "/assets/shared-span-model.mjs", &state);
        assert_eq!(shared_span_model.status, 200);
        assert_eq!(
            shared_span_model.content_type,
            "application/javascript; charset=utf-8"
        );
    }

    #[test]
    fn serves_live_screenplay_page_and_assets() {
        let state = live_state();

        let page = route_request("GET", "/screenplay", &state);
        assert_eq!(page.status, 200);
        assert_eq!(page.content_type, "text/html; charset=utf-8");
        let page_body = String::from_utf8(page.body).expect("utf8 page");
        assert!(page_body.contains("The Life of Pete Listenbury"));
        assert!(page_body.contains("by Pete Listenbury"));

        let script = route_request("GET", "/assets/screenplay.js", &state);
        assert_eq!(script.status, 200);
        assert_eq!(script.content_type, "application/javascript; charset=utf-8");

        let model = route_request("GET", "/assets/screenplay-model.mjs", &state);
        assert_eq!(model.status, 200);
        assert_eq!(model.content_type, "application/javascript; charset=utf-8");

        let scene_heading = route_request("GET", "/assets/scene-heading.mjs", &state);
        assert_eq!(scene_heading.status, 200);
        assert_eq!(scene_heading.content_type, "application/javascript; charset=utf-8");
        let shared_span_model = route_request("GET", "/assets/shared-span-model.mjs", &state);
        assert_eq!(shared_span_model.status, 200);
        assert_eq!(
            shared_span_model.content_type,
            "application/javascript; charset=utf-8"
        );

        let styles = route_request("GET", "/assets/screenplay.css", &state);
        assert_eq!(styles.status, 200);
        assert_eq!(styles.content_type, "text/css; charset=utf-8");
    }

    #[test]
    fn serves_shared_live_events_modules() {
        let state = live_state();

        let schema = route_request("GET", "/assets/shared/events/schema.mjs", &state);
        assert_eq!(schema.status, 200);
        assert_eq!(schema.content_type, "application/javascript; charset=utf-8");

        let reducers = route_request("GET", "/assets/shared/events/reducers.mjs", &state);
        assert_eq!(reducers.status, 200);
        assert_eq!(reducers.content_type, "application/javascript; charset=utf-8");

        let selectors = route_request("GET", "/assets/shared/events/selectors.mjs", &state);
        assert_eq!(selectors.status, 200);
        assert_eq!(selectors.content_type, "application/javascript; charset=utf-8");
    }

    #[test]
    fn live_server_serves_index_with_unrelated_query_param() {
        let response = route_request("GET", "/?foo=bar", &live_state());
        assert_eq!(response.status, 200);
        assert_eq!(response.content_type, "text/html; charset=utf-8");
    }

    #[test]
    fn live_events_returns_404_without_broadcaster() {
        let response = route_request("GET", "/api/live-events", &empty_state());
        // route_request doesn't handle /api/live-events (that's handled in handle_connection)
        // so it falls through to not_found
        assert_eq!(response.status, 404);
    }

    #[test]
    fn live_events_connection_returns_sse_unavailable_without_broadcaster() {
        let listener = TcpListener::bind(("127.0.0.1", 0)).expect("bind test listener");
        let addr = listener.local_addr().expect("local addr");
        let state = empty_state();

        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept test connection");
            handle_connection(&mut stream, &state).expect("handle test request");
        });

        let mut client = TcpStream::connect(addr).expect("connect to test server");
        client
            .write_all(b"GET /api/live-events HTTP/1.1\r\nHost: localhost\r\n\r\n")
            .expect("write request");

        let mut response = String::new();
        client.read_to_string(&mut response).expect("read response");
        server.join().expect("server thread");

        assert!(response.starts_with("HTTP/1.1 200 OK"));
        assert!(response.contains("Content-Type: text/event-stream"));
        assert!(response.contains("event: live-unavailable"));
        assert!(response.contains("no active listen session"));
    }

    #[test]
    fn bind_reports_port_in_use_before_serving() {
        let listener = TcpListener::bind(("127.0.0.1", 0)).expect("bind occupied test port");
        let addr = listener.local_addr().expect("occupied local addr");

        let error = bind(ServeConfig {
            host: "127.0.0.1".to_string(),
            port: addr.port(),
            payload: None,
            trace: None,
            broadcaster: None,
        })
        .expect_err("second bind should fail");

        assert!(
            format!("{error:#}").contains("bind web viewer server to 127.0.0.1"),
            "unexpected bind error: {error:#}"
        );
    }
}
