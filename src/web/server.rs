use std::io::{BufRead, BufReader, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};

use crate::audio::{AudioFrame, write_wav_bytes};
use crate::live_trace::{SseBroadcaster, read_trace_jsonl, read_trace_session};
use crate::trace::viewer_payload::trace_session_to_viewer_payload;

use super::assets;

#[derive(Debug, Clone)]
pub struct ServeConfig {
    pub host: String,
    pub port: u16,
    pub payload: Option<PathBuf>,
    pub trace: Option<PathBuf>,
    pub broadcaster: Option<SseBroadcaster>,
    pub live_audio: Option<LiveSessionAudioStore>,
}

#[derive(Debug, Clone)]
struct ServerState {
    payload: Option<PathBuf>,
    trace: Option<PathBuf>,
    broadcaster: Option<SseBroadcaster>,
    live_audio: Option<LiveSessionAudioStore>,
}

#[derive(Clone, Debug, Default)]
pub struct LiveSessionAudioStore {
    frames: Arc<Mutex<Vec<AudioFrame>>>,
}

impl LiveSessionAudioStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push_frame(&self, frame: AudioFrame) {
        match self.frames.lock() {
            Ok(mut frames) => frames.push(frame),
            Err(error) => tracing::error!("live audio mutex poisoned; dropping frame: {error}"),
        }
    }

    fn snapshot(&self) -> Result<Vec<AudioFrame>> {
        let frames = self
            .frames
            .lock()
            .map_err(|error| anyhow::anyhow!("live audio mutex poisoned: {error}"))?
            .clone();
        Ok(frames)
    }
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
        println!(
            "Routes: /, /wavedeck, /replay, /screenplay, /assets/*, /fixtures/*, /api/*, /api/trace-session, /api/live-session-audio.wav, /api/session-audio/*, /api/live-events, /healthz"
        );

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

    fn range_not_satisfiable(total_len: usize) -> Self {
        Self {
            status: 416,
            reason: "Range Not Satisfiable",
            content_type: "text/plain; charset=utf-8",
            cache_control: "no-store",
            body: b"requested range not satisfiable\n".to_vec(),
            headers: vec![
                ("Accept-Ranges", "bytes".to_string()),
                ("Content-Range", format!("bytes */{total_len}")),
            ],
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ByteRange {
    start: usize,
    end: usize,
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
        live_audio: config.live_audio,
    });

    Ok(BoundServer {
        listener,
        local_addr,
        state,
    })
}

fn handle_connection(stream: &mut TcpStream, state: &Arc<ServerState>) -> Result<()> {
    let mut first_line = String::new();
    let mut headers = Vec::<(String, String)>::new();
    {
        let mut reader = BufReader::new(
            stream
                .try_clone()
                .context("clone stream for request line read")?,
        );
        reader
            .read_line(&mut first_line)
            .context("read request line")?;
        loop {
            let mut line = String::new();
            reader.read_line(&mut line).context("read request header")?;
            let trimmed = line.trim_end();
            if trimmed.is_empty() {
                break;
            }
            if let Some((name, value)) = trimmed.split_once(':') {
                headers.push((name.trim().to_ascii_lowercase(), value.trim().to_string()));
            }
        }
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

    let range_header = request_header(&headers, "range");
    let response = route_request_with_range(method, target, state, range_header);
    write_response(stream, &response, is_head)?;
    Ok(())
}

fn request_header<'a>(headers: &'a [(String, String)], name: &str) -> Option<&'a str> {
    headers
        .iter()
        .find(|(header_name, _)| header_name.eq_ignore_ascii_case(name))
        .map(|(_, value)| value.as_str())
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

#[cfg(test)]
fn route_request(method: &str, target: &str, state: &Arc<ServerState>) -> HttpResponse {
    route_request_with_range(method, target, state, None)
}

fn route_request_with_range(
    method: &str,
    target: &str,
    state: &Arc<ServerState>,
    range_header: Option<&str>,
) -> HttpResponse {
    if !method.eq_ignore_ascii_case("GET") && !method.eq_ignore_ascii_case("HEAD") {
        return HttpResponse::method_not_allowed("only GET/HEAD are supported\n");
    }

    let path = target.split('?').next().unwrap_or("/");
    match path {
        "/" | "/wavedeck" | "/wavedeck/" => {
            HttpResponse::ok("text/html; charset=utf-8", assets::INDEX_HTML)
        }
        "/replay" | "/replay/" => HttpResponse::ok("text/html; charset=utf-8", assets::REPLAY_HTML),
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
        "/timeline-viewport.mjs" | "/assets/timeline-viewport.mjs" => HttpResponse::ok(
            "application/javascript; charset=utf-8",
            assets::TIMELINE_VIEWPORT_MJS,
        ),
        "/energy-timing.mjs" | "/assets/energy-timing.mjs" => HttpResponse::ok(
            "application/javascript; charset=utf-8",
            assets::ENERGY_TIMING_MJS,
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
        "/welcome.wav" | "/assets/welcome.wav" => audio_response(assets::WELCOME_WAV, range_header),
        "/assets/index.html" => HttpResponse::ok("text/html; charset=utf-8", assets::INDEX_HTML),

        // Fixture files (organised under /fixtures/*)
        "/fixtures/demo.json" | "/api/demo-payload" => {
            HttpResponse::static_asset("application/json; charset=utf-8", assets::DEMO_JSON)
        }
        "/fixtures/live-trace.sample.jsonl" | "/assets/live-trace.sample.jsonl" => {
            HttpResponse::static_asset(
                "application/x-ndjson; charset=utf-8",
                assets::LIVE_TRACE_SAMPLE_JSONL,
            )
        }
        "/fixtures/live-trace.sample.viewer.json" | "/assets/live-trace.sample.viewer.json" => {
            HttpResponse::static_asset(
                "application/json; charset=utf-8",
                assets::LIVE_TRACE_SAMPLE_VIEWER_JSON,
            )
        }

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
        "/api/live-session-audio.wav" => match load_live_session_audio(state) {
            Ok(Some(audio)) => audio_response(audio, range_header),
            Ok(None) => HttpResponse::not_found("live session audio is not available yet\n"),
            Err(error) => HttpResponse::internal_error(format!("{error:#}\n")),
        },

        _ if path.starts_with("/api/session-audio/") => match load_session_audio(path, state) {
            Ok(Some(audio)) => audio_response(audio, range_header),
            Ok(None) => HttpResponse::not_found("session audio artifact not found\n"),
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
    let payload = trace_session_to_viewer_payload(&session);
    let json = serde_json::to_vec_pretty(&payload)
        .with_context(|| format!("serialize viewer payload for {}", path.display()))?;
    Ok(Some(json))
}

fn load_live_session_audio(state: &Arc<ServerState>) -> Result<Option<Vec<u8>>> {
    let Some(store) = state.live_audio.as_ref() else {
        return Ok(None);
    };
    let frames = store.snapshot()?;
    if frames.is_empty() {
        return Ok(None);
    }
    let audio = write_wav_bytes(&frames).context("encode live session audio as WAV")?;
    Ok(Some(audio))
}

fn load_session_audio(path: &str, state: &Arc<ServerState>) -> Result<Option<Vec<u8>>> {
    let Some(trace_path) = state.trace.as_ref() else {
        return Ok(None);
    };
    let artifact_id = path
        .trim_start_matches("/api/session-audio/")
        .split('/')
        .next()
        .unwrap_or_default();
    if artifact_id.is_empty() || artifact_id.contains("..") {
        return Ok(None);
    }

    let session = read_trace_session(trace_path)?;
    let Some(artifact) = session
        .metadata
        .audio_artifacts
        .iter()
        .find(|artifact| artifact.artifact_id == artifact_id)
    else {
        return Ok(None);
    };
    let base_dir = trace_path
        .parent()
        .filter(|_| {
            trace_path
                .file_name()
                .is_some_and(|name| name == "metadata.json")
        })
        .unwrap_or(trace_path.as_path());
    let audio_path = base_dir.join(&artifact.path);
    let audio = std::fs::read(&audio_path)
        .with_context(|| format!("read session audio artifact {}", audio_path.display()))?;
    Ok(Some(audio))
}

fn audio_response(audio: impl Into<Vec<u8>>, range_header: Option<&str>) -> HttpResponse {
    let audio = audio.into();
    let total_len = audio.len();
    let mut response = match parse_byte_range(range_header, total_len) {
        Ok(Some(range)) => {
            let body = audio[range.start..=range.end].to_vec();
            HttpResponse {
                status: 206,
                reason: "Partial Content",
                content_type: "audio/wav",
                cache_control: "no-store",
                body,
                headers: vec![(
                    "Content-Range",
                    format!("bytes {}-{}/{}", range.start, range.end, total_len),
                )],
            }
        }
        Ok(None) => HttpResponse::ok("audio/wav", audio),
        Err(()) => return HttpResponse::range_not_satisfiable(total_len),
    };
    response
        .headers
        .push(("Accept-Ranges", "bytes".to_string()));
    response
}

fn parse_byte_range(
    range_header: Option<&str>,
    total_len: usize,
) -> std::result::Result<Option<ByteRange>, ()> {
    let Some(range_header) = range_header else {
        return Ok(None);
    };
    if total_len == 0 {
        return Err(());
    }

    let Some(spec) = range_header.trim().strip_prefix("bytes=") else {
        return Err(());
    };
    if spec.contains(',') {
        return Err(());
    }
    let Some((start_text, end_text)) = spec.split_once('-') else {
        return Err(());
    };

    if start_text.is_empty() {
        let suffix_len = end_text.parse::<usize>().map_err(|_| ())?;
        if suffix_len == 0 {
            return Err(());
        }
        let start = total_len.saturating_sub(suffix_len);
        return Ok(Some(ByteRange {
            start,
            end: total_len - 1,
        }));
    }

    let start = start_text.parse::<usize>().map_err(|_| ())?;
    if start >= total_len {
        return Err(());
    }
    let end = if end_text.is_empty() {
        total_len - 1
    } else {
        end_text
            .parse::<usize>()
            .map_err(|_| ())?
            .min(total_len - 1)
    };
    if end < start {
        return Err(());
    }
    Ok(Some(ByteRange { start, end }))
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
            live_audio: None,
        })
    }

    fn live_state() -> Arc<ServerState> {
        Arc::new(ServerState {
            payload: None,
            trace: None,
            broadcaster: Some(SseBroadcaster::new()),
            live_audio: None,
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

        let viewer_json = route_request(
            "GET",
            "/fixtures/live-trace.sample.viewer.json",
            &empty_state(),
        );
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
    fn serves_wavedeck_route_and_timeline_viewport_module() {
        let page = route_request("GET", "/wavedeck", &empty_state());
        assert_eq!(page.status, 200);
        assert_eq!(page.content_type, "text/html; charset=utf-8");
        let body = String::from_utf8(page.body).expect("utf8 wavedeck page");
        assert!(body.contains("WaveDeck"));

        let module = route_request("GET", "/assets/timeline-viewport.mjs", &empty_state());
        assert_eq!(module.status, 200);
        assert_eq!(module.content_type, "application/javascript; charset=utf-8");

        let energy_module = route_request("GET", "/assets/energy-timing.mjs", &empty_state());
        assert_eq!(energy_module.status, 200);
        assert_eq!(
            energy_module.content_type,
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
            include_str!(
                "../../examples/browser-transcript-player/fixtures/live-trace.sample.jsonl"
            ),
        )
        .expect("write trace");

        let state = Arc::new(ServerState {
            payload: Some(payload_path.clone()),
            trace: Some(trace_path.clone()),
            broadcaster: None,
            live_audio: None,
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
    fn serves_session_audio_artifact_from_trace_session_metadata() {
        let root = temp_path("session-audio");
        let audio_dir = root.join("audio");
        std::fs::create_dir_all(&audio_dir).expect("create audio dir");
        std::fs::write(audio_dir.join("session.wav"), b"RIFFtest").expect("write audio");

        let session_id = crate::live_trace::SessionId::new();
        let mut metadata = crate::live_trace::TraceSessionMetadata::new(
            session_id,
            crate::time::ExactTimestamp {
                unix_nanos: 1_000_000_000,
            },
            crate::live_trace::TraceRuntimeMetadata::new("test"),
        );
        metadata
            .audio_artifacts
            .push(crate::live_trace::TraceSessionAudioArtifact {
                session_id,
                artifact_id: "session-audio".to_string(),
                path: "audio/session.wav".to_string(),
                duration_ms: 100,
                sample_rate_hz: 16_000,
                channels: 1,
                created_at_unix_ns: 1_000_000_000,
            });
        std::fs::write(
            root.join(crate::live_trace::TRACE_SESSION_METADATA_FILE),
            serde_json::to_vec_pretty(&metadata).expect("metadata json"),
        )
        .expect("write metadata");
        std::fs::write(root.join(crate::live_trace::TRACE_SESSION_EVENTS_FILE), "")
            .expect("write events");

        let state = Arc::new(ServerState {
            payload: None,
            trace: Some(root.clone()),
            broadcaster: None,
            live_audio: None,
        });
        let response = route_request("GET", "/api/session-audio/session-audio", &state);
        assert_eq!(response.status, 200);
        assert_eq!(response.content_type, "audio/wav");
        assert_eq!(response.body, b"RIFFtest");

        std::fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn serves_live_session_audio_from_shared_store() {
        let live_audio = LiveSessionAudioStore::new();
        live_audio.push_frame(AudioFrame {
            captured_at: crate::time::ExactTimestamp { unix_nanos: 0 },
            sample_rate_hz: 16_000,
            channels: 1,
            samples: vec![0.0, 0.25, -0.25],
            voice_signatures: Vec::new(),
        });
        let state = Arc::new(ServerState {
            payload: None,
            trace: None,
            broadcaster: None,
            live_audio: Some(live_audio),
        });

        let response = route_request("GET", "/api/live-session-audio.wav", &state);
        assert_eq!(response.status, 200);
        assert_eq!(response.content_type, "audio/wav");
        assert!(response.body.starts_with(b"RIFF"));
        assert!(response.body.windows(4).any(|window| window == b"data"));
    }

    #[test]
    fn serves_live_session_audio_byte_ranges() {
        let live_audio = LiveSessionAudioStore::new();
        live_audio.push_frame(AudioFrame {
            captured_at: crate::time::ExactTimestamp { unix_nanos: 0 },
            sample_rate_hz: 16_000,
            channels: 1,
            samples: vec![0.0; 100],
            voice_signatures: Vec::new(),
        });
        let state = Arc::new(ServerState {
            payload: None,
            trace: None,
            broadcaster: None,
            live_audio: Some(live_audio),
        });

        let full = route_request("GET", "/api/live-session-audio.wav", &state);
        let ranged = route_request_with_range(
            "GET",
            "/api/live-session-audio.wav",
            &state,
            Some("bytes=4-11"),
        );

        assert_eq!(ranged.status, 206);
        assert_eq!(ranged.content_type, "audio/wav");
        assert_eq!(ranged.body, full.body[4..=11]);
        assert!(
            ranged
                .headers
                .iter()
                .any(|(name, value)| { *name == "Accept-Ranges" && value == "bytes" })
        );
        assert!(
            ranged
                .headers
                .iter()
                .any(|(name, value)| { *name == "Content-Range" && value == "bytes 4-11/244" })
        );
    }

    #[test]
    fn rejects_unsatisfiable_audio_byte_ranges() {
        let response = route_request_with_range(
            "GET",
            "/welcome.wav",
            &empty_state(),
            Some("bytes=999999999-1000000000"),
        );
        assert_eq!(response.status, 416);
        assert!(
            response
                .headers
                .iter()
                .any(|(name, value)| { *name == "Content-Range" && value.starts_with("bytes */") })
        );
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
            live_audio: None,
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
        assert_eq!(
            scene_heading.content_type,
            "application/javascript; charset=utf-8"
        );
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
        assert_eq!(
            reducers.content_type,
            "application/javascript; charset=utf-8"
        );

        let selectors = route_request("GET", "/assets/shared/events/selectors.mjs", &state);
        assert_eq!(selectors.status, 200);
        assert_eq!(
            selectors.content_type,
            "application/javascript; charset=utf-8"
        );
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
            live_audio: None,
        })
        .expect_err("second bind should fail");

        assert!(
            format!("{error:#}").contains("bind web viewer server to 127.0.0.1"),
            "unexpected bind error: {error:#}"
        );
    }
}
