use std::io::{BufRead, BufReader, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};

use anyhow::{Context, Result};
use serde_json::json;

use crate::audio::{
    AcousticAnalysis, AudioFrame, analyze_audio_frames, read_wav_frames,
    segment_pronunciation_with_acoustics, write_wav_bytes,
};
use crate::live_trace::{LiveTraceEvent, LiveTraceSink};
use crate::live_trace::{SseBroadcaster, read_trace_jsonl, read_trace_session};
use crate::time::{ExactTimestamp, SessionClock};
use crate::trace::viewer_payload::{ViewerPayload, trace_session_to_viewer_payload};
use crate::vision::{
    VisionFrame, VisualProvenance, VisualSpeechFrame, extract_visual_speech_frame_from_rgba,
};

use super::assets;

#[derive(Debug, Clone)]
pub struct ServeConfig {
    pub host: String,
    pub port: u16,
    pub payload: Option<PathBuf>,
    pub trace: Option<PathBuf>,
    pub broadcaster: Option<SseBroadcaster>,
    pub live_audio: Option<LiveSessionAudioStore>,
    pub live_visual_speech: Option<LiveSessionVisualSpeechStore>,
    pub input_control: WebInputControl,
}

#[derive(Debug, Clone)]
struct ServerState {
    payload: Option<PathBuf>,
    trace: Option<PathBuf>,
    broadcaster: Option<SseBroadcaster>,
    live_audio: Option<LiveSessionAudioStore>,
    live_visual_speech: Option<LiveSessionVisualSpeechStore>,
    input_control: WebInputControl,
}

#[derive(Clone, Debug)]
pub struct InputRouter {
    clock: SessionClock,
    arbitration_lock: Arc<Mutex<()>>,
    native_capture_enabled: Option<Arc<AtomicBool>>,
    browser_audio_enabled: Arc<AtomicBool>,
    browser_audio_tx: Option<crossbeam_channel::Sender<AudioFrame>>,
    browser_visual_speech_tx: Option<crossbeam_channel::Sender<VisualSpeechFrame>>,
}

pub type WebInputControl = InputRouter;

impl Default for InputRouter {
    fn default() -> Self {
        Self {
            clock: SessionClock::start_now(),
            arbitration_lock: Arc::new(Mutex::new(())),
            native_capture_enabled: None,
            browser_audio_enabled: Arc::new(AtomicBool::new(true)),
            browser_audio_tx: None,
            browser_visual_speech_tx: None,
        }
    }
}

impl InputRouter {
    pub fn new(
        native_capture_enabled: Option<Arc<AtomicBool>>,
        browser_audio_tx: Option<crossbeam_channel::Sender<AudioFrame>>,
    ) -> Self {
        Self {
            clock: SessionClock::start_now(),
            arbitration_lock: Arc::new(Mutex::new(())),
            native_capture_enabled,
            browser_audio_enabled: Arc::new(AtomicBool::new(true)),
            browser_audio_tx,
            browser_visual_speech_tx: None,
        }
    }

    pub fn with_visual_speech_tx(
        mut self,
        browser_visual_speech_tx: Option<crossbeam_channel::Sender<VisualSpeechFrame>>,
    ) -> Self {
        self.browser_visual_speech_tx = browser_visual_speech_tx;
        self
    }

    fn now(&self) -> ExactTimestamp {
        self.clock.now()
    }

    fn normalize_elapsed_ms(&self, at: ExactTimestamp) -> u64 {
        self.clock.elapsed_ms(at)
    }

    fn browser_audio_enabled(&self) -> bool {
        self.browser_audio_enabled.load(Ordering::Relaxed)
    }

    fn native_mic_available(&self) -> bool {
        self.native_capture_enabled.is_some()
    }

    fn browser_mic_available(&self, live_audio: Option<&LiveSessionAudioStore>) -> bool {
        self.browser_audio_tx.is_some() || live_audio.is_some()
    }

    fn set_browser_audio_enabled(&self, enabled: bool) {
        let _guard = self
            .arbitration_lock
            .lock()
            .expect("input router mutex poisoned");
        self.browser_audio_enabled.store(enabled, Ordering::Relaxed);
        if enabled && let Some(native_capture_enabled) = self.native_capture_enabled.as_ref() {
            native_capture_enabled.store(false, Ordering::Relaxed);
        }
    }

    fn set_native_capture_enabled(&self, enabled: bool) -> bool {
        let Some(native_capture_enabled) = self.native_capture_enabled.as_ref() else {
            return false;
        };
        let _guard = self
            .arbitration_lock
            .lock()
            .expect("input router mutex poisoned");
        native_capture_enabled.store(enabled, Ordering::Relaxed);
        if enabled {
            self.browser_audio_enabled.store(false, Ordering::Relaxed);
        }
        true
    }
}

#[derive(Clone, Debug, Default)]
pub struct LiveSessionVisualSpeechStore {
    frames: Arc<Mutex<Vec<VisualSpeechFrame>>>,
}

impl LiveSessionVisualSpeechStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push_frame(&self, frame: VisualSpeechFrame) {
        match self.frames.lock() {
            Ok(mut frames) => frames.push(frame),
            Err(error) => {
                tracing::error!("live visual-speech mutex poisoned; dropping frame: {error}")
            }
        }
    }

    fn snapshot(&self) -> Result<Vec<VisualSpeechFrame>> {
        let frames = self
            .frames
            .lock()
            .map_err(|error| anyhow::anyhow!("live visual-speech mutex poisoned: {error}"))?
            .clone();
        Ok(frames)
    }
}

#[derive(Clone, Debug, Default)]
pub struct LiveSessionAudioStore {
    frames: Arc<Mutex<Vec<AudioFrame>>>,
    acoustic: Arc<Mutex<Option<CachedAcousticAnalysis>>>,
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
        if let Ok(mut acoustic) = self.acoustic.lock() {
            *acoustic = None;
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

    fn acoustic_analysis(&self) -> Result<Option<AcousticAnalysis>> {
        let frames = self.snapshot()?;
        if frames.is_empty() {
            return Ok(None);
        }
        let frame_count = frames.len();
        let mut cached = self
            .acoustic
            .lock()
            .map_err(|error| anyhow::anyhow!("live acoustic mutex poisoned: {error}"))?;
        if let Some(cached) = cached
            .as_ref()
            .filter(|cached| cached.frame_count == frame_count)
        {
            return Ok(Some(cached.analysis.clone()));
        }
        let Some(analysis) = analyze_audio_frames(&frames) else {
            return Ok(None);
        };
        *cached = Some(CachedAcousticAnalysis {
            frame_count,
            analysis: analysis.clone(),
        });
        Ok(Some(analysis))
    }
}

#[derive(Clone, Debug)]
struct CachedAcousticAnalysis {
    frame_count: usize,
    analysis: AcousticAnalysis,
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
            "Routes: /, /wavedeck, /replay, /screenplay, /assets/*, /fixtures/*, /api/*, /api/trace-session, /api/live-session-audio.wav, /api/live-session-acoustic.json, /api/live-visual-speech.json, /api/session-audio/*, /api/live-events, /api/browser-audio, /api/browser-video-frame, /api/input-status, /api/native-mic, /api/browser-mic, /healthz"
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

    fn accepted(message: impl Into<String>) -> Self {
        Self {
            status: 202,
            reason: "Accepted",
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
            headers: vec![("Allow", "GET, HEAD, POST".to_string())],
        }
    }

    fn bad_request(message: impl Into<String>) -> Self {
        Self {
            status: 400,
            reason: "Bad Request",
            content_type: "text/plain; charset=utf-8",
            cache_control: "no-store",
            body: message.into().into_bytes(),
            headers: Vec::new(),
        }
    }

    fn service_unavailable(message: impl Into<String>) -> Self {
        Self {
            status: 503,
            reason: "Service Unavailable",
            content_type: "text/plain; charset=utf-8",
            cache_control: "no-store",
            body: message.into().into_bytes(),
            headers: Vec::new(),
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
        live_visual_speech: config.live_visual_speech,
        input_control: config.input_control,
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
    let mut body = Vec::<u8>::new();
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
        if let Some(content_length) = request_header(&headers, "content-length") {
            let content_length = content_length
                .parse::<usize>()
                .context("parse Content-Length header")?;
            body.resize(content_length, 0);
            reader.read_exact(&mut body).context("read request body")?;
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
    let response =
        route_request_with_range_and_body(method, target, state, range_header, &headers, &body);
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

    while let Ok(event) = rx.recv() {
        let json = serde_json::to_string(&event).context("serialize SSE live trace event")?;
        write!(stream, "data: {json}\n\n").context("write SSE data frame")?;
        stream.flush().context("flush SSE data frame")?;
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

#[cfg(test)]
fn route_request_with_range(
    method: &str,
    target: &str,
    state: &Arc<ServerState>,
    range_header: Option<&str>,
) -> HttpResponse {
    route_request_with_range_and_body(method, target, state, range_header, &[], &[])
}

fn route_request_with_range_and_body(
    method: &str,
    target: &str,
    state: &Arc<ServerState>,
    range_header: Option<&str>,
    headers: &[(String, String)],
    body: &[u8],
) -> HttpResponse {
    let path = target.split('?').next().unwrap_or("/");
    if path == "/api/input-status" {
        if !method.eq_ignore_ascii_case("GET") && !method.eq_ignore_ascii_case("HEAD") {
            return HttpResponse::method_not_allowed("only GET/HEAD are supported\n");
        }
        return HttpResponse::ok("application/json; charset=utf-8", input_status_json(state));
    }
    if path == "/api/native-mic" {
        if !method.eq_ignore_ascii_case("POST") {
            return HttpResponse::method_not_allowed("only POST is supported\n");
        }
        return update_native_mic(state, body);
    }
    if path == "/api/browser-mic" {
        if !method.eq_ignore_ascii_case("POST") {
            return HttpResponse::method_not_allowed("only POST is supported\n");
        }
        return update_browser_mic(state, body);
    }
    if path == "/api/browser-audio" {
        if !method.eq_ignore_ascii_case("POST") {
            return HttpResponse::method_not_allowed("only POST is supported\n");
        }
        return receive_browser_audio(state, headers, body);
    }
    if path == "/api/browser-video-frame" {
        if !method.eq_ignore_ascii_case("POST") {
            return HttpResponse::method_not_allowed("only POST is supported\n");
        }
        return receive_browser_video_frame(state, headers, body);
    }

    if !method.eq_ignore_ascii_case("GET") && !method.eq_ignore_ascii_case("HEAD") {
        return HttpResponse::method_not_allowed("only GET/HEAD are supported\n");
    }

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
        "/spectrogram.mjs" | "/assets/spectrogram.mjs" => HttpResponse::ok(
            "application/javascript; charset=utf-8",
            assets::SPECTROGRAM_MJS,
        ),
        "/phoneme-projection.mjs" | "/assets/phoneme-projection.mjs" => HttpResponse::ok(
            "application/javascript; charset=utf-8",
            assets::PHONEME_PROJECTION_MJS,
        ),
        "/mechanical-asr.mjs" | "/assets/mechanical-asr.mjs" => HttpResponse::ok(
            "application/javascript; charset=utf-8",
            assets::MECHANICAL_ASR_MJS,
        ),
        "/viterbi-phone-alignment.mjs" | "/assets/viterbi-phone-alignment.mjs" => HttpResponse::ok(
            "application/javascript; charset=utf-8",
            assets::VITERBI_PHONE_ALIGNMENT_MJS,
        ),
        "/hypothesis-lattice.mjs" | "/assets/hypothesis-lattice.mjs" => HttpResponse::ok(
            "application/javascript; charset=utf-8",
            assets::HYPOTHESIS_LATTICE_MJS,
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
            Ok(None) if state.live_audio.is_some() => {
                HttpResponse::accepted("live session audio is not available yet\n")
            }
            Ok(None) => HttpResponse::not_found("live session audio is not available yet\n"),
            Err(error) => HttpResponse::internal_error(format!("{error:#}\n")),
        },
        "/api/live-session-acoustic.json" => match load_live_session_acoustic(state) {
            Ok(Some(analysis)) => HttpResponse::ok("application/json; charset=utf-8", analysis),
            Ok(None) if state.live_audio.is_some() => {
                HttpResponse::accepted("live session acoustic analysis is not available yet\n")
            }
            Ok(None) => {
                HttpResponse::not_found("live session acoustic analysis is not available yet\n")
            }
            Err(error) => HttpResponse::internal_error(format!("{error:#}\n")),
        },
        "/api/live-visual-speech.json" => match load_live_visual_speech(state) {
            Ok(Some(trace)) => HttpResponse::ok("application/json; charset=utf-8", trace),
            Ok(None) if state.live_visual_speech.is_some() => {
                HttpResponse::accepted("live visual speech is not available yet\n")
            }
            Ok(None) => HttpResponse::not_found("live visual speech is not available yet\n"),
            Err(error) => HttpResponse::internal_error(format!("{error:#}\n")),
        },

        _ if path.starts_with("/api/session-audio/") && path.ends_with("/acoustic.json") => {
            match load_session_acoustic(path, state) {
                Ok(Some(analysis)) => HttpResponse::ok("application/json; charset=utf-8", analysis),
                Ok(None) => HttpResponse::not_found("session acoustic analysis not found\n"),
                Err(error) => HttpResponse::internal_error(format!("{error:#}\n")),
            }
        }
        _ if path.starts_with("/api/session-audio/") => match load_session_audio(path, state) {
            Ok(Some(audio)) => audio_response(audio, range_header),
            Ok(None) => HttpResponse::not_found("session audio artifact not found\n"),
            Err(error) => HttpResponse::internal_error(format!("{error:#}\n")),
        },
        _ => HttpResponse::not_found("not found\n"),
    }
}

fn input_status_json(state: &Arc<ServerState>) -> Vec<u8> {
    let native_available = state.input_control.native_mic_available();
    let native_enabled = state
        .input_control
        .native_capture_enabled
        .as_ref()
        .is_some_and(|enabled| enabled.load(Ordering::Relaxed));
    let browser_available = state
        .input_control
        .browser_mic_available(state.live_audio.as_ref());
    let browser_enabled = state.input_control.browser_audio_enabled();
    let browser_camera_available = state.input_control.browser_visual_speech_tx.is_some()
        || state.live_visual_speech.is_some();
    format!(
        "{{\"nativeMic\":{{\"available\":{},\"enabled\":{}}},\"browserMic\":{{\"available\":{},\"enabled\":{}}},\"browserCamera\":{{\"available\":{}}}}}\n",
        native_available, native_enabled, browser_available, browser_enabled, browser_camera_available
    )
    .into_bytes()
}

fn update_native_mic(state: &Arc<ServerState>, body: &[u8]) -> HttpResponse {
    if !state.input_control.native_mic_available() {
        return HttpResponse::not_found("native microphone control is not available\n");
    }
    let payload: serde_json::Value = match serde_json::from_slice(body) {
        Ok(payload) => payload,
        Err(error) => return HttpResponse::bad_request(format!("invalid JSON body: {error}\n")),
    };
    let Some(enabled) = payload.get("enabled").and_then(serde_json::Value::as_bool) else {
        return HttpResponse::bad_request("expected JSON body with boolean field enabled\n");
    };
    state.input_control.set_native_capture_enabled(enabled);
    HttpResponse::ok("application/json; charset=utf-8", input_status_json(state))
}

fn update_browser_mic(state: &Arc<ServerState>, body: &[u8]) -> HttpResponse {
    if !state
        .input_control
        .browser_mic_available(state.live_audio.as_ref())
    {
        return HttpResponse::not_found("browser microphone control is not available\n");
    }
    let payload: serde_json::Value = match serde_json::from_slice(body) {
        Ok(payload) => payload,
        Err(error) => return HttpResponse::bad_request(format!("invalid JSON body: {error}\n")),
    };
    let Some(enabled) = payload.get("enabled").and_then(serde_json::Value::as_bool) else {
        return HttpResponse::bad_request("expected JSON body with boolean field enabled\n");
    };
    state.input_control.set_browser_audio_enabled(enabled);
    HttpResponse::ok("application/json; charset=utf-8", input_status_json(state))
}

fn receive_browser_audio(
    state: &Arc<ServerState>,
    headers: &[(String, String)],
    body: &[u8],
) -> HttpResponse {
    let routed_at = state.input_control.now();
    if !state.input_control.browser_audio_enabled() {
        return HttpResponse::accepted(
            "browser audio ignored because browser microphone is disabled\n",
        );
    }
    if body.is_empty() {
        return HttpResponse::bad_request("audio body must not be empty\n");
    }
    if !body.len().is_multiple_of(std::mem::size_of::<f32>()) {
        return HttpResponse::bad_request("audio body must contain little-endian f32 samples\n");
    }
    if body.len() > 512 * 1024 {
        return HttpResponse::bad_request("audio chunk is too large\n");
    }
    let sample_rate_hz = match request_header(headers, "x-sample-rate-hz")
        .unwrap_or("16000")
        .parse::<u32>()
    {
        Ok(sample_rate_hz) if (8_000..=192_000).contains(&sample_rate_hz) => sample_rate_hz,
        _ => return HttpResponse::bad_request("invalid X-Sample-Rate-Hz header\n"),
    };
    let channels = match request_header(headers, "x-channels")
        .unwrap_or("1")
        .parse::<u16>()
    {
        Ok(channels) if (1..=2).contains(&channels) => channels,
        _ => return HttpResponse::bad_request("invalid X-Channels header\n"),
    };

    let samples = body
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]).clamp(-1.0, 1.0))
        .collect::<Vec<_>>();
    let frame = AudioFrame {
        captured_at: routed_at,
        sample_rate_hz,
        channels,
        samples,
        voice_signatures: Vec::new(),
    };

    if let Some(tx) = &state.input_control.browser_audio_tx {
        match tx.try_send(frame) {
            Ok(()) => HttpResponse::accepted("browser audio accepted\n"),
            Err(crossbeam_channel::TrySendError::Full(_)) => {
                HttpResponse::service_unavailable("browser audio queue is full\n")
            }
            Err(crossbeam_channel::TrySendError::Disconnected(_)) => {
                HttpResponse::service_unavailable("browser audio processor is disconnected\n")
            }
        }
    } else if let Some(live_audio) = &state.live_audio {
        live_audio.push_frame(frame);
        HttpResponse::accepted("browser audio recorded\n")
    } else {
        HttpResponse::not_found("browser audio input is not available\n")
    }
}

fn receive_browser_video_frame(
    state: &Arc<ServerState>,
    headers: &[(String, String)],
    body: &[u8],
) -> HttpResponse {
    let routed_at = state.input_control.now();
    if body.is_empty() {
        return HttpResponse::bad_request("video frame body must not be empty\n");
    }
    if body.len() > 2 * 1024 * 1024 {
        return HttpResponse::bad_request("video frame is too large\n");
    }
    let width = match request_header(headers, "x-width").and_then(|value| value.parse::<u32>().ok())
    {
        Some(width) if (16..=1920).contains(&width) => width,
        _ => return HttpResponse::bad_request("invalid X-Width header\n"),
    };
    let height =
        match request_header(headers, "x-height").and_then(|value| value.parse::<u32>().ok()) {
            Some(height) if (16..=1080).contains(&height) => height,
            _ => return HttpResponse::bad_request("invalid X-Height header\n"),
        };
    if !request_header(headers, "x-pixel-format")
        .unwrap_or("rgba8")
        .eq_ignore_ascii_case("rgba8")
    {
        return HttpResponse::bad_request("only X-Pixel-Format: rgba8 is supported\n");
    }
    let expected_len = usize::try_from(width)
        .unwrap_or(usize::MAX)
        .saturating_mul(usize::try_from(height).unwrap_or(usize::MAX))
        .saturating_mul(4);
    if body.len() != expected_len {
        return HttpResponse::bad_request("rgba8 body length does not match width*height*4\n");
    }

    let start_ms = request_header(headers, "x-capture-elapsed-ms")
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(0);
    let duration_ms = request_header(headers, "x-frame-duration-ms")
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|duration| (1..=250).contains(duration))
        .unwrap_or(67);
    let frame_index =
        request_header(headers, "x-frame-index").and_then(|value| value.parse::<u64>().ok());
    let capture_time_ms =
        request_header(headers, "x-capture-time-ms").and_then(|value| value.parse::<u64>().ok());
    let audio_offset_ms = request_header(headers, "x-audio-offset-ms")
        .and_then(|value| value.parse::<i64>().ok())
        .unwrap_or(0);

    let vision_frame = VisionFrame {
        captured_at: routed_at,
        width,
        height,
        bytes: body.to_vec(),
    };
    let provenance = VisualProvenance {
        source: "browser_camera.backend_rgba".to_string(),
        frame_index,
        capture_time_ms,
        audio_offset_ms: Some(audio_offset_ms),
        notes: vec![
            "browser transported transient rgba8 frame; backend extracted derived features"
                .to_string(),
        ],
    };
    let Some(feature_frame) = extract_visual_speech_frame_from_rgba(
        &vision_frame,
        start_ms..start_ms.saturating_add(duration_ms),
        provenance,
    ) else {
        return HttpResponse::bad_request("unable to extract visual speech features from frame\n");
    };

    if let Some(tx) = &state.input_control.browser_visual_speech_tx
        && let Err(error) = tx.try_send(feature_frame.clone())
    {
        return match error {
            crossbeam_channel::TrySendError::Full(_) => {
                HttpResponse::service_unavailable("visual speech queue is full\n")
            }
            crossbeam_channel::TrySendError::Disconnected(_) => {
                HttpResponse::service_unavailable("visual speech processor is disconnected\n")
            }
        };
    }
    if let Some(store) = &state.live_visual_speech {
        store.push_frame(feature_frame.clone());
    }
    broadcast_visual_speech_frame(state, &feature_frame);
    HttpResponse::accepted("browser video frame accepted; derived features stored\n")
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
    let mut payload = trace_session_to_viewer_payload(&session);
    if let Some(analysis) = load_primary_trace_session_acoustic_analysis(path, &session)? {
        enrich_payload_phone_segmentations(&mut payload, &analysis);
    }
    let json = serde_json::to_vec_pretty(&payload)
        .with_context(|| format!("serialize viewer payload for {}", path.display()))?;
    Ok(Some(json))
}

fn enrich_payload_phone_segmentations(payload: &mut ViewerPayload, analysis: &AcousticAnalysis) {
    for lane in &mut payload.streams {
        for word in &mut lane.stream.words {
            let Some(timing) = word.timing else {
                continue;
            };
            let Some(pronunciation) = word.pronunciation.as_mut() else {
                continue;
            };
            pronunciation.phone_segmentation = segment_pronunciation_with_acoustics(
                &word.text,
                timing.start_ms,
                timing.end_ms,
                pronunciation,
                analysis,
            );
        }
    }
}

fn load_primary_trace_session_acoustic_analysis(
    trace_path: &Path,
    session: &crate::live_trace::TraceSessionEnvelope,
) -> Result<Option<AcousticAnalysis>> {
    let Some(artifact) = session.metadata.audio_artifacts.first() else {
        return Ok(None);
    };
    let Some(acoustic_path) = artifact.acoustic_analysis_path.as_ref() else {
        return Ok(None);
    };
    let base_dir = trace_path
        .parent()
        .filter(|_| {
            trace_path
                .file_name()
                .is_some_and(|name| name == "metadata.json")
        })
        .unwrap_or(trace_path);
    let analysis_path = base_dir.join(acoustic_path);
    let bytes = std::fs::read(&analysis_path).with_context(|| {
        format!(
            "read trace-session acoustic analysis for payload enrichment {}",
            analysis_path.display()
        )
    })?;
    let analysis: AcousticAnalysis = serde_json::from_slice(&bytes).with_context(|| {
        format!(
            "deserialize trace-session acoustic analysis {}",
            analysis_path.display()
        )
    })?;
    Ok(Some(analysis))
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

fn load_live_session_acoustic(state: &Arc<ServerState>) -> Result<Option<Vec<u8>>> {
    let Some(store) = state.live_audio.as_ref() else {
        return Ok(None);
    };
    let Some(analysis) = store.acoustic_analysis()? else {
        return Ok(None);
    };
    let json = serde_json::to_vec(&analysis).context("serialize live session acoustic analysis")?;
    Ok(Some(json))
}

fn load_live_visual_speech(state: &Arc<ServerState>) -> Result<Option<Vec<u8>>> {
    let Some(store) = state.live_visual_speech.as_ref() else {
        return Ok(None);
    };
    let frames = store.snapshot()?;
    if frames.is_empty() {
        return Ok(None);
    }
    let json = serde_json::to_vec(&frames).context("serialize live visual speech frames")?;
    Ok(Some(json))
}

fn broadcast_visual_speech_frame(state: &Arc<ServerState>, frame: &VisualSpeechFrame) {
    let Some(broadcaster) = &state.broadcaster else {
        return;
    };
    let now = state.input_control.now();
    let normalized_elapsed_ms = state.input_control.normalize_elapsed_ms(now);
    let mut event = LiveTraceEvent {
        turn: 0,
        soundscape_id: None,
        voice_id: None,
        voice_label: None,
        voice_attributions: Vec::new(),
        session_id: None,
        turn_id: None,
        utterance_id: None,
        speech_unit_id: None,
        transcript_revision_id: None,
        span_id: None,
        audio_clip_id: None,
        kind: "visual_speech_frame".to_string(),
        source: Some("browser.camera".to_string()),
        emitter_id: Some("browser.camera".to_string()),
        seq_id: None,
        t_unix_ns: now.unix_nanos.min(u128::from(u64::MAX)) as u64,
        elapsed_ms: frame.time_range_ms.start,
        normalized_elapsed_ms: Some(normalized_elapsed_ms),
        normalized_unix_ns: Some(now.unix_nanos.min(u128::from(u64::MAX)) as u64),
        observed_elapsed_ms: None,
        observed_unix_ns: None,
        ts_local_monotonic: Some(frame.time_range_ms.start),
        ts_wall_approx_unix_ns: Some(now.unix_nanos.min(u128::from(u64::MAX)) as u64),
        clock_sync: None,
        cause: None,
        text: Some("derived_mouth_motion_features".to_string()),
        confidence: Some(frame.visibility.weighted()),
        group_id: None,
        reason: None,
        face: None,
        unit_kind: None,
        expected_until_unix_ns: None,
        artifact: Some(json!(frame)),
        runtime_event: None,
    };
    if frame.visibility.value < 0.35 {
        event.reason = Some("low_mouth_visibility".to_string());
    }
    let mut broadcaster = broadcaster.clone();
    if let Err(error) = broadcaster.emit(event) {
        tracing::error!("failed to broadcast visual speech frame: {error:#}");
    }
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

fn load_session_acoustic(path: &str, state: &Arc<ServerState>) -> Result<Option<Vec<u8>>> {
    let Some(trace_path) = state.trace.as_ref() else {
        return Ok(None);
    };
    let artifact_id = path
        .trim_start_matches("/api/session-audio/")
        .trim_end_matches("/acoustic.json")
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
    if let Some(path) = artifact.acoustic_analysis_path.as_ref() {
        let analysis_path = base_dir.join(path);
        let analysis = std::fs::read(&analysis_path).with_context(|| {
            format!("read session acoustic analysis {}", analysis_path.display())
        })?;
        return Ok(Some(analysis));
    }

    let audio_path = base_dir.join(&artifact.path);
    let frames = read_wav_frames(&audio_path, 1_600).with_context(|| {
        format!(
            "read session audio for acoustic analysis {}",
            audio_path.display()
        )
    })?;
    let Some(analysis) = analyze_audio_frames(&frames) else {
        return Ok(None);
    };
    let json = serde_json::to_vec(&analysis).context("serialize session acoustic analysis")?;
    Ok(Some(json))
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
    use crate::trace::viewer_payload::{ViewerPayload, ViewerWordLane};
    use crate::word::{
        BoundarySource, PronunciationLookupStatus, TimedWordStream, WordCommitment, WordId,
        WordNode, WordPronunciation, WordStreamId, WordStreamSource, WordTiming,
    };

    fn empty_state() -> Arc<ServerState> {
        Arc::new(ServerState {
            payload: None,
            trace: None,
            broadcaster: None,
            live_audio: None,
            live_visual_speech: None,
            input_control: WebInputControl::default(),
        })
    }

    fn live_state() -> Arc<ServerState> {
        Arc::new(ServerState {
            payload: None,
            trace: None,
            broadcaster: Some(SseBroadcaster::new()),
            live_audio: None,
            live_visual_speech: None,
            input_control: WebInputControl::default(),
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

        let spectrogram_module = route_request("GET", "/assets/spectrogram.mjs", &empty_state());
        assert_eq!(spectrogram_module.status, 200);
        assert_eq!(
            spectrogram_module.content_type,
            "application/javascript; charset=utf-8"
        );

        let mechanical_asr = route_request("GET", "/assets/mechanical-asr.mjs", &empty_state());
        assert_eq!(mechanical_asr.status, 200);
        assert_eq!(
            mechanical_asr.content_type,
            "application/javascript; charset=utf-8"
        );
        let body = String::from_utf8(mechanical_asr.body).expect("utf8");
        assert!(body.contains("boundaryHypothesesFromLandmarks"));

        let viterbi_module =
            route_request("GET", "/assets/viterbi-phone-alignment.mjs", &empty_state());
        assert_eq!(viterbi_module.status, 200);
        assert_eq!(
            viterbi_module.content_type,
            "application/javascript; charset=utf-8"
        );
        let body = String::from_utf8(viterbi_module.body).expect("utf8");
        assert!(body.contains("phoneSpansFromHypotheses"));
        assert!(body.contains("drawViterbiPhoneSpans"));

        let hypothesis_lattice =
            route_request("GET", "/assets/hypothesis-lattice.mjs", &empty_state());
        assert_eq!(hypothesis_lattice.status, 200);
        assert_eq!(
            hypothesis_lattice.content_type,
            "application/javascript; charset=utf-8"
        );
        let body = String::from_utf8(hypothesis_lattice.body).expect("utf8");
        assert!(body.contains("HypothesisLattice"));
        assert!(body.contains("fuseHypotheses"));
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
            live_visual_speech: None,
            input_control: WebInputControl::default(),
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
    fn enriches_payload_words_with_backend_phone_segmentation() {
        let sample_rate_hz = 16_000;
        let samples = (0..6400)
            .map(|index| {
                ((2.0 * std::f32::consts::PI * 220.0 * index as f32) / sample_rate_hz as f32).sin()
            })
            .collect::<Vec<_>>();
        let analysis = crate::audio::analyze_mono_samples(&samples, sample_rate_hz);
        let mut payload = ViewerPayload {
            title: "test".to_string(),
            audio: None,
            streams: vec![ViewerWordLane {
                label: "User transcript".to_string(),
                stream: TimedWordStream {
                    id: WordStreamId(1),
                    source: WordStreamSource::RecordedAudio,
                    words: vec![WordNode {
                        id: WordId(1),
                        text: "three".to_string(),
                        lexical_span: None,
                        timing: Some(WordTiming {
                            start_ms: 1000,
                            end_ms: 1300,
                        }),
                        timing_confidence: Some(0.9),
                        commitment: WordCommitment::Final,
                        boundary_source: BoundarySource::Whisper,
                        audio_ref: None,
                        pronunciation: Some(WordPronunciation {
                            source: "cmudict".to_string(),
                            lookup: "THREE".to_string(),
                            phonemes: vec!["TH".to_string(), "R".to_string(), "IY1".to_string()],
                            stress_pattern: "1".to_string(),
                            status: PronunciationLookupStatus::Exact,
                            phone_segmentation: None,
                        }),
                    }],
                },
            }],
            span_graph: None,
            events: Vec::new(),
            markers: Vec::new(),
        };

        enrich_payload_phone_segmentations(&mut payload, &analysis);

        let segmentation = payload.streams[0].stream.words[0]
            .pronunciation
            .as_ref()
            .and_then(|pron| pron.phone_segmentation.as_ref())
            .expect("backend phone segmentation should be attached");
        assert_eq!(segmentation.phone_spans.len(), 3);
        assert_eq!(segmentation.phone_spans[0].start_ms, 1000);
        assert_eq!(segmentation.phone_spans[2].end_ms, 1300);
    }

    #[test]
    fn serves_session_audio_artifact_from_trace_session_metadata() {
        let root = temp_path("session-audio");
        let audio_dir = root.join("audio");
        std::fs::create_dir_all(&audio_dir).expect("create audio dir");
        std::fs::write(audio_dir.join("session.wav"), b"RIFFtest").expect("write audio");
        std::fs::write(
            audio_dir.join("session.acoustic.json"),
            br#"{"spectrogram":{"levels":[]},"energyEnvelope":{"frames":[]}}"#,
        )
        .expect("write acoustic analysis");

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
                acoustic_analysis_path: Some("audio/session.acoustic.json".to_string()),
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
            live_visual_speech: None,
            input_control: WebInputControl::default(),
        });
        let response = route_request("GET", "/api/session-audio/session-audio", &state);
        assert_eq!(response.status, 200);
        assert_eq!(response.content_type, "audio/wav");
        assert_eq!(response.body, b"RIFFtest");

        let acoustic_response = route_request(
            "GET",
            "/api/session-audio/session-audio/acoustic.json",
            &state,
        );
        assert_eq!(acoustic_response.status, 200);
        assert_eq!(
            acoustic_response.content_type,
            "application/json; charset=utf-8"
        );
        let acoustic_body =
            String::from_utf8(acoustic_response.body).expect("utf8 acoustic analysis");
        assert!(acoustic_body.contains("\"spectrogram\""));
        assert!(acoustic_body.contains("\"energyEnvelope\""));

        std::fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn serves_live_session_acoustic_analysis_from_audio_store() {
        let store = LiveSessionAudioStore::new();
        let sample_rate_hz = 16_000;
        let samples = (0..1600)
            .map(|index| {
                ((2.0 * std::f32::consts::PI * 440.0 * index as f32) / sample_rate_hz as f32).sin()
            })
            .collect::<Vec<_>>();
        store.push_frame(crate::audio::AudioFrame {
            captured_at: crate::time::ExactTimestamp { unix_nanos: 0 },
            sample_rate_hz,
            channels: 1,
            samples,
            voice_signatures: Vec::new(),
        });
        let state = Arc::new(ServerState {
            payload: None,
            trace: None,
            broadcaster: None,
            live_audio: Some(store),
            live_visual_speech: None,
            input_control: WebInputControl::default(),
        });

        let response = route_request("GET", "/api/live-session-acoustic.json", &state);
        assert_eq!(response.status, 200);
        assert_eq!(response.content_type, "application/json; charset=utf-8");
        let body = String::from_utf8(response.body).expect("utf8 acoustic analysis");
        assert!(body.contains("\"spectrogram\""));
        assert!(body.contains("\"energyEnvelope\""));
        assert!(body.contains("\"energyLandmarks\""));
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
            live_visual_speech: None,
            input_control: WebInputControl::default(),
        });

        let response = route_request("GET", "/api/live-session-audio.wav", &state);
        assert_eq!(response.status, 200);
        assert_eq!(response.content_type, "audio/wav");
        assert!(response.body.starts_with(b"RIFF"));
        assert!(response.body.windows(4).any(|window| window == b"data"));
    }

    #[test]
    fn browser_audio_post_appends_to_live_audio_store() {
        let live_audio = LiveSessionAudioStore::new();
        let state = Arc::new(ServerState {
            payload: None,
            trace: None,
            broadcaster: None,
            live_audio: Some(live_audio),
            live_visual_speech: None,
            input_control: WebInputControl::default(),
        });
        let headers = vec![
            ("x-sample-rate-hz".to_string(), "16000".to_string()),
            ("x-channels".to_string(), "1".to_string()),
        ];
        let mut body = Vec::new();
        for sample in [0.0_f32, 0.5, -0.5] {
            body.extend_from_slice(&sample.to_le_bytes());
        }

        let response = route_request_with_range_and_body(
            "POST",
            "/api/browser-audio",
            &state,
            None,
            &headers,
            &body,
        );
        assert_eq!(response.status, 202);

        let audio_response = route_request("GET", "/api/live-session-audio.wav", &state);
        assert_eq!(audio_response.status, 200);
        assert!(audio_response.body.starts_with(b"RIFF"));
    }

    #[test]
    fn browser_video_frame_post_stores_derived_visual_features_only() {
        let live_visual_speech = LiveSessionVisualSpeechStore::new();
        let state = Arc::new(ServerState {
            payload: None,
            trace: None,
            broadcaster: None,
            live_audio: None,
            live_visual_speech: Some(live_visual_speech),
            input_control: WebInputControl::default(),
        });
        let width = 40usize;
        let height = 40usize;
        let mut body = vec![180_u8; width * height * 4];
        for px in body.chunks_exact_mut(4) {
            px[3] = 255;
        }
        for y in 24..30 {
            for x in 16..24 {
                let i = (y * width + x) * 4;
                body[i] = 10;
                body[i + 1] = 10;
                body[i + 2] = 10;
            }
        }
        let headers = vec![
            ("x-pixel-format".to_string(), "rgba8".to_string()),
            ("x-width".to_string(), width.to_string()),
            ("x-height".to_string(), height.to_string()),
            ("x-capture-elapsed-ms".to_string(), "1200".to_string()),
            ("x-frame-duration-ms".to_string(), "33".to_string()),
            ("x-frame-index".to_string(), "7".to_string()),
        ];

        let response = route_request_with_range_and_body(
            "POST",
            "/api/browser-video-frame",
            &state,
            None,
            &headers,
            &body,
        );
        assert_eq!(response.status, 202);

        let visual_response = route_request("GET", "/api/live-visual-speech.json", &state);
        assert_eq!(visual_response.status, 200);
        let visual_json = String::from_utf8(visual_response.body).expect("utf8");
        assert!(visual_json.contains("\"timeRangeMs\":[1200,1233]"));
        assert!(visual_json.contains("\"lipClosure\""));
        assert!(!visual_json.contains("rawVideo"));
        assert!(!visual_json.contains(&format!("{body:?}")));
    }

    #[test]
    fn native_mic_control_updates_shared_capture_flag() {
        let enabled = Arc::new(AtomicBool::new(true));
        let state = Arc::new(ServerState {
            payload: None,
            trace: None,
            broadcaster: None,
            live_audio: None,
            live_visual_speech: None,
            input_control: WebInputControl::new(Some(Arc::clone(&enabled)), None),
        });

        let response = route_request_with_range_and_body(
            "POST",
            "/api/native-mic",
            &state,
            None,
            &[],
            br#"{"enabled":false}"#,
        );

        assert_eq!(response.status, 200);
        assert!(!enabled.load(Ordering::Relaxed));
        let body = String::from_utf8(response.body).expect("utf8 input status");
        assert!(body.contains("\"enabled\":false"));
    }

    #[test]
    fn browser_and_native_mic_controls_arbitrate_active_capture_source() {
        let enabled = Arc::new(AtomicBool::new(true));
        let live_audio = LiveSessionAudioStore::new();
        let state = Arc::new(ServerState {
            payload: None,
            trace: None,
            broadcaster: None,
            live_audio: Some(live_audio),
            live_visual_speech: None,
            input_control: WebInputControl::new(Some(Arc::clone(&enabled)), None),
        });

        let browser_enable = route_request_with_range_and_body(
            "POST",
            "/api/browser-mic",
            &state,
            None,
            &[],
            br#"{"enabled":true}"#,
        );
        assert_eq!(browser_enable.status, 200);
        assert!(!enabled.load(Ordering::Relaxed));
        let browser_status = String::from_utf8(browser_enable.body).expect("utf8 input status");
        assert!(browser_status.contains("\"browserMic\":{\"available\":true,\"enabled\":true}"));

        let native_enable = route_request_with_range_and_body(
            "POST",
            "/api/native-mic",
            &state,
            None,
            &[],
            br#"{"enabled":true}"#,
        );
        assert_eq!(native_enable.status, 200);
        assert!(enabled.load(Ordering::Relaxed));
        let native_status = String::from_utf8(native_enable.body).expect("utf8 input status");
        assert!(native_status.contains("\"nativeMic\":{\"available\":true,\"enabled\":true}"));
        assert!(native_status.contains("\"browserMic\":{\"available\":true,\"enabled\":false}"));
    }

    #[test]
    fn marks_empty_live_session_audio_store_as_pending() {
        let live_audio = LiveSessionAudioStore::new();
        let state = Arc::new(ServerState {
            payload: None,
            trace: None,
            broadcaster: None,
            live_audio: Some(live_audio),
            live_visual_speech: None,
            input_control: WebInputControl::default(),
        });

        let audio_response = route_request("GET", "/api/live-session-audio.wav", &state);
        assert_eq!(audio_response.status, 202);
        assert_eq!(audio_response.reason, "Accepted");

        let acoustic_response = route_request("GET", "/api/live-session-acoustic.json", &state);
        assert_eq!(acoustic_response.status, 202);
        assert_eq!(acoustic_response.reason, "Accepted");
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
            live_visual_speech: None,
            input_control: WebInputControl::default(),
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
            live_visual_speech: None,
            input_control: WebInputControl::default(),
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
            live_visual_speech: None,
            input_control: WebInputControl::default(),
        })
        .expect_err("second bind should fail");

        assert!(
            format!("{error:#}").contains("bind web viewer server to 127.0.0.1"),
            "unexpected bind error: {error:#}"
        );
    }
}
