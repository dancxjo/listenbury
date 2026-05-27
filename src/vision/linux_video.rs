use std::io::Read;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};

use anyhow::{Context, Result};

use crate::memory::{MemoryImageVector, MemorySink, MemoryTrace};
use crate::time::ExactTimestamp;
use crate::vision::{VisionFrame, vectorize_rgba_frame};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LinuxVideoCaptureConfig {
    pub device: PathBuf,
    pub width: u32,
    pub height: u32,
    pub fps: u32,
    pub retain_image: bool,
    pub content_node_id: Option<String>,
}

impl Default for LinuxVideoCaptureConfig {
    fn default() -> Self {
        Self {
            device: PathBuf::from("/dev/video0"),
            width: 320,
            height: 240,
            fps: 2,
            retain_image: false,
            content_node_id: None,
        }
    }
}

pub struct NativeVideoCaptureHandle {
    stop: Arc<AtomicBool>,
    child: Arc<Mutex<Child>>,
    join: Option<JoinHandle<()>>,
}

impl NativeVideoCaptureHandle {
    pub fn stop(mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Ok(mut child) = self.child.lock() {
            let _ = child.kill();
        }
        if let Some(join) = self.join.take() {
            let _ = join.join();
        }
    }
}

impl Drop for NativeVideoCaptureHandle {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Ok(mut child) = self.child.lock() {
            let _ = child.kill();
        }
        if let Some(join) = self.join.take() {
            let _ = join.join();
        }
    }
}

pub fn ffmpeg_linux_video_args(config: &LinuxVideoCaptureConfig) -> Vec<String> {
    vec![
        "-hide_banner".to_string(),
        "-loglevel".to_string(),
        "error".to_string(),
        "-f".to_string(),
        "v4l2".to_string(),
        "-video_size".to_string(),
        format!("{}x{}", config.width, config.height),
        "-framerate".to_string(),
        config.fps.to_string(),
        "-i".to_string(),
        config.device.display().to_string(),
        "-pix_fmt".to_string(),
        "rgba".to_string(),
        "-f".to_string(),
        "rawvideo".to_string(),
        "-".to_string(),
    ]
}

pub fn spawn_linux_video_vector_capture(
    config: LinuxVideoCaptureConfig,
    memory_sink: Arc<dyn MemorySink>,
) -> Result<NativeVideoCaptureHandle> {
    let mut child = Command::new("ffmpeg")
        .args(ffmpeg_linux_video_args(&config))
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .with_context(|| {
            format!(
                "spawn ffmpeg for native Linux video capture from {}",
                config.device.display()
            )
        })?;
    let stdout = child.stdout.take().context("capture ffmpeg stdout")?;
    let child = Arc::new(Mutex::new(child));
    let stop = Arc::new(AtomicBool::new(false));
    let thread_stop = Arc::clone(&stop);
    let thread_child = Arc::clone(&child);
    let join = thread::spawn(move || {
        read_rgba_frames(stdout, config, memory_sink, thread_stop);
        if let Ok(mut child) = thread_child.lock() {
            let _ = child.kill();
            let _ = child.wait();
        }
    });

    Ok(NativeVideoCaptureHandle {
        stop,
        child,
        join: Some(join),
    })
}

fn read_rgba_frames(
    mut stdout: impl Read,
    config: LinuxVideoCaptureConfig,
    memory_sink: Arc<dyn MemorySink>,
    stop: Arc<AtomicBool>,
) {
    let Some(frame_len) = usize::try_from(config.width)
        .ok()
        .and_then(|width| {
            usize::try_from(config.height)
                .ok()
                .and_then(|height| width.checked_mul(height))
        })
        .and_then(|pixels| pixels.checked_mul(4))
    else {
        return;
    };
    let mut bytes = vec![0_u8; frame_len];
    while !stop.load(Ordering::Relaxed) {
        if stdout.read_exact(&mut bytes).is_err() {
            break;
        }
        let captured_at = ExactTimestamp::now();
        let frame = VisionFrame {
            captured_at,
            width: config.width,
            height: config.height,
            bytes: bytes.clone(),
        };
        let Some(observation) = vectorize_rgba_frame(&frame) else {
            continue;
        };
        memory_sink.submit(MemoryTrace::ImageVectorCaptured {
            image: MemoryImageVector {
                image_id: observation.image_id,
                source: format!("linux_v4l2:{}", config.device.display()),
                width: config.width,
                height: config.height,
                vector: observation.vector,
                content_node_id: config.content_node_id.clone(),
                retained_image: config.retain_image,
            },
            captured_at,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ffmpeg_args_capture_v4l2_as_rgba_rawvideo() {
        let config = LinuxVideoCaptureConfig {
            device: PathBuf::from("/dev/video2"),
            width: 640,
            height: 480,
            fps: 5,
            retain_image: false,
            content_node_id: None,
        };

        let args = ffmpeg_linux_video_args(&config);

        assert!(args.windows(2).any(|window| window == ["-f", "v4l2"]));
        assert!(args.windows(2).any(|window| window == ["-pix_fmt", "rgba"]));
        assert!(
            args.windows(2)
                .any(|window| window == ["-video_size", "640x480"])
        );
        assert!(args.iter().any(|arg| arg == "/dev/video2"));
        assert_eq!(args.last().map(String::as_str), Some("-"));
    }
}
