const CALLBACK_QUEUE_SECONDS: usize = 8;
const MIN_CALLBACK_SAMPLE_CAPACITY: usize = 16_384;
const MAX_CALLBACK_SAMPLE_CAPACITY: usize = 2_000_000;

pub fn callback_sample_queue_capacity(sample_rate_hz: u32, channels: u16) -> usize {
    let samples_per_second = usize::try_from(sample_rate_hz)
        .unwrap_or(usize::MAX)
        .saturating_mul(usize::from(channels).max(1));
    samples_per_second
        .saturating_mul(CALLBACK_QUEUE_SECONDS)
        .clamp(MIN_CALLBACK_SAMPLE_CAPACITY, MAX_CALLBACK_SAMPLE_CAPACITY)
}

pub fn boost_current_thread_for_capture(label: &str) {
    match try_boost_current_thread_for_capture() {
        Ok(priority) => eprintln!("{label}: capture thread priority set to {priority}"),
        Err(error) => eprintln!("{label}: capture thread priority unchanged ({error})"),
    }
}

#[cfg(target_os = "linux")]
fn try_boost_current_thread_for_capture() -> Result<&'static str, String> {
    let nice_result = unsafe {
        let tid = libc::syscall(libc::SYS_gettid) as libc::id_t;
        libc::setpriority(libc::PRIO_PROCESS, tid, -10)
    };
    if nice_result == 0 {
        return Ok("nice -10");
    }

    Err(format!(
        "nice -10 failed: {}",
        std::io::Error::last_os_error()
    ))
}

#[cfg(not(target_os = "linux"))]
fn try_boost_current_thread_for_capture() -> Result<&'static str, String> {
    Err("no capture priority boost is implemented for this OS".to_string())
}

#[cfg(test)]
mod tests {
    use super::callback_sample_queue_capacity;

    #[test]
    fn callback_queue_scales_with_capture_format() {
        assert_eq!(callback_sample_queue_capacity(48_000, 2), 768_000);
    }

    #[test]
    fn callback_queue_keeps_a_floor_for_invalid_formats() {
        assert_eq!(callback_sample_queue_capacity(0, 0), 16_384);
    }
}
