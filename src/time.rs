use std::time::{Instant, SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ExactTimestamp {
    pub unix_nanos: u128,
}

impl ExactTimestamp {
    pub fn now() -> Self {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock is set before Unix epoch")
            .as_nanos();
        Self { unix_nanos: nanos }
    }
}

#[derive(Debug, Clone)]
pub struct Timed<T> {
    pub at: Instant,
    pub body: T,
}

impl<T> Timed<T> {
    pub fn now(body: T) -> Self {
        Self {
            at: Instant::now(),
            body,
        }
    }
}
