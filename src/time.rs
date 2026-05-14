use std::time::Instant;

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
