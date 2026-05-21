use serde::{Deserialize, Serialize};

/// Millisecond offset used by soundscape frames and events.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct TimePoint {
    pub millis: u64,
}

impl TimePoint {
    pub fn from_millis(millis: u64) -> Self {
        Self { millis }
    }
}

/// Inclusive time range for a soundscape frame or event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TimeRange {
    pub start: TimePoint,
    pub end: TimePoint,
}

impl TimeRange {
    /// Creates an inclusive range with `start <= end`.
    pub fn new(start: TimePoint, end: TimePoint) -> Self {
        debug_assert!(start <= end, "time range start must be <= end");
        Self { start, end }
    }

    pub fn duration_millis(self) -> u64 {
        self.end.millis.saturating_sub(self.start.millis)
    }
}
