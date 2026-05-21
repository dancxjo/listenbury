use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use chrono::{DateTime, TimeZone, Utc};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ExactTimestamp {
    pub unix_nanos: u128,
}

impl Serialize for ExactTimestamp {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let secs = (self.unix_nanos / 1_000_000_000) as i64;
        let sub_nanos = (self.unix_nanos % 1_000_000_000) as u32;
        let dt = Utc
            .timestamp_opt(secs, sub_nanos)
            .single()
            .ok_or_else(|| serde::ser::Error::custom("timestamp out of representable range"))?;
        serializer.serialize_str(&dt.to_rfc3339())
    }
}

impl<'de> Deserialize<'de> for ExactTimestamp {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let s = String::deserialize(deserializer)?;
        let dt = DateTime::parse_from_rfc3339(&s).map_err(serde::de::Error::custom)?;
        let utc: DateTime<Utc> = dt.into();
        let nanos =
            (utc.timestamp() as i128) * 1_000_000_000 + utc.timestamp_subsec_nanos() as i128;
        if nanos < 0 {
            return Err(serde::de::Error::custom(
                "timestamp is before the Unix epoch (1970-01-01)",
            ));
        }
        Ok(ExactTimestamp {
            unix_nanos: nanos as u128,
        })
    }
}

impl ExactTimestamp {
    pub fn from_unix_nanos(unix_nanos: u128) -> Self {
        Self { unix_nanos }
    }

    pub fn now() -> Self {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock is set before Unix epoch")
            .as_nanos();
        Self { unix_nanos: nanos }
    }

    pub fn saturating_add(self, duration: Duration) -> Self {
        Self {
            unix_nanos: self.unix_nanos.saturating_add(duration.as_nanos()),
        }
    }
}

pub trait Clock: Send + Sync {
    fn now(&self) -> ExactTimestamp;
}

#[derive(Debug, Clone, Copy, Default)]
pub struct SystemClock;

impl Clock for SystemClock {
    fn now(&self) -> ExactTimestamp {
        ExactTimestamp::now()
    }
}

#[derive(Debug, Clone)]
pub struct FakeClock {
    now: Arc<Mutex<ExactTimestamp>>,
}

impl FakeClock {
    pub fn new(now: ExactTimestamp) -> Self {
        Self {
            now: Arc::new(Mutex::new(now)),
        }
    }

    pub fn from_unix_nanos(unix_nanos: u128) -> Self {
        Self::new(ExactTimestamp::from_unix_nanos(unix_nanos))
    }

    pub fn set(&self, now: ExactTimestamp) {
        *self.now.lock().expect("fake clock mutex poisoned") = now;
    }

    pub fn advance(&self, duration: Duration) -> ExactTimestamp {
        let mut now = self.now.lock().expect("fake clock mutex poisoned");
        *now = now.saturating_add(duration);
        *now
    }
}

impl Clock for FakeClock {
    fn now(&self) -> ExactTimestamp {
        *self.now.lock().expect("fake clock mutex poisoned")
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NormalizedTimestamp {
    pub unix_ns: u64,
    pub elapsed_ms: u64,
}

#[derive(Debug, Clone)]
pub struct SessionClock {
    session_started_at: ExactTimestamp,
    monotonic_origin: Instant,
}

impl SessionClock {
    pub fn start_now() -> Self {
        Self {
            session_started_at: ExactTimestamp::now(),
            monotonic_origin: Instant::now(),
        }
    }

    pub fn with_session_start(session_started_at: ExactTimestamp) -> Self {
        Self {
            session_started_at,
            monotonic_origin: Instant::now(),
        }
    }

    pub fn session_started_at(&self) -> ExactTimestamp {
        self.session_started_at
    }

    pub fn now(&self) -> ExactTimestamp {
        self.at_elapsed(self.monotonic_origin.elapsed())
    }

    pub fn at_elapsed_ms(&self, elapsed_ms: u64) -> ExactTimestamp {
        self.at_elapsed(Duration::from_millis(elapsed_ms))
    }

    pub fn at_elapsed(&self, elapsed: Duration) -> ExactTimestamp {
        ExactTimestamp {
            unix_nanos: self
                .session_started_at
                .unix_nanos
                .saturating_add(elapsed.as_nanos()),
        }
    }

    pub fn elapsed_ms(&self, at: ExactTimestamp) -> u64 {
        at.unix_nanos
            .saturating_sub(self.session_started_at.unix_nanos)
            .saturating_div(1_000_000)
            .min(u128::from(u64::MAX)) as u64
    }

    pub fn normalize(&self, at: ExactTimestamp) -> NormalizedTimestamp {
        NormalizedTimestamp {
            unix_ns: at.unix_nanos.min(u128::from(u64::MAX)) as u64,
            elapsed_ms: self.elapsed_ms(at),
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn session_clock_normalizes_elapsed_time_from_session_start() {
        let session_started_at = ExactTimestamp {
            unix_nanos: 10_000_000_000,
        };
        let clock = SessionClock::with_session_start(session_started_at);
        let at = clock.at_elapsed_ms(225);
        let normalized = clock.normalize(at);
        assert_eq!(normalized.unix_ns, 10_225_000_000);
        assert_eq!(normalized.elapsed_ms, 225);
    }

    #[test]
    fn fake_clock_returns_controlled_timestamps_without_sleeping() {
        let clock = FakeClock::from_unix_nanos(1_000);
        assert_eq!(clock.now(), ExactTimestamp::from_unix_nanos(1_000));

        clock.advance(Duration::from_millis(25));
        assert_eq!(clock.now(), ExactTimestamp::from_unix_nanos(25_001_000));

        clock.set(ExactTimestamp::from_unix_nanos(42));
        assert_eq!(clock.now(), ExactTimestamp::from_unix_nanos(42));
    }
}
