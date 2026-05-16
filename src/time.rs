use std::time::{Instant, SystemTime, UNIX_EPOCH};

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
        let dt = DateTime::parse_from_rfc3339(&s).map_err(|e| serde::de::Error::custom(e))?;
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
