//! Developer diagnostics globals.
//!
//! This module owns process-wide developer diagnostics flags. Runtime/session
//! architecture types live in dedicated runtime-oriented modules (for example
//! `runtime_event` and `time`).

use std::sync::atomic::{AtomicBool, Ordering};

static DEVELOPER_DIAGNOSTICS_ENABLED: AtomicBool = AtomicBool::new(false);

pub fn set_developer_diagnostics_enabled(enabled: bool) {
    DEVELOPER_DIAGNOSTICS_ENABLED.store(enabled, Ordering::Relaxed);
}

pub fn developer_diagnostics_enabled() -> bool {
    DEVELOPER_DIAGNOSTICS_ENABLED.load(Ordering::Relaxed)
}
