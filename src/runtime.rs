//! Compatibility shim for diagnostics flags.
//!
//! Diagnostics globals now live in [`crate::diagnostics`]. This module remains
//! as a migration path for existing `listenbury::runtime::*` imports.

pub use crate::diagnostics::{developer_diagnostics_enabled, set_developer_diagnostics_enabled};
