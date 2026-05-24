//! MBROLA-compatible phone-timed rendering.
//!
//! This module treats MBROLA voice databases as native phone-timed renderers:
//! `.pho` phone durations and pitch targets drive diphone selection, duration
//! control, and TD-PSOLA overlap-add synthesis.
//!
//! `mbrola` in module/type names means file-format and synthesis-flow
//! compatibility (voice database parsing + `.pho` behavior), not source-code
//! derivation from upstream MBROLA implementations.

pub mod database;
pub mod diphone_provider;
pub mod fallback;
pub mod manifest;
pub mod pho;
pub mod render;
pub mod symbols;
pub mod units;
pub mod voice;

pub use database::{
    MbrolaDatabase, MbrolaDatabaseError, MbrolaDatabaseUnit, MbrolaDatabaseWriteOptions,
    MbrolaDiphone, encode_mbrola_database, write_mbrola_database,
};
pub use diphone_provider::{
    DiphoneKey, DiphoneLookup, DiphoneProvider, DiphoneUnit, DiphoneUnitMetadata,
    DiphoneUnitSource, ForgeProvenance, MbrolaDiphoneProvider,
};
pub use fallback::{
    FallbackReason, FallbackResult, fallback_warning, resolve_left_half, resolve_right_half,
};
pub use manifest::{ManifestError, VoiceManifest};
pub use pho::{
    MbrolaPhoParseError, MbrolaPhone, MbrolaPitchTarget, PhoneTimedPlan, phone_timed_plan_to_pho,
    prosody_timing_plan_to_phone_timed_plan, read_pho_file, write_pho_file,
};
pub use render::{
    MbrolaRenderer, MbrolaRendererConfig, PhoneTimedRenderer, RenderReport,
    render_phone_plan_with_diphone_provider_to_frames,
};
pub use symbols::{MbrolaSymbolMap, UnmappedPhone};
pub use units::{
    JoinPoint, UnitAssemblyReport, assemble_unit, left_half_samples, right_half_samples,
};
pub use voice::MbrolaVoice;
