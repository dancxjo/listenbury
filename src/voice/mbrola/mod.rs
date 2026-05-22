//! MBROLA-compatible phone-timed rendering.
//!
//! This module treats MBROLA voice databases as native phone-timed renderers:
//! `.pho` phone durations and pitch targets drive diphone selection, duration
//! control, and TD-PSOLA overlap-add synthesis.

pub mod database;
pub mod diphone_provider;
pub mod pho;
pub mod render;
pub mod symbols;
pub mod voice;

pub use database::{MbrolaDatabase, MbrolaDatabaseError, MbrolaDiphone};
pub use diphone_provider::{
    DiphoneKey, DiphoneLookup, DiphoneProvider, DiphoneUnit, DiphoneUnitMetadata,
    DiphoneUnitSource, MbrolaDiphoneProvider,
};
pub use pho::{
    MbrolaPhoParseError, MbrolaPhone, MbrolaPitchTarget, PhoneTimedPlan, phone_timed_plan_to_pho,
    prosody_timing_plan_to_phone_timed_plan, read_pho_file, write_pho_file,
};
pub use render::{MbrolaRenderer, MbrolaRendererConfig, PhoneTimedRenderer, RenderReport};
pub use symbols::{MbrolaSymbolMap, UnmappedPhone};
pub use voice::MbrolaVoice;
