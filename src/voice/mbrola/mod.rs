//! MBROLA-compatible phone-timed rendering.
//!
//! This module treats MBROLA as a renderer behind an explicit phone timing
//! contract. The first implementation shells out to an installed `mbrola`
//! voice database. The current native probe path validates the phone-timed
//! contract and emits a simple WAV while the real database decoder is being
//! reverse engineered.

pub mod database;
pub mod pho;
pub mod render;
pub mod symbols;
pub mod voice;

pub use database::{MbrolaDatabase, MbrolaDatabaseError, MbrolaDiphone};
pub use pho::{
    MbrolaPhoParseError, MbrolaPhone, MbrolaPitchTarget, PhoneTimedPlan, phone_timed_plan_to_pho,
    prosody_timing_plan_to_phone_timed_plan, read_pho_file, write_pho_file,
};
pub use render::{MbrolaRenderer, MbrolaRendererConfig, PhoneTimedRenderer, RenderReport};
pub use symbols::{MbrolaSymbolMap, UnmappedPhone};
pub use voice::MbrolaVoice;
