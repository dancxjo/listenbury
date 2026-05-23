mod breath_transcribe;
mod continue_generation;
mod cpal_diag;
mod demos;
mod diphone_cache;
mod dogfood_two;
mod live_half_duplex;
mod llama;
mod mbrola_inventory;
mod mbrola_render;
mod mic_transcribe;
mod models;
mod prosody_plan;
mod round_trip;
mod sing_demo;
mod soundscape_debug;
mod speech_cache;
mod trace_viewer_export;
mod transcribe;
mod vad_calibration;
mod vad_trace;
mod web;

pub(crate) use breath_transcribe::run_breath_transcribe;
pub(crate) use continue_generation::run_continue;
#[cfg(feature = "audio-cpal")]
pub(crate) use cpal_diag::{play_audio_frame_stream, play_audio_frames, prepare_audio_playback};
pub(crate) use cpal_diag::{run_play_wav, run_record_wav};
pub(crate) use demos::{run_demo_vad, run_fake_turn};
pub(crate) use diphone_cache::run_diphone;
pub(crate) use diphone_cache::run_diphone_cache;
pub(crate) use dogfood_two::run_dogfood_two;
pub(crate) use live_half_duplex::run_live_half_duplex;
pub(crate) use llama::run_llama_turn;
pub(crate) use mbrola_inventory::{run_mbrola_audit, run_mbrola_inventory};
pub(crate) use mbrola_render::run_mbrola_render;
pub(crate) use mic_transcribe::run_mic_transcribe;
pub(crate) use models::run_models;
pub(crate) use prosody_plan::run_prosody_plan;
pub(crate) use round_trip::run_round_trip_wav;
pub(crate) use sing_demo::run_sing_demo;
pub(crate) use soundscape_debug::run_soundscape_debug;
pub(crate) use speech_cache::run_speech_cache;
pub(crate) use trace_viewer_export::run_trace_viewer_export;
pub(crate) use transcribe::run_transcribe;
pub(crate) use vad_calibration::run_vad;
pub(crate) use vad_trace::run_vad_trace;
pub(crate) use web::run_web;

#[cfg(feature = "tts-piper")]
pub(crate) use crate::cli::piper::run_echo;
#[cfg(feature = "tts-piper")]
pub(crate) use crate::cli::piper::run_riper_compare;
#[cfg(feature = "tts-piper")]
pub(crate) use crate::cli::piper::run_say;

#[cfg(not(feature = "tts-piper"))]
use crate::cli::EchoCommand;
#[cfg(not(feature = "tts-piper"))]
use crate::cli::RiperCompareCommand;
#[cfg(not(feature = "tts-piper"))]
use crate::cli::SayCommand;
#[cfg(not(feature = "tts-piper"))]
use anyhow::Result;

#[cfg(not(feature = "tts-piper"))]
pub(crate) fn run_say(_command: SayCommand) -> Result<()> {
    anyhow::bail!("listenbury was built without the `tts-piper` feature")
}

#[cfg(not(feature = "tts-piper"))]
pub(crate) fn run_riper_compare(_command: RiperCompareCommand) -> Result<()> {
    anyhow::bail!("listenbury was built without the `tts-piper` feature")
}

#[cfg(not(feature = "tts-piper"))]
pub(crate) fn run_echo(_command: EchoCommand) -> Result<()> {
    anyhow::bail!("listenbury was built without the `tts-piper` feature")
}
