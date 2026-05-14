mod cpal_diag;
mod demos;
mod llama;
mod models;
mod round_trip;
mod speech_cache;
mod transcribe;
mod vad_trace;

pub(crate) use cpal_diag::{run_play_wav, run_record_wav};
pub(crate) use demos::{run_demo_vad, run_fake_turn};
pub(crate) use llama::run_llama_turn;
pub(crate) use models::run_models;
pub(crate) use round_trip::run_round_trip_wav;
pub(crate) use speech_cache::run_speech_cache;
pub(crate) use transcribe::run_transcribe_synthetic;
pub(crate) use vad_trace::run_vad_trace;

#[cfg(feature = "tts-piper")]
pub(crate) use crate::cli::piper::run_piper_say;

#[cfg(not(feature = "tts-piper"))]
use crate::cli::PiperSayCommand;
#[cfg(not(feature = "tts-piper"))]
use anyhow::Result;

#[cfg(not(feature = "tts-piper"))]
pub(crate) fn run_piper_say(_command: PiperSayCommand) -> Result<()> {
    anyhow::bail!("listenbury was built without the `tts-piper` feature")
}
