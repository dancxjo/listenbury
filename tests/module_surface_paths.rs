use listenbury::hearing::{
    AuditorySceneAnalyzer, BreathGroupSegmenter, VadBackendKind, create_vad_backend,
};
use listenbury::memory::{MemoryTrace, NoopMemorySink, SpeakerRole};
use listenbury::time::ExactTimestamp;

#[test]
fn subsystem_module_surfaces_are_public() {
    let _ = VadBackendKind::Energy;
    let _ = create_vad_backend(VadBackendKind::Energy).expect("energy VAD backend should build");
    let _ = std::mem::size_of::<BreathGroupSegmenter>();
    let _ = std::mem::size_of::<AuditorySceneAnalyzer>();

    let _ = NoopMemorySink;
    let _ = MemoryTrace::ConversationTurnFinalized {
        speaker: SpeakerRole::User,
        text: "hello".to_string(),
        occurred_at: ExactTimestamp::now(),
    };
}
