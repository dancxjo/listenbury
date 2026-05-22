use std::path::PathBuf;

#[test]
fn mbrola_smoke_renders_when_environment_is_configured() {
    let voice = std::env::var_os("MBROLA_VOICE")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("data/mbrola/us3/us3"));
    if !voice.is_file() {
        eprintln!(
            "skipping MBROLA smoke test: voice database not found at {}",
            voice.display()
        );
        return;
    }

    let dir = std::env::temp_dir().join(format!("listenbury-mbrola-smoke-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("create smoke temp dir");
    let pho = dir.join("hello.pho");
    let wav = dir.join("hello.wav");
    std::fs::write(
        &pho,
        "h 80\n@ 120 0 120 50 130 100 125\nl 90\n@U 180 0 125 60 135 100 130\n_ 100\n",
    )
    .expect("write smoke .pho");

    let report = listenbury::voice::mbrola::render::render_raw_pho(None, voice, &pho, &wav)
        .expect("MBROLA smoke render should succeed");

    assert!(wav.is_file(), "expected MBROLA to create {}", wav.display());
    assert_eq!(report.phone_count, 5);
    assert!(report.duration_ms >= 500);
}
