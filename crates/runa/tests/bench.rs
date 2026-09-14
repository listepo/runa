//! P2.10: `runa bench --json` schema + calibration DB insert.

use std::path::PathBuf;

use assert_cmd::Command;

fn runa() -> Command {
    let mut cmd = Command::cargo_bin("runa").expect("runa binary builds");
    cmd.env("RUNA_NO_PROMPT_CACHE", "1");
    cmd
}

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures")
        .join(name)
}

#[test]
fn bench_json_and_calibration_db() {
    let model = fixture("qwen2-0_5b-instruct-q4_0.gguf");
    let db = std::env::temp_dir().join(format!(
        "runa-bench-e2e-{}-{}.json",
        std::process::id(),
        "p210"
    ));
    let _ = std::fs::remove_file(&db);

    let out = runa()
        .env("RUNA_CALIBRATION", &db)
        .args([
            "bench",
            "--mode",
            "cpu",
            "--ctx",
            "256",
            "--pp",
            "8",
            "--tg",
            "4",
            "--json",
            model.to_str().unwrap(),
        ])
        .assert()
        .success()
        .get_output()
        .clone();
    let stdout = String::from_utf8(out.stdout).expect("utf8 stdout");
    for key in [
        "\"n_prompt\"",
        "\"n_gen\"",
        "\"pp_tok_s\"",
        "\"tg_tok_s\"",
        "\"predicted_pp\"",
        "\"predicted_tg\"",
        "\"model_hash\"",
        "\"placement\"",
    ] {
        assert!(stdout.contains(key), "missing {key} in {stdout}");
    }
    assert!(stdout.contains("\"n_prompt\":8"), "{stdout}");
    assert!(stdout.contains("\"n_gen\":4"), "{stdout}");

    let text = std::fs::read_to_string(&db).expect("calibration db written");
    assert!(text.contains("\"measured_pp\""), "{text}");
    assert!(text.contains("\"measured_tg\""), "{text}");
    let _ = std::fs::remove_file(db);
}
