//! Full CLI output fixtures via trycmd (help screens).
//!
//! trycmd pins the complete stdout of deterministic, model-free commands.
//! Exit codes and partial output matches stay in the assert_cmd tests
//! (`e2e.rs`, `doctor.rs`, `bench.rs`, …) — each tool where it fits.

#[test]
fn cli_help_fixtures() {
    let t = trycmd::TestCases::new();
    t.case("tests/cmd/*.toml");
    t.run();
}
