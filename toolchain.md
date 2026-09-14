# Toolchain

Программы проекта и прямые пакеты из манифестов.

## Программы

| Программа | Как ставить | Зачем здесь | Источник |
| --- | --- | --- | --- |
| mise | brew / curl, затем `mise install` | Пины версий инструментов | https://github.com/jdx/mise |
| rust | mise | Компилятор и std | https://github.com/rust-lang/rust |
| rustc | mise (pin rust) | Компилятор Rust | https://github.com/rust-lang/rust |
| cargo | mise (pin rust) | Сборка и зависимости Rust | https://github.com/rust-lang/cargo |
| moon | mise | Таски монорепы | https://github.com/moonrepo/moon |
| ffmpeg | mise | Аудио/видео фикстуры | https://github.com/FFmpeg/FFmpeg |
| python | mise | Скрипты | https://github.com/python/cpython |
| node | mise | JS runtime | https://github.com/nodejs/node |
| cargo-dist | mise | Релизные артефакты | https://github.com/axodotdev/cargo-dist |
| cargo-cache | mise | Чистка cargo home | https://github.com/matthiaskrgr/cargo-cache |
| zig | mise | Свои ядра / native | https://github.com/ziglang/zig |

## cargo

| Пакет | Где | Источник | Зачем здесь |
| --- | --- | --- | --- |
| anyhow | локально | https://crates.io/crates/anyhow | Ошибки CLI |
| assert_cmd | локально | https://crates.io/crates/assert_cmd | P2.3 e2e CLI tests. |
| async-openai | локально | https://crates.io/crates/async-openai | Зависимость Rust |
| axum | локально | https://crates.io/crates/axum | Зависимость Rust |
| chrono | локально | https://crates.io/crates/chrono | Зависимость Rust |
| clap | локально | https://crates.io/crates/clap | CLI |
| clap_mangen | локально | https://crates.io/crates/clap_mangen | Зависимость Rust |
| criterion | локально | https://crates.io/crates/criterion | Зависимость Rust |
| encoding_rs | локально | https://crates.io/crates/encoding_rs | Зависимость Rust |
| ffmpeg-sidecar | локально | https://crates.io/crates/ffmpeg-sidecar | P4.5: system ffmpeg or auto-download. |
| futures | локально | https://crates.io/crates/futures | Зависимость Rust |
| heed | локально | https://crates.io/crates/heed | P2.8 prompt cache: LMDB via heed (mmap KV, fastest local store). |
| hf-hub | локально | https://crates.io/crates/hf-hub | P2.4 HF downloads (blocking API: sync CLI, runtime on bg thread). |
| hound | локально | https://crates.io/crates/hound | Зависимость Rust |
| keyring | локально | https://crates.io/crates/keyring | P3.8 OS keychain. Env vars still win. |
| llama-cpp-2 | локально | https://crates.io/crates/llama-cpp-2 | Локальный GGUF / llama.cpp |
| llama-cpp-sys-2 | локально | https://crates.io/crates/llama-cpp-sys-2 | Зависимость Rust |
| proptest | локально | https://crates.io/crates/proptest | Зависимость Rust |
| raw-cpuid | локально | https://crates.io/crates/raw-cpuid | Зависимость Rust |
| rayon | локально | https://crates.io/crates/rayon | Зависимость Rust |
| reqwest | локально | https://crates.io/crates/reqwest | HTTP |
| rubato | локально | https://crates.io/crates/rubato | Зависимость Rust |
| rustyline | локально | https://crates.io/crates/rustyline | P2.3 REPL line editing + file history. |
| serde | локально | https://crates.io/crates/serde | Сериализация |
| serde_json | локально | https://crates.io/crates/serde_json | JSON |
| sha2 | локально | https://crates.io/crates/sha2 | Зависимость Rust |
| symphonia | локально | https://crates.io/crates/symphonia | P4.1 audio decode. |
| sysinfo | локально | https://crates.io/crates/sysinfo | Зависимость Rust |
| tempfile | локально | https://crates.io/crates/tempfile | Зависимость Rust |
| thiserror | локально | https://crates.io/crates/thiserror | Ошибки |
| tokio | локально | https://crates.io/crates/tokio | Асинхронность |
| toml | локально | https://crates.io/crates/toml | Конфиг |
| whisper-rs | локально | https://crates.io/crates/whisper-rs | Зависимость Rust |
| wiremock | локально | https://crates.io/crates/wiremock | Зависимость Rust |
