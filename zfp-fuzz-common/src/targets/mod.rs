//! Fuzz target bodies.
//!
//! Each module exposes `pub fn run(data: &[u8])`. The libFuzzer harnesses in
//! `fuzz/fuzz_targets/` are five-line wrappers around these, which is what
//! lets crash artifacts be replayed on stable Rust by
//! `zfp-fuzz-common/tests/regressions.rs` — no nightly toolchain required to
//! keep a fixed bug fixed.
//!
//! A `run` must never panic on *malformed* input: inputs it cannot use are
//! skipped by returning early. A panic therefore always signals a real defect.

pub mod bitstream_ops;
pub mod block_codec;
pub mod config_mode;
pub mod decompress_stream;
pub mod header_decode;
pub mod roundtrip;
