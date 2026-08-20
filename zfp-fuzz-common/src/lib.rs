//! Shared input model and target bodies for the `zfp-rs` fuzz targets.
//!
//! The fuzz crate itself lives in `fuzz/`, outside the cargo workspace, because
//! it needs a nightly toolchain and libFuzzer. All of the actual logic lives
//! here instead, inside the workspace, for two reasons:
//!
//! 1. **Regressions replay on stable.** `tests/regressions.rs` feeds committed
//!    crash inputs to the same `run` functions the fuzzer calls, so a fixed bug
//!    stays fixed under plain `cargo test --workspace`.
//! 2. **The code gets linted.** `cargo clippy --workspace --all-targets` does
//!    not see `fuzz/`; it does see this crate.
//!
//! See `fuzz/README.md` for the workflow.

#![warn(clippy::pedantic)]
#![allow(clippy::missing_panics_doc)] // panicking *is* the reporting mechanism here

pub mod input;
pub mod limits;
pub mod scalar;
pub mod targets;
