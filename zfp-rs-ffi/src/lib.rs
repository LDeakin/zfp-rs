//! # zfp-rs-ffi
//!
//! C ABI bindings layer around the pure-Rust `zfp-rs` implementation.
//!
//! This crate mirrors the API surface of `zfp-sys` (generated from
//! `#include <zfp.h>`), providing zero-cost FFI adapters that delegate to the
//! existing `zfp-rs` compression logic.
//!
//! All ABI functions export upstream-compatible `zfp_*` and `stream_*` symbol
//! names, so C consumers can include the generated `zfp.h` and link this crate's
//! static library in place of the upstream C library.
//!
//! ## Scope
//!
//! The target API is `zfp-sys` version `0.1.15` as generated in this workspace.
//! This includes `stream_*` functions except `stream_set_stride` (not enabled
//! in `zfp-sys`), and excludes CFP APIs and `zfp_block_maximum_size`.

// Missing docs are suppressed during development. The FFI layer also
// intentionally mirrors C/bindgen names and keeps unsafe pointer operations
// inside extern shims.
#![allow(
    missing_docs,
    dead_code,
    unused_unsafe,
    unsafe_op_in_unsafe_fn,
    non_upper_case_globals,
    clippy::missing_safety_doc
)]
#![warn(clippy::pedantic)]

mod abi;
#[path = "bitstream.rs"]
mod bitstream_api;
mod block;
mod field;
mod header;
mod promote;
mod stream;
mod util;

pub use abi::*;
pub use bitstream_api::*;
pub use block::*;
pub use field::*;
pub use header::*;
pub use promote::*;
pub use stream::*;
