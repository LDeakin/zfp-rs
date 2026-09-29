// Integration test binary for property-based compatibility tests against the
// reference C library, covering the public API.
//
// Block-level differential tests live in `proptest_internals.rs`, which
// requires the `ffi` feature.

#[path = "proptest/bitstream_compat.rs"]
mod bitstream_compat;
#[path = "proptest/budget.rs"]
mod budget;
#[path = "proptest/compress_compat.rs"]
mod compress_compat;
#[path = "proptest/decompress_compat.rs"]
mod decompress_compat;
#[path = "proptest/header_compat.rs"]
mod header_compat;
#[path = "proptest/rounding.rs"]
mod rounding;
#[path = "proptest/tiny_blocks.rs"]
mod tiny_blocks;
