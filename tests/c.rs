// Integration test binary for C-ported tests that exercise the public API.
// Each submodule corresponds to a set of C (cmocka) test sources.
//
// Tests covering the low-level codec internals live in `c_internals.rs`, which
// requires the `ffi` feature to reach them.

#[path = "c/checksums.rs"]
mod checksums;

#[path = "c/constants.rs"]
mod constants;
#[path = "c/endtoend.rs"]
mod endtoend;
#[path = "c/field.rs"]
mod field;
#[path = "c/header.rs"]
mod header;
#[path = "c/stream.rs"]
mod stream;
