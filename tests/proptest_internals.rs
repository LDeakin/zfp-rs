// Property-based differential tests for the low-level block codec.
//
// Reaches `zfp_rs::codec::{encode, decode}`, which are only public with the
// `internals` feature — hence `required-features` on this target in
// `Cargo.toml`.

#[path = "proptest/block_decode_compat.rs"]
mod block_decode_compat;
#[path = "proptest/block_encode_compat.rs"]
mod block_encode_compat;
