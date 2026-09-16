// Integration test binary for C-ported tests of the low-level codec.
//
// These reach `zfp_rs::codec::{block, promote}`, public only with the `ffi`
// feature, and `zfp_rs::codec::{encode, decode}`, public only with the
// `internals` feature — hence `required-features` on this target in
// `Cargo.toml`. The strided entry points they call are `unsafe`; each call site
// documents why its buffer covers the strides it passes.

#[path = "c/checksums.rs"]
mod checksums;

#[path = "c/decode_block.rs"]
mod decode_block;
#[path = "c/decode_block_strided.rs"]
mod decode_block_strided;
#[path = "c/encode_block.rs"]
mod encode_block;
#[path = "c/encode_block_strided.rs"]
mod encode_block_strided;
#[path = "c/promote.rs"]
mod promote;
