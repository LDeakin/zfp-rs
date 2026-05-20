// Integration test binary for C-ported tests.
// Each submodule corresponds to a set of C (cmocka) test sources.

#[path = "c/checksums.rs"]
mod checksums;

#[path = "c/constants.rs"]
mod constants;
#[path = "c/decode_block.rs"]
mod decode_block;
#[path = "c/decode_block_strided.rs"]
mod decode_block_strided;
#[path = "c/encode_block.rs"]
mod encode_block;
#[path = "c/encode_block_strided.rs"]
mod encode_block_strided;
#[path = "c/endtoend.rs"]
mod endtoend;
#[path = "c/field.rs"]
mod field;
#[path = "c/header.rs"]
mod header;
#[path = "c/promote.rs"]
mod promote;
#[path = "c/stream.rs"]
mod stream;
