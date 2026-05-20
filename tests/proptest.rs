// Integration test binary for property-based compatibility tests.

#[path = "proptest/bitstream_compat.rs"]
mod bitstream_compat;
#[path = "proptest/block_decode_compat.rs"]
mod block_decode_compat;
#[path = "proptest/block_encode_compat.rs"]
mod block_encode_compat;
#[path = "proptest/compress_compat.rs"]
mod compress_compat;
#[path = "proptest/decompress_compat.rs"]
mod decompress_compat;
#[path = "proptest/header_compat.rs"]
mod header_compat;
