# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased](https://github.com/LDeakin/zfp-rs/compare/v1.0.0...HEAD)

## [1.0.0](https://github.com/LDeakin/zfp-rs/releases/tag/v1.0.0) - 2026-10-03

### Added

- `ZfpExecution::Rayon` now pipelines variable-rate decompression with a serial plane reader and parallel reconstruction. A bounded queue of up to 16 reusable buffers (about 1 MiB by default) feeds reconstruction batches without a barrier between them; no stream index or format change is required. `chunk_size` sets the blocks per batch, and 0 targets 64 KiB of planes per batch, including through the C ABI's `zfp_stream_set_omp_chunk_size`. 1-D fields and fields of one batch decode serially, as the pipeline is slower for them.
- `ZfpAllocError`, returned when a stream's buffer cannot be allocated.
- `ZfpConfig::from_raw_params`, behind `ffi`, which holds unvalidated parameters from a C `zfp_stream`.
- `ZfpConfig::from_raw_rate`, behind `ffi`, which computes a fixed-rate budget as C's `zfp_stream_set_rate` does, without validating it.
- `ZfpConfig::checked_mode_bits` and `ZfpConfigError`, which report parameters that a header's mode word cannot hold.
- `field::checked_index_span`, the index span of a field's dimensions and strides, or `None` if it overflows.
- `ZfpFieldError`, returned by the field constructors and setters, and `ZfpFieldMut::set_strides`.
- `ZfpField::from_raw_unchecked` and `ZfpFieldMut::from_raw_unchecked`, behind `ffi`.
- The bitstream trait methods are inherent on `ZfpBitStream`, `ZfpBitStreamRef` and `ZfpBitStreamRefMut`, so the traits need not be in scope. This adds a dependency on `inherent`.
- `ZfpBitStreamOps::{read_header, decompress, decompress_with_execution}` and `ZfpBitStreamMutOps::{write_header, compress, compress_with_execution}`, so `ZfpBitStreamRef` gains the decoding methods and `ZfpBitStreamRefMut` has all of them.
- `ZfpBitStreamOps::overflowed`, which reports writes dropped past the end of the buffer, and `ZfpBitStreamOps::as_words`.
- `ZfpBlockError` and the `ZFP_*` constants are re-exported at the crate root without `ffi`.
- `docs/differences-from-c.md`, which lists every known difference from the C library.

### Changed

- What the `ffi` and `internals` features expose is `#[doc(hidden)]` and exempt from semver. They exist for `zfp-rs-ffi` and this workspace's tests.
- No function panics for any argument, as the crate documentation now states; the entries below list the panics removed. Clippy lints and `tests/panic_free.rs` enforce this.
- **Breaking**: `ZfpField::{new, new_strided, from_raw}` and `ZfpFieldMut::{new, new_strided, from_raw}` validate the dimensions and buffer and return `Result<Self, ZfpFieldError>`, so a valid field no longer fails at compression time.
- **Breaking**: `ZfpField{,Mut}::metadata` returns `ZfpFieldMetadata` (encode it with `to_bits`), and `set_metadata` takes one and returns `Result<(), ZfpFieldError>` instead of `bool`. `ZfpField::set_stride` is renamed `set_strides` and returns `Result<(), ZfpFieldError>`.
- **Breaking**: `ZfpField{,Mut}::field_index_span` is renamed `index_span`, and `ZfpField::field_index_span_static` is replaced by `field::checked_index_span`. `ZfpField{,Mut}::begin` is removed; use `data().as_ptr()`.
- **Breaking**: `ZfpCompressionError` is now `Field(ZfpFieldError)`, `BufferTooSmall`, `Metadata(ZfpMetadataError)` or `Config(ZfpConfigError)`, and `ZfpDecompressionError` is `Field(ZfpFieldError)` or `Truncated`. Both implement `Error::source` and `From` their inner errors.
- **Breaking**: `decompress` and `decompress_with_execution` return `ZfpDecompressionError::Truncated` when decoding loads a word past the end of the buffer, instead of decoding the missing words as zeros. C reads past the capacity given to `stream_open` there, so the C ABI's `zfp_decompress` returns `0`. A stream missing only padding that decoding skips is not truncated, and returns its whole size, as in C. The count is of whole words, so a cut inside the last word of a stream from `ZfpBitStream::from_bytes`, which zero-pads it, is not seen.
- **Breaking**: `ZfpMetadataError::Null` is renamed `InvalidDims` and also covers malformed dimensions such as `[0, 5, 0, 0]`. `ZfpMetadataError` and `ZfpHeaderError` are `#[non_exhaustive]`.
- **Breaking**: `ZfpConfig::maximum_size` takes `impl ZfpDims` instead of `&[usize]` and returns `None` instead of `0` for malformed dimensions or overflow. The zero-padded `[usize; 4]` from `ZfpField::dims` now gives the right size rather than just the header's.
- **Breaking**: `ZfpConfig::{rate, precision, accuracy}` return `None` instead of `0` in other modes. `ZfpConfig::from_mode` is renamed `from_mode_bits` and `compression_mode` is renamed `mode`. The `*_from_params` functions are removed; call these methods on `ZfpConfig::from_raw_params` (`ffi`) instead.
- **Breaking**: `ZfpConfig::expert` validates its parameters as C `zfp_stream_set_params` does, and returns `Result<Self, ZfpConfigError>`, so every `ZfpConfig` is valid. It accepted `min_bits > max_bits` or a `max_prec` outside `1..=64`, which compressed without meaning.
- **Breaking**: `ZfpConfig::fixed_rate` returns `Result<Self, ZfpConfigError>`, rejecting a rate that is negative, NaN or infinite, rounds to no bits for an integer type, or gives more than `ZFP_MAX_BITS` bits per block. It raised a negative or NaN rate to the block header for float types, gave integer types a zero budget, and for an infinite or very large rate gave a budget above `ZFP_MAX_BITS`, or overflowed if word-aligned.
- **Breaking**: `ZfpScalar::scalar_type()` is replaced by the associated const `SCALAR_TYPE`. `ZfpScalarType` variants are renamed `I32`, `I64`, `F32` and `F64`, and its methods take `self` by value.
- **Breaking**: `ZfpStreamAlignment::None` is renamed `Unaligned`, to avoid confusion with `Option::None`. `ZfpHeaderMask::NONE` is removed in favour of `empty()`, and `ZfpDims::dimensionality` and `ZfpStrides::dimensionality` are removed.
- **Breaking**: `ZfpBitStreamOps` and `ZfpBitStreamMutOps` are sealed.
- **Breaking**: `read_bit` returns `bool`, `write_bit` takes `bool`, and `write_bit` and `write_word` return nothing. `skip`, `pad` and `copy_from` take `u64` bit counts, and `flush` returns `u32`, like `align`.
- **Breaking**: `bits_written`, `word_pos` and `size` are removed; use `write_pos` or `as_bytes().len()` (`size` remains behind `ffi`). `words` is renamed `backing_words`.
- **Breaking**: `ZfpBitStream::from_buffer` is renamed `from_words` and `into_vec` is renamed `into_bytes`. `into_words` returns only the words written, and `from_bytes` zero-pads a trailing partial word instead of dropping it. `ZfpBitStreamRefMut::{from_words_mut, from_bytes_mut}` are renamed `from_words` and `from_bytes`.
- **Breaking**: `ZfpBitStream::{new, from_bytes, into_bytes}` return `Result<_, ZfpAllocError>`. They panicked for a capacity beyond the address space, and aborted the process when the allocator failed. Rayon compression compresses serially if a chunk's buffer cannot be allocated, and the C ABI's `stream_open` and `stream_clone` return null, as their documentation says.
- **Breaking**: `ZfpBitStream::write_header` takes `&ZfpFieldMetadata` instead of `&ZfpField`, and returns `Result<usize, ZfpCompressionError>` instead of `0` on failure, writing nothing.
- **Breaking**: `codec::block::{encode_block, decode_block}` take a `&ZfpConfig` after the stream, so blocks can be coded in any mode, not just unconstrained full precision. A reversible config codes every scalar type losslessly, and matches field compression. `{encode,decode}_block_reversible_{f32,f64}` are removed.
- **Breaking**: The strided `codec::block` functions (`ffi`) take strides as `&[isize; 4]` and lengths as `[usize; 4]`, and the partial ones return `Result<usize, ZfpBlockError>`, rejecting a length outside `1..=4`. A slice shorter than the dimensionality panicked, a length above 4 indexed past the block, and a length of 0 read before the block's origin.
- **Breaking**: `codec::promote::*` (`ffi`) return `Result<(), ZfpBlockError>`, rejecting a slice shorter than a block, which they indexed out of bounds.
- **Breaking**: The block functions in `codec::{encode,decode}::reversible` (`internals`) take a `&ZfpConfig` instead of nothing or a `ZfpRounding`, so they can honour its limits.
- **Breaking**: `codec::block::{encode,decode}_block_strided_reversible` are removed, as the strided block functions code a reversible config losslessly. `compress_bitstream`, `decompress_bitstream` and `read_header_bitstream` are removed; call the bitstream methods. `ZfpBitStreamOps::data_ptr` is removed; use `backing_words().as_ptr()`. All were behind `ffi`.

### Fixed

- `ZfpFieldMetadata::to_bits` no longer panics (or wraps) for a zero leading dimension.
- The crate builds on 32-bit targets. `ZfpFieldMetadata::from_bits` returns `None` for a dimension that does not fit in `usize`.
- Field constructors and setters return `ZfpFieldError::ShapeTooLarge` for overlapping strides whose element or block count overflows. Compressing such a field panicked in debug builds and wrote nothing in release. For unchecked fields, `num_elements`, `num_blocks`, `index_span` and `size_bytes` saturate, and the C ABI rejects a `zfp_field` whose span overflows.
- `write_header` returns `ZfpCompressionError::Config`, writing nothing, if the mode word cannot hold the config's parameters: a budget of 0 or above 32768 bits, or a `min_exp` outside -16495 to 16272. It wrote a different config, so decoding with the header could change values and misplace every block after the first.
- Rayon decompression decodes serially when a fixed-rate `max_bits` is below the float block header (9 bits for `f32`, 12 for `f64`), which only expert configs allow. Block sizes then vary, so it read blocks from the wrong offsets.
- Lossy float compression no longer overflows computing the precision for an expert `min_exp` near `i32::MIN` or `i32::MAX`, which panicked in debug builds and wrapped in release.
- `zfp_stream_set_rate` returns `0` and leaves the stream unchanged for a rate that is NaN or rounds outside `0..=u32::MAX` bits per block, where C's conversion is undefined, and for a word-aligned budget that C wraps around to zero, which panicked in debug builds.
- `read_bits` and `write_bits` read or write 64 bits for a count above 64, the most C supports, instead of shifting out of range, which panicked in debug builds and gave wrong values or looped forever in release. The C ABI's `stream_read_bits` and `stream_write_bits` do the same, instead of truncating the count to 32 bits.
- The C ABI's `stream_write_bit` writes the low bit of its argument, where C adds the value whole and corrupts the buffer for a value above 1. This is listed in `docs/differences-from-c.md`.
- `pad` and `copy_from` skip at once the words they would write past the end of the buffer, so a huge count returns promptly instead of looping up to 2^58 times.
- Decoding a block no longer panics in debug builds when the read position has wrapped around, as it does after a write in the first word or a seek near `u64::MAX`.
- The stream cursor saturates instead of overflowing, and so do the sizes `compress` and `decompress` return, which panicked in debug builds on 32-bit targets after a seek far past the end.
- Writing past the end of a bitstream no longer panics: the write is dropped, and `compress` and `write_header` return `ZfpCompressionError::BufferTooSmall`.
- Seeking past the end of a bitstream keeps the offset, as in C, instead of clamping it; reads there yield zeros and writes are dropped. Decoding a truncated stream no longer moves the cursor backwards, which panicked in debug builds and made the strided `codec::block` decoders, and so the C ABI's `zfp_decode_block_*`, return a wrapped bit count.
- Rayon execution runs serially if the thread pool cannot be built, instead of panicking. It no longer overflows computing chunk boundaries for huge block counts, and decompresses serially if the blocks would pass bit offset `u64::MAX`, where the per-block seeks overflowed.
- Rayon decompression leaves the cursor where serial decompression does, and returns the same size, or the same `Truncated` error.
- Reversible expert configurations (`min_exp < ZFP_MIN_EXP`) honour `min_bits`, `max_bits` and `max_prec`, as in C, instead of coding every block losslessly. Unlike C, an all-zero float block is padded to `min_bits`, which C's decoder expects but its encoder skips, and a `max_bits` below the block header leaves no budget instead of wrapping around.
- The strided `codec::block` functions, and so the C ABI's `zfp_encode_block_*` and `zfp_decode_block_*`, use the reversible coder for a reversible config, as C does, instead of the lossy coder.
- `ZfpConfig::maximum_size`, and so `zfp_stream_maximum_size`, is never less than the block headers, which a block writes even when `max_bits` is smaller; C under-reports these configurations too. Rayon compression sizes its chunks with the same bound, so it no longer loses bits when blocks exceed `max_bits`.
- `zfp_stream_maximum_size` ignores the dimensions from the first zero one on, as C does. A field such as `[4, 0, 4, 0]` was sized for the header alone.
- Lossy compression no longer destroys float blocks whose largest magnitude is below 2^-98 (`f32`) or 2^-962 (`f64`): their scale factor overflowed, as in C (zfp issue #119). They are now scaled exactly; their bytes differ from C's, but C decodes them.
- `write_header` returns `ZfpCompressionError::BufferTooSmall` when the cursor has passed bit `u64::MAX`, where its position wraps. It returned `Ok` while dropping every bit.
- `ZfpRounding::Last` no longer biases reversible decoding, which made it lossy, as in a C build with `ZFP_ROUND_LAST`. Streams are unchanged.

## [0.2.0](https://github.com/LDeakin/zfp-rs/releases/tag/v0.2.0) - 2026-09-28

### Added

- Runtime rounding via `ZfpRounding` and `ZfpConfig::with_rounding` (default `Never`).
- `zfp-rs-ffi`: `round-tight-error`, `FFI_ROUNDING`, and `zfp_block_maximum_size`.
- `ZfpScalarType::align` and `is_aligned` for buffer alignment.
- Fuzz targets, stable crash replay, and Miri regression tests.

### Changed

- **Breaking**: `ZfpCompressionError` and `ZfpDecompressionError` are non-exhaustive and gain `InvalidField` and `MisalignedData`.
- **Breaking**: Strided `codec::block` functions are now `unsafe`, take raw pointers, and require `ffi`. Non-reversible dispatchers take `&ZfpConfig`, drop `_with_params`, and no longer have parameterless forms.
- **Breaking**: `codec::{encode, decode}` require `internals`, `codec::promote` requires `ffi`, and `codec::transform` is private.
- **Breaking**: Reversible decoders gain a `ZfpRounding` parameter; parameterized non-reversible block functions under `codec::{encode, decode}` take `&ZfpConfig`.
- **Breaking**: `zfp-rs-ffi::zfp_type` is an integer newtype instead of a Rust enum; the C ABI is unchanged.

### Fixed

- Reject undersized or misaligned field buffers.
- Correct negative-stride buffer origins in `ZfpField::begin`, `ZfpFieldMut::begin`, and the C ABI.
- Decode fields whose strides may alias serially under `ZfpExecution::Rayon`.
- Fix undefined behaviour in strided gather/scatter by replacing per-block slices with raw pointers.
- Handle out-of-range bitstream reads and seeks without panicking; reads yield zero.
- Avoid arithmetic panics for extreme expert parameters; `ZfpConfig::maximum_size` returns zero on overflow.

## [0.1.1](https://github.com/LDeakin/zfp-rs/releases/tag/v0.1.1) - 2026-05-21

### Added

- Add trusted publishing

## [0.1.0](https://github.com/LDeakin/zfp-rs/releases/tag/v0.1.0) - 2026-05-21

### Added

- Initial public release
