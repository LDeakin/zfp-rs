# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased](https://github.com/LDeakin/zfp-rs/compare/v0.2.0...HEAD)

### Added

- `ZfpConfig::try_expert`, which rejects invalid expert-mode parameters.
- `ZfpFieldError`, returned by the field constructors and setters, and `ZfpFieldMut::set_strides`.
- `ZfpField::from_raw_unchecked`, `ZfpFieldMut::from_raw_unchecked` and `field::index_span`, behind `ffi`.
- The bitstream trait methods are inherent on `ZfpBitStream`, `ZfpBitStreamRef` and `ZfpBitStreamRefMut`, so the traits need not be in scope. This adds a dependency on `inherent`.
- `ZfpBitStreamOps::{read_header, decompress, decompress_with_execution}` and `ZfpBitStreamMutOps::{write_header, compress, compress_with_execution}`, so `ZfpBitStreamRef` gains the decoding methods and `ZfpBitStreamRefMut` has all of them.
- `ZfpBitStreamOps::overflowed`, which reports writes dropped past the end of the buffer, and `ZfpBitStreamOps::as_words`.
- `ZfpBlockError` and the `ZFP_*` constants are re-exported at the crate root without `ffi`.
- `docs/differences-from-c.md`, which lists every known difference from the C library.

### Changed

- **Breaking**: `ZfpField::{new, new_strided, from_raw}` and `ZfpFieldMut::{new, new_strided, from_raw}` validate the dimensions and buffer and return `Result<Self, ZfpFieldError>`, so a valid field no longer fails at compression time.
- **Breaking**: `ZfpField{,Mut}::metadata` returns `ZfpFieldMetadata` (encode it with `to_bits`), and `set_metadata` takes one and returns `Result<(), ZfpFieldError>` instead of `bool`. `ZfpField::set_stride` is renamed `set_strides` and returns `Result<(), ZfpFieldError>`.
- **Breaking**: `ZfpField{,Mut}::field_index_span` is renamed `index_span`, and `ZfpField::field_index_span_static` is replaced by `field::index_span` (`ffi`). `ZfpField{,Mut}::begin` is removed; use `data().as_ptr()`.
- **Breaking**: `ZfpCompressionError` is now `Field(ZfpFieldError)`, `BufferTooSmall` or `Metadata(ZfpMetadataError)`, and `ZfpDecompressionError` is `Field(ZfpFieldError)`. Both implement `Error::source` and `From` their inner errors.
- **Breaking**: `ZfpMetadataError::Null` is renamed `InvalidDims` and also covers malformed dimensions such as `[0, 5, 0, 0]`. `ZfpMetadataError` and `ZfpHeaderError` are `#[non_exhaustive]`.
- **Breaking**: `ZfpConfig::maximum_size` takes `impl ZfpDims` instead of `&[usize]` and returns `None` instead of `0` for malformed dimensions or overflow. The zero-padded `[usize; 4]` from `ZfpField::dims` now gives the right size rather than just the header's.
- **Breaking**: `ZfpConfig::{rate, precision, accuracy}` return `None` instead of `0` in other modes. `ZfpConfig::from_mode` is renamed `from_mode_bits` and `compression_mode` is renamed `mode`. The `*_from_params` functions require `ffi`.
- **Breaking**: `ZfpScalar::scalar_type()` is replaced by the associated const `SCALAR_TYPE`. `ZfpScalarType` variants are renamed `I32`, `I64`, `F32` and `F64`, and its methods take `self` by value.
- **Breaking**: `ZfpStreamAlignment::None` is renamed `Unaligned`, to avoid confusion with `Option::None`. `ZfpHeaderMask::NONE` is removed in favour of `empty()`, and `ZfpDims::dimensionality` and `ZfpStrides::dimensionality` are removed.
- **Breaking**: `ZfpBitStreamOps` and `ZfpBitStreamMutOps` are sealed.
- **Breaking**: `read_bit` returns `bool`, `write_bit` takes `bool`, and `write_bit` and `write_word` return nothing. `skip`, `pad` and `copy_from` take `u64` bit counts, and `flush` returns `u32`, like `align`.
- **Breaking**: `bits_written`, `word_pos` and `size` are removed; use `write_pos` or `as_bytes().len()` (`size` remains behind `ffi`). `words` is renamed `backing_words`.
- **Breaking**: `ZfpBitStream::from_buffer` is renamed `from_words` and `into_vec` is renamed `into_bytes`. `into_words` returns only the words written, and `from_bytes` zero-pads a trailing partial word instead of dropping it. `ZfpBitStreamRefMut::{from_words_mut, from_bytes_mut}` are renamed `from_words` and `from_bytes`.
- **Breaking**: `ZfpBitStream::write_header` takes `&ZfpFieldMetadata` instead of `&ZfpField`, and returns `Result<usize, ZfpCompressionError>` instead of `0` on failure, writing nothing.
- **Breaking**: `codec::block::{encode_block, decode_block}` take a `&ZfpConfig` after the stream, so blocks can be coded in any mode, not just unconstrained full precision. A reversible config codes every scalar type losslessly, and matches field compression. `{encode,decode}_block_reversible_{f32,f64}` are removed.
- **Breaking**: `codec::block::encode_block_strided_reversible` (`ffi`) and the block functions in `codec::{encode,decode}::reversible` (`internals`) take a `&ZfpConfig` instead of nothing or a `ZfpRounding`, so they can honour its limits.

### Fixed

- `ZfpFieldMetadata::to_bits` no longer panics (or wraps) for a zero leading dimension.
- Writing past the end of a bitstream no longer panics: the write is dropped, and `compress` and `write_header` return `ZfpCompressionError::BufferTooSmall`.
- Seeking past the end of a bitstream keeps the offset, as in C, instead of clamping it; reads there yield zeros and writes are dropped. Decoding a truncated stream no longer moves the cursor backwards, which panicked in debug builds and made the strided `codec::block` decoders, and so the C ABI's `zfp_decode_block_*`, return a wrapped bit count.
- Rayon decompression leaves the cursor where serial decompression does, and returns the same size, truncated streams included.
- Reversible expert configurations (`min_exp < ZFP_MIN_EXP`) honour `min_bits`, `max_bits` and `max_prec`, as in C, instead of coding every block losslessly. Unlike C, an all-zero float block is padded to `min_bits`, which C's decoder expects but its encoder skips, and a `max_bits` below the block header leaves no budget instead of wrapping around.
- The strided `codec::block` functions, and so the C ABI's `zfp_encode_block_*` and `zfp_decode_block_*`, use the reversible coder for a reversible config, as C does, instead of the lossy coder.
- `ZfpConfig::maximum_size`, and so `zfp_stream_maximum_size`, is never less than the block headers, which a block writes even when `max_bits` is smaller; C under-reports these configurations too. Rayon compression sizes its chunks with the same bound, so it no longer loses bits when blocks exceed `max_bits`.
- `zfp_stream_maximum_size` ignores the dimensions from the first zero one on, as C does. A field such as `[4, 0, 4, 0]` was sized for the header alone.
- Lossy compression no longer destroys float blocks whose largest magnitude is below 2^-98 (`f32`) or 2^-962 (`f64`): their scale factor overflowed, as in C (zfp issue #119). They are now scaled exactly; their bytes differ from C's, but C decodes them.
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
