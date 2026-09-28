# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased](https://github.com/LDeakin/zfp-rs/compare/v0.2.0...HEAD)

### Added

- `ZfpConfig::try_expert`, which rejects invalid expert-mode parameters.
- `ZfpBitStreamOps` and `ZfpBitStreamMutOps` methods are inherent on `ZfpBitStream`, `ZfpBitStreamRef` and `ZfpBitStreamRefMut`, so the traits no longer need to be in scope. This adds a dependency on `inherent`.
- `ZfpBitStreamOps::overflowed`, which reports writes dropped past the end of the buffer.
- `ZfpFieldError`, returned by the field constructors and setters.
- `ZfpField::from_raw_unchecked` and `ZfpFieldMut::from_raw_unchecked`, and `field::index_span`, behind `ffi`.
- `ZfpFieldMut::set_strides`.
- `ZfpBitStreamOps::{as_words, backing_words}`.
- `ZfpBlockError`, `ZfpFieldError` and the documented `ZFP_*` constants are re-exported at the crate root without `ffi`.
- `ZfpBitStreamOps::{read_header, decompress, decompress_with_execution}` and `ZfpBitStreamMutOps::{write_header, compress, compress_with_execution}` provided methods. `ZfpBitStreamRef` and `ZfpBitStreamRefMut` now have the same codec methods as `ZfpBitStream`, so borrowed buffers can be encoded and decoded without copying.

### Changed

- Serial compression is 2–10x faster and decompression 1.4–22x faster across the `api_compare` benchmark cases, with byte-identical output. The embedded coder transposes each block into bit planes and codes a whole group test at a time, the transforms are lifted as vectors, and all-zero planes and blocks take shortcuts.
- **Breaking**: `ZfpBitStreamMutOps::write_header` (and so `write_header` on every stream type) takes `&ZfpFieldMetadata` instead of `&ZfpField`, returns `Result<usize, ZfpCompressionError>` instead of `0` on failure, and writes nothing on failure.
- **Breaking**: `ZfpField::{new, new_strided, from_raw}` and `ZfpFieldMut::{new, new_strided, from_raw}` validate the dimensions and buffer and return `Result<Self, ZfpFieldError>`. A valid field no longer fails at compression time.
- **Breaking**: `ZfpCompressionError` is now `Field(ZfpFieldError)`, `BufferTooSmall` or `Metadata(ZfpMetadataError)`; `ZfpDecompressionError` is now `Field(ZfpFieldError)`. Both implement `Error::source`.
- **Breaking**: `ZfpField{,Mut}::metadata` returns `ZfpFieldMetadata`; encode it with `ZfpFieldMetadata::to_bits`.
- **Breaking**: `ZfpField{,Mut}::set_metadata` takes `ZfpFieldMetadata` and returns `Result<(), ZfpFieldError>` instead of `bool`.
- **Breaking**: `ZfpField::set_stride` is renamed `set_strides` and returns `Result<(), ZfpFieldError>`.
- **Breaking**: `ZfpField{,Mut}::field_index_span` is renamed `index_span`. `ZfpField::field_index_span_static` is removed; use `field::index_span` (`ffi`).
- **Breaking**: `ZfpField{,Mut}::begin` is removed; use `data().as_ptr()`.
- **Breaking**: `ZfpScalar::scalar_type()` is replaced by the associated const `ZfpScalar::SCALAR_TYPE`.
- **Breaking**: `ZfpMetadataError::Null` is renamed `InvalidDims` and also covers malformed dimensions such as `[0, 5, 0, 0]`. `ZfpMetadataError` and `ZfpHeaderError` are now `#[non_exhaustive]`.
- **Breaking**: `ZfpConfig::maximum_size` returns `Option<usize>` instead of `0` for malformed dimensions or overflow, and takes `impl ZfpDims` instead of `&[usize]`. The zero-padded `[usize; 4]` from `ZfpField::dims` now gives the right size rather than just the header size.
- **Breaking**: `ZfpConfig::{rate, precision, accuracy}` return `Option` instead of `0` when the config is in another mode.
- **Breaking**: `ZfpConfig::from_mode` is renamed `from_mode_bits`, and `ZfpConfig::compression_mode` is renamed `mode`.
- **Breaking**: `ZfpDims::dimensionality` and `ZfpStrides::dimensionality` are removed.
- **Breaking**: The `*_from_params` functions are no longer public without `ffi`.
- **Breaking**: `ZfpScalarType` variants are renamed `I32`, `I64`, `F32` and `F64`, and its `size`, `align`, `precision` and `is_aligned` methods take `self` by value.
- **Breaking**: `ZfpStreamAlignment::None` is renamed `Unaligned`, so it cannot be confused with `Option::None`.
- **Breaking**: `ZfpHeaderMask::NONE` is removed; use `ZfpHeaderMask::empty()`.
- **Breaking**: `ZfpBitStreamOps` and `ZfpBitStreamMutOps` are sealed.
- **Breaking**: `read_bit` returns `bool`, and `write_bit` takes `bool` and returns nothing. `write_word` returns nothing.
- **Breaking**: Bit counts are `u64`: `skip`, `pad` and `copy_from` take `u64`. `flush` returns `u32`, like `align`.
- **Breaking**: `bits_written`, `word_pos` and `size` are removed from the stream API; use `write_pos` or `as_bytes().len()`. `size` remains behind `ffi`. `words` is renamed `backing_words`.
- **Breaking**: `ZfpBitStream::from_buffer` is renamed `from_words`, and `into_vec` is renamed `into_bytes`. `into_words` returns only the words written, like `into_bytes`. `from_bytes` zero-pads a trailing partial word instead of dropping it.
- **Breaking**: `ZfpBitStreamRefMut::{from_words_mut, from_bytes_mut}` are renamed `from_words` and `from_bytes`.
- **Breaking**: `codec::block::{encode_block, decode_block}` take a `&ZfpConfig` after the stream, so every mode is available for block coding, not just unconstrained full precision. A reversible config selects lossless coding for every scalar type, and the output matches field compression. `encode_block_reversible_{f32,f64}` and `decode_block_reversible_{f32,f64}` are removed.

### Fixed

- `ZfpFieldMetadata::to_bits` no longer panics (or wraps) for a zero leading dimension.
- Rayon decompression leaves the stream cursor where serial decompression does, and returns the same size.
- Writing past the end of a bitstream no longer panics. The write is dropped, and `compress` and `write_header` return `ZfpCompressionError::BufferTooSmall`.

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
