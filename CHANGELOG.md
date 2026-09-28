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
- `ZfpBitStreamOps::{read_header, decompress, decompress_with_execution}` and `ZfpBitStreamMutOps::{write_header, compress, compress_with_execution}` provided methods. `ZfpBitStreamRef` and `ZfpBitStreamRefMut` now have the same codec methods as `ZfpBitStream`, so borrowed buffers can be encoded and decoded without copying.

### Changed

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
- **Breaking**: `ZfpConfig::maximum_size` returns `Option<usize>` instead of `0` for unsupported dimensionality or overflow.

### Fixed

- `ZfpFieldMetadata::to_bits` no longer panics (or wraps) for a zero leading dimension.
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
