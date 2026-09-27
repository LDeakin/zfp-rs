# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased](https://github.com/LDeakin/zfp-rs/compare/v0.2.0...HEAD)

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
