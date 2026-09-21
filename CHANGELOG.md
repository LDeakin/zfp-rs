# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased](https://github.com/LDeakin/zfp-rs/compare/v0.1.1...HEAD)

### Changed
- **Breaking**: Add `InvalidField` variant to `ZfpCompressionError` and mark `#[non_exhaustive]`
- **Breaking**: The `codec::block::*_strided*` entry points are now `unsafe`
  - They index through caller-supplied strides with no bounds check, so calling them from safe Rust could read or write out of bounds. The precondition is documented once on the `codec::block` module
- **Breaking**: Drop the `_with_params` suffix from the strided block dispatchers
  - With the parameterless forms gone the suffix distinguishes nothing, and the shorter names mirror the C symbols these stand in for
- **Breaking**: Remove the parameterless `codec::block::{encode,decode}_[partial_]block_strided` dispatchers
  - No caller anywhere: the C ABI and the whole-field driver both go through the `*_with_params` forms, and they encoded with lossless defaults, which mirrors nothing in the C API — `zfp_encode_block_strided_*` reads its parameters from the `zfp_stream`
  - The `dim*`-level parameterless wrappers stay, behind `internals`, for the C-port tests
- **Breaking**: Narrow the `codec` module's public surface
  - The strided entry points move to `codec::block::strided`, public only with `ffi`; `codec::{encode, decode}` are public only with the new `internals` feature; `codec::promote` only with `ffi`; `codec::transform` is now private
  - These are the monomorphised codec internals. `ffi` is the C-ABI seam; `internals` exists for the C-port and proptest suites, which are the only consumers of `codec::encode` and `codec::decode`
  - Without either feature the public `codec` API is six safe functions
- Bump the `zfp-sys` dev-dependency to 0.4

### Fixed
- Validate `ZfpField` length in `FieldPlan::new`
- Fix `ZfpField::begin`/`ZfpFieldMut::begin` and the `zfp_field` conversion in `zfp-rs-ffi` disagreeing about where a strided buffer starts
  - `ZfpField` takes a buffer beginning at the *lowest* address of its strided span, as `compress`/`decompress` have always assumed, but `begin` shifted by `imin` as if the buffer began at element `[0, 0, 0, 0]`. With a negative stride it returned a pointer before the start of the allocation
  - The C ABI layer had the mirror-image bug: C's `zfp_field.data` *is* element `[0, 0, 0, 0]`, so a negative stride made the constructed slice claim `-imin` elements past the end of the caller's buffer. It now shifts to the span's low end, matching `zfp_field_begin` in the reference implementation
  - A field's span now covers only the axes below its dimensionality, which is what the block walk has always used. A dimension declared past the first zero one is inert — `zfp_field_3d(.., 5, 0, 5)` is a 1-D field — so folding it in claimed a buffer the codec never touches, and with a negative stride on that axis it anchored the converted slice *before* the start of the caller's allocation
  - This is a deliberate difference from the reference implementation, whose `zfp_field_size_bytes`/`zfp_field_begin` fold in all four axes even though `zfp_compress` ignores the inert ones
- Fix a data race decompressing a field with aliasing strides under `ZfpExecution::Rayon`
  - Two blocks can write the same element, so such fields now decompress serially
- Fix undefined behaviour in the strided block gather/scatter paths
  - `compress`/`decompress` built a `&[T]` for each block with `slice::from_raw_parts`, then indexed far outside it through raw pointers. Under Stacked Borrows a slice reference carries provenance over exactly its own elements, so non-unit and negative strides were out-of-provenance accesses
  - The length was also wrong in both directions: it is `lx * ly * lz * lw * elem_size`, and the unused axes are 0, so it was **zero** for every 1-D, 2-D and 3-D field, while in 4-D it could reach past the end of the buffer for boundary blocks of a field whose dimensions are not multiples of 4
  - The `*_strided*` entry points now take `*const T`/`*mut T`, derived from the whole field buffer so their provenance covers every offset the strides generate
- Validate `ZfpField` data alignment in `FieldPlan::new`, reported as the new `MisalignedData` error variant
  - The required alignment is `ZfpScalarType::align`, the target alignment of the Rust type, rather than its size: 64-bit scalars are 4-byte aligned on some 32-bit targets

### Added
- `zfp-rs-ffi`: `zfp_block_maximum_size`, new in the zfp version `zfp-sys` 0.4 bundles
- `ZfpScalarType::align`, the alignment a buffer passed to `ZfpField::from_raw` must satisfy
- `ZfpScalarType::is_aligned`, checking a buffer pointer against `align`
- Fuzz targets (`cargo fuzz`) with a stable-toolchain crash-replay harness
- A Miri regression suite for the strided codec, run in CI under `-Zmiri-strict-provenance`

## [0.1.1](https://github.com/LDeakin/zfp-rs/releases/tag/v0.1.1) - 2026-05-21

### Added

- Add trusted publishing

## [0.1.0](https://github.com/LDeakin/zfp-rs/releases/tag/v0.1.0) - 2026-05-21

### Added

- Initial public release
