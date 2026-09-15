# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased](https://github.com/LDeakin/zfp-rs/compare/v0.1.1...HEAD)

### Changed
- **Breaking**: Add `InvalidField` variant to `ZfpCompressionError` and mark `#[non_exhaustive]`

### Fixed
- Validate `ZfpField` length in `CompressInfo::new` and `DecompressInfo::new`
- Fix `ZfpField::begin`/`ZfpFieldMut::begin` and the `zfp_field` conversion in `zfp-rs-ffi` disagreeing about where a strided buffer starts
  - `ZfpField` takes a buffer beginning at the *lowest* address of its strided span, as `compress`/`decompress` have always assumed, but `begin` shifted by `imin` as if the buffer began at element `[0, 0, 0, 0]`. With a negative stride it returned a pointer before the start of the allocation
  - The C ABI layer had the mirror-image bug: C's `zfp_field.data` *is* element `[0, 0, 0, 0]`, so a negative stride made the constructed slice claim `-imin` elements past the end of the caller's buffer. It now shifts to the span's low end, matching `zfp_field_begin` in the reference implementation
  - A field's span now covers only the axes below its dimensionality, which is what the block walk has always used. A dimension declared past the first zero one is inert — `zfp_field_3d(.., 5, 0, 5)` is a 1-D field — so folding it in claimed a buffer the codec never touches, and with a negative stride on that axis it anchored the converted slice *before* the start of the caller's allocation
  - This is a deliberate difference from the reference implementation, whose `zfp_field_size_bytes`/`zfp_field_begin` fold in all four axes even though `zfp_compress` ignores the inert ones

## [0.1.1](https://github.com/LDeakin/zfp-rs/releases/tag/v0.1.1) - 2026-05-21

### Added

- Add trusted publishing

## [0.1.0](https://github.com/LDeakin/zfp-rs/releases/tag/v0.1.0) - 2026-05-21

### Added

- Initial public release
