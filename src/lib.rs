//! # zfp-rs
//!
//! Pure-Rust implementation of [ZFP](https://github.com/llnl/zfp): a compression algorithm for
//! compressed floating-point and integer arrays.
//!
//! ## Platform Requirements
//!
//! Bit-for-bit compatibility with the C reference implementation requires
//! a **little-endian** platform. The bit stream uses 64-bit words written in
//! native byte order, meaning the compressed byte sequence differs between
//! little-endian and big-endian architectures.
//!
//! The C reference implementation (`llnl/zfp`) documents this trade-off in its
//! [FAQ][faq-portability] and [bit-stream documentation][bs-portability]:
//!
//! - On little-endian platforms (x86-64, AMD64, ARM64), the default 64-bit
//!   word size produces portable output within the little-endian ecosystem.
//! - On big-endian platforms, the C implementation requires compiling with
//!   `-DBIT_STREAM_WORD_TYPE=uint8` (8-bit word size) to achieve byte-order
//!   independence. zfp-rs does not support this configuration.
//! - The C reference assumes IEEE 754 floating-point format.
//!
//! [faq-portability]: https://zfp.readthedocs.io/en/latest/faq.html#q-portability
//! [bs-portability]: https://zfp.readthedocs.io/en/latest/bit-stream.html
//!
//! ## Byte-for-byte compatibility
//!
//! This crate produces bit-for-bit identical compressed output to the reference
//! C implementation for all supported scalar types (`i32`, `i64`, `f32`, `f64`),
//! all dimensionalities (1-D through 4-D), and all compression modes,
//! **when running on a little-endian platform**.
//!
//! ## Quick start
//!
//! See the [CHANGELOG] for release notes.
//!
//! [CHANGELOG]: https://github.com/LDeakin/zfp-rs/blob/main/CHANGELOG.md
//!
//! ```rust
//! # use std::error::Error;
//! # fn main() -> Result<(), Box<dyn Error>> {
//! use zfp_rs::{ZfpBitStream, ZfpField, ZfpFieldMut, ZfpConfig};
//! use zfp_rs::{ZfpScalarType, ZfpDimensionality, ZfpStreamAlignment};
//!
//! let data: Vec<f64> = (0..16).map(|i| i as f64 * 0.1).collect();
//! let field = ZfpField::new(&data, [4usize, 4]);
//!
//! // Configure fixed-rate compression.
//! let config = ZfpConfig::fixed_rate(
//!     8.0,
//!     ZfpScalarType::Double,
//!     ZfpDimensionality::D2,
//!     ZfpStreamAlignment::None,
//! );
//!
//! // Compress.
//! let mut bs = ZfpBitStream::new(1024);
//! let compressed_bytes = bs.compress(&config, &field)?;
//!
//! // Decompress back.
//! bs.rewind();
//! let mut output = vec![0.0f64; 16];
//! let mut out_field = ZfpFieldMut::new(&mut output, [4usize, 4]);
//! bs.decompress(&config, &mut out_field)?;
//! # Ok(())
//! # }
//! ```

#![allow(clippy::too_many_arguments)]
#![warn(clippy::pedantic)]

pub mod bitstream;
pub mod codec;
pub(crate) mod compress;
pub mod config;
pub(crate) mod decompress;
pub mod execution;
pub mod field;
mod field_plan;
pub mod header;
pub mod types;

// ---------------------------------------------------------------------------
// Re-exports: high-level public API at the crate root
// ---------------------------------------------------------------------------

pub use bitstream::{
    ZfpBitStream, ZfpBitStreamMutOps, ZfpBitStreamOps, ZfpBitStreamRef, ZfpBitStreamRefMut,
};
pub use config::{STREAM_WORD_BITS, STREAM_WORD_BYTES, ZfpConfig, ZfpRounding, ZfpStreamAlignment};
pub use execution::ZfpExecution;
pub use field::{ZfpField, ZfpFieldMetadata, ZfpFieldMut};
pub use header::{ZfpHeader, ZfpHeaderError};
pub use types::{
    InvalidDimensionalityError, ZfpBitStreamWord, ZfpCompressionError, ZfpDecompressionError,
    ZfpDimensionality, ZfpDims, ZfpHeaderMask, ZfpMetadataError, ZfpMode, ZfpScalar, ZfpScalarType,
    ZfpStrides,
};

// ---------------------------------------------------------------------------
// FFI feature: low-level APIs for the C-ABI bindings layer
// ---------------------------------------------------------------------------

// Re-export FFI-specific helper functions at the crate root.
#[cfg(feature = "ffi")]
pub use config::{
    accuracy_from_params, compression_mode_from_params, mode_bits_from_params,
    precision_from_params, rate_from_params,
};

// Re-export FFI-specific numeric constants at the crate root.
#[cfg(feature = "ffi")]
pub use types::{
    ZFP_HEADER_MAX_BITS, ZFP_MAGIC_BITS, ZFP_MAX_BITS, ZFP_MAX_PREC, ZFP_META_BITS, ZFP_MIN_BITS,
    ZFP_MIN_EXP, ZFP_MODE_LONG_BITS, ZFP_MODE_SHORT_BITS,
};

/// Compress through any writable bitstream implementation.
///
/// # Errors
///
/// Returns [`ZfpCompressionError`] if the field type or dimensions are unsupported
/// for the selected configuration.
#[cfg(feature = "ffi")]
pub fn compress_bitstream(
    bs: &mut dyn ZfpBitStreamMutOps,
    field: &ZfpField,
    config: &ZfpConfig,
) -> Result<usize, ZfpCompressionError> {
    compress::compress(bs, field, config)
}

/// Decompress through any readable bitstream implementation.
///
/// # Errors
///
/// Returns [`ZfpDecompressionError`] if the target field type or dimensions are
/// unsupported for the selected configuration.
#[cfg(feature = "ffi")]
pub fn decompress_bitstream(
    bs: &mut dyn ZfpBitStreamOps,
    field: &mut ZfpFieldMut,
    config: &ZfpConfig,
) -> Result<usize, ZfpDecompressionError> {
    decompress::decompress(bs, field, config)
}

/// Read a ZFP header through any readable bitstream implementation.
///
/// # Errors
///
/// Returns [`ZfpHeaderError`] if a requested header section is invalid.
#[cfg(feature = "ffi")]
pub fn read_header_bitstream(
    bs: &mut dyn ZfpBitStreamOps,
    mask: ZfpHeaderMask,
) -> Result<ZfpHeader, ZfpHeaderError> {
    header::read_header_bs(bs, mask)
}
