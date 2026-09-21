//! Core ZFP types, constants, and the sealed `ZfpScalar` trait.

use bitflags::bitflags;
use std::convert::TryFrom;
use std::fmt;

mod dims;
mod strides;

pub use dims::ZfpDims;
pub use strides::ZfpStrides;

// ---------------------------------------------------------------------------
// Bitstream word type
// ---------------------------------------------------------------------------

/// ZFP bitstream word type.
///
/// Mirrors the C reference implementation's `BIT_STREAM_WORD_TYPE`, which
/// defaults to `uint64` (`unsigned long long`). All bit-level I/O in the
/// library operates on 64-bit word units.
///
/// This is a type alias for `u64`, providing semantic documentation without
/// runtime overhead. Callers can use it to make their code self-documenting
/// when working with raw bitstream words.
///
/// For zero-copy conversion between `Vec<u8>` and `Vec<ZfpBitStreamWord>`,
/// use [`bytemuck`] when the byte buffer is 8-byte aligned. The crate
/// provides [`ZfpBitStream::from_bytes`](crate::bitstream::ZfpBitStream::from_bytes)
/// as a convenience that handles both aligned and unaligned buffers.
pub type ZfpBitStreamWord = u64;

// ---------------------------------------------------------------------------
// Metadata errors
// ---------------------------------------------------------------------------

/// Errors that can occur when encoding field metadata to a 52-bit word.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ZfpMetadataError {
    /// The field has no dimensions (dimensionality is 0).
    Null,
    /// One or more dimensions exceed the maximum encodable range.
    ///
    /// The 52-bit metadata word can represent:
    /// - 1-D: up to 2⁴⁸ elements
    /// - 2-D: up to 2²⁴ × 2²⁴ elements
    /// - 3-D: up to 2¹⁶ × 2¹⁶ × 2¹⁶ elements
    /// - 4-D: up to 2¹² × 2¹² × 2¹² × 2¹² elements
    DimensionTooLarge,
}

impl fmt::Display for ZfpMetadataError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ZfpMetadataError::Null => write!(f, "field metadata is null (no dimensions)"),
            ZfpMetadataError::DimensionTooLarge => {
                write!(
                    f,
                    "field dimension exceeds encodable range for 52-bit metadata"
                )
            }
        }
    }
}

impl std::error::Error for ZfpMetadataError {}

// ---------------------------------------------------------------------------
// Block errors
// ---------------------------------------------------------------------------

/// Errors that can occur during block-level encode/decode.
///
/// Returned by the public block codec functions in
/// [`codec::block`][crate::codec::block].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ZfpBlockError;

impl fmt::Display for ZfpBlockError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "data slice length does not match expected block size")
    }
}

impl std::error::Error for ZfpBlockError {}

// ---------------------------------------------------------------------------
// Compression/decompression errors
// ---------------------------------------------------------------------------

/// Errors that can occur during compression.
///
/// Returned by [`ZfpBitStream::compress`][crate::ZfpBitStream::compress],
/// [`ZfpBitStream::compress_with_execution`][crate::ZfpBitStream::compress_with_execution],
/// and `compress_bitstream` when the `ffi` feature is enabled.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ZfpCompressionError {
    /// The input field has no data buffer.
    ///
    /// This occurs when a `ZfpField` was created with empty data
    /// or from a null pointer or zero byte count.
    NoData,
    /// The input field's data buffer is smaller than its dimensions and
    /// strides require.
    ///
    /// Compressing such a field would read out of bounds, so it is rejected.
    InvalidField {
        /// Bytes spanned by the field's dimensions and strides. `usize::MAX`
        /// if the span itself overflows `usize`.
        required: usize,
        /// Bytes actually available in the field's data buffer.
        actual: usize,
    },
    /// The input field's data buffer is not aligned for its scalar type.
    ///
    /// The codec reinterprets the buffer as the scalar type and indexes it
    /// through raw pointers, so a misaligned buffer is rejected. Only
    /// reachable via `from_raw`, i.e. from the C ABI.
    MisalignedData {
        /// Alignment the scalar type requires, in bytes.
        align: usize,
    },
}

impl fmt::Display for ZfpCompressionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ZfpCompressionError::NoData => write!(f, "input field has no data buffer"),
            ZfpCompressionError::InvalidField { required, actual } => write!(
                f,
                "input field spans {required} bytes but its data buffer holds only {actual}"
            ),
            ZfpCompressionError::MisalignedData { align } => write!(
                f,
                "input field data buffer is not {align}-byte aligned for its scalar type"
            ),
        }
    }
}

impl std::error::Error for ZfpCompressionError {}

/// Errors that can occur during decompression.
///
/// Returned by [`ZfpBitStream::decompress`][crate::ZfpBitStream::decompress],
/// [`ZfpBitStream::decompress_with_execution`][crate::ZfpBitStream::decompress_with_execution],
/// and `decompress_bitstream` when the `ffi` feature is enabled.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ZfpDecompressionError {
    /// The output field has no data buffer.
    ///
    /// This occurs when a `ZfpFieldMut` was created from a null pointer or
    /// zero byte count.
    NoData,
    /// The output field's data buffer is smaller than its dimensions and
    /// strides require.
    ///
    /// Decompressing into such a field would write out of bounds, so it is
    /// rejected.
    InvalidField {
        /// Bytes spanned by the field's dimensions and strides. `usize::MAX`
        /// if the span itself overflows `usize`.
        required: usize,
        /// Bytes actually available in the field's data buffer.
        actual: usize,
    },
    /// The output field's data buffer is not aligned for its scalar type.
    ///
    /// The codec reinterprets the buffer as the scalar type and indexes it
    /// through raw pointers, so a misaligned buffer is rejected. Only
    /// reachable via `from_raw`, i.e. from the C ABI.
    MisalignedData {
        /// Alignment the scalar type requires, in bytes.
        align: usize,
    },
}

impl fmt::Display for ZfpDecompressionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ZfpDecompressionError::NoData => write!(f, "output field has no data buffer"),
            ZfpDecompressionError::InvalidField { required, actual } => write!(
                f,
                "output field spans {required} bytes but its data buffer holds only {actual}"
            ),
            ZfpDecompressionError::MisalignedData { align } => write!(
                f,
                "output field data buffer is not {align}-byte aligned for its scalar type"
            ),
        }
    }
}

impl std::error::Error for ZfpDecompressionError {}

// ---------------------------------------------------------------------------
// Numeric constants (mirror zfp.h macros)
// ---------------------------------------------------------------------------

pub const ZFP_MIN_BITS: u32 = 1;
pub const ZFP_MAX_BITS: u32 = 16658;
pub const ZFP_MAX_PREC: u32 = 64;
pub const ZFP_MIN_EXP: i32 = -1074;
pub const ZFP_MAGIC_BITS: u32 = 32;
pub const ZFP_META_BITS: u32 = 52;
pub const ZFP_MODE_SHORT_BITS: u32 = 12;
pub const ZFP_MODE_LONG_BITS: u32 = 64;
pub const ZFP_HEADER_MAX_BITS: u32 = 148;

// ---------------------------------------------------------------------------
// ZfpDimensionality
// ---------------------------------------------------------------------------

/// Block dimensionality: 1-D through 4-D.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ZfpDimensionality {
    D1 = 1,
    D2 = 2,
    D3 = 3,
    D4 = 4,
}

impl ZfpDimensionality {
    /// Return the number of elements in a block for this dimensionality.
    ///
    /// - `D1` → 4
    /// - `D2` → 16
    /// - `D3` → 64
    /// - `D4` → 256
    #[inline]
    #[must_use]
    pub const fn block_size(self) -> usize {
        4usize.pow(self as u32)
    }
}

impl From<ZfpDimensionality> for u32 {
    fn from(d: ZfpDimensionality) -> Self {
        d as u32
    }
}

impl From<ZfpDimensionality> for usize {
    fn from(d: ZfpDimensionality) -> Self {
        d as usize
    }
}

impl TryFrom<u32> for ZfpDimensionality {
    type Error = InvalidDimensionalityError;
    fn try_from(v: u32) -> Result<Self, Self::Error> {
        match v {
            1 => Ok(ZfpDimensionality::D1),
            2 => Ok(ZfpDimensionality::D2),
            3 => Ok(ZfpDimensionality::D3),
            4 => Ok(ZfpDimensionality::D4),
            _ => Err(InvalidDimensionalityError(v)),
        }
    }
}

// ---------------------------------------------------------------------------
// InvalidDimensionalityError
// ---------------------------------------------------------------------------

/// Error returned when a `u32` cannot be converted to [`ZfpDimensionality`].
///
/// Valid dimensionalities are 1 through 4.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct InvalidDimensionalityError(
    /// The invalid value that was supplied.
    pub u32,
);

impl fmt::Display for InvalidDimensionalityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid ZFP dimensionality: {} (expected 1–4)", self.0)
    }
}

impl std::error::Error for InvalidDimensionalityError {}

// ---------------------------------------------------------------------------
// ZfpScalarType
// ---------------------------------------------------------------------------

/// Scalar element type tag: mirrors `zfp_type` in `zfp.h`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ZfpScalarType {
    Int32,
    Int64,
    Float,
    Double,
}

impl ZfpScalarType {
    /// Number of bytes per scalar value.
    #[must_use]
    pub const fn size(&self) -> usize {
        match self {
            ZfpScalarType::Int32 | ZfpScalarType::Float => 4,
            ZfpScalarType::Int64 | ZfpScalarType::Double => 8,
        }
    }

    /// Alignment a buffer of this scalar type requires, in bytes.
    ///
    /// This is the target's alignment for the corresponding Rust type, which
    /// is not always [`size`][Self::size]: 64-bit scalars are 4-byte aligned
    /// on some 32-bit targets. Buffers passed to
    /// [`ZfpField::from_raw`][crate::ZfpField::from_raw] must satisfy this.
    #[must_use]
    pub const fn align(&self) -> usize {
        match self {
            ZfpScalarType::Int32 => align_of::<i32>(),
            ZfpScalarType::Int64 => align_of::<i64>(),
            ZfpScalarType::Float => align_of::<f32>(),
            ZfpScalarType::Double => align_of::<f64>(),
        }
    }

    /// Number of bits per scalar value (32 or 64).
    #[must_use]
    pub const fn precision(&self) -> u32 {
        match self {
            ZfpScalarType::Int32 | ZfpScalarType::Float => 32,
            ZfpScalarType::Int64 | ZfpScalarType::Double => 64,
        }
    }
}

impl fmt::Display for ZfpScalarType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ZfpScalarType::Int32 => write!(f, "int32"),
            ZfpScalarType::Int64 => write!(f, "int64"),
            ZfpScalarType::Float => write!(f, "float"),
            ZfpScalarType::Double => write!(f, "double"),
        }
    }
}

// ---------------------------------------------------------------------------
// ZfpMode
// ---------------------------------------------------------------------------

/// Compression mode: mirrors `zfp_mode` in `zfp.h`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ZfpMode {
    /// Invalid / unset.
    Null,
    /// All four expert parameters set manually.
    Expert,
    /// Fixed compressed bits per scalar.
    FixedRate,
    /// Fixed number of uncompressed bits per scalar.
    FixedPrecision,
    /// Absolute error tolerance.
    FixedAccuracy,
    /// Lossless.
    Reversible,
}

impl fmt::Display for ZfpMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ZfpMode::Null => write!(f, "null"),
            ZfpMode::Expert => write!(f, "expert"),
            ZfpMode::FixedRate => write!(f, "fixed-rate"),
            ZfpMode::FixedPrecision => write!(f, "fixed-precision"),
            ZfpMode::FixedAccuracy => write!(f, "fixed-accuracy"),
            ZfpMode::Reversible => write!(f, "reversible"),
        }
    }
}

// ---------------------------------------------------------------------------
// ZfpHeaderMask (bitflags)
// ---------------------------------------------------------------------------

bitflags! {
    /// Header section flags: mirrors `ZFP_HEADER_*` constants in `zfp.h`.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct ZfpHeaderMask: u32 {
        const NONE  = 0x0;
        /// 32-bit magic word.
        const MAGIC = 0x1;
        /// 52-bit field metadata.
        const META  = 0x2;
        /// 12- or 64-bit compression mode.
        const MODE  = 0x4;
        /// All of the above.
        const FULL  = 0x7;
    }
}

// ---------------------------------------------------------------------------
// ZfpScalar sealed trait
// ---------------------------------------------------------------------------

mod sealed {
    pub trait Sealed {}
    impl Sealed for i32 {}
    impl Sealed for i64 {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
}

/// Sealed trait for types ZFP can compress: `i32`, `i64`, `f32`, `f64`.
pub trait ZfpScalar: sealed::Sealed + Copy + Default + bytemuck::Pod + 'static {
    /// Return the [`ZfpScalarType`] tag for this primitive type.
    fn scalar_type() -> ZfpScalarType;
}

impl ZfpScalar for i32 {
    fn scalar_type() -> ZfpScalarType {
        ZfpScalarType::Int32
    }
}

impl ZfpScalar for i64 {
    fn scalar_type() -> ZfpScalarType {
        ZfpScalarType::Int64
    }
}

impl ZfpScalar for f32 {
    fn scalar_type() -> ZfpScalarType {
        ZfpScalarType::Float
    }
}

impl ZfpScalar for f64 {
    fn scalar_type() -> ZfpScalarType {
        ZfpScalarType::Double
    }
}
