//! Scalar helpers shared by the fuzz target bodies.

use zfp_rs::types::{ZfpScalar, ZfpScalarType};

/// Extra operations the fuzz targets need on top of [`ZfpScalar`].
pub trait FuzzScalar: ZfpScalar + std::fmt::Debug {
    /// Width of this scalar in bytes.
    const SIZE: usize;

    /// Build a value from the first [`SIZE`][Self::SIZE] bytes of `chunk`,
    /// interpreted in native byte order.
    fn from_ne_chunk(chunk: [u8; 8]) -> Self;

    /// Raw bit pattern, widened to `u64`.
    ///
    /// All equality checks in the targets go through this. Comparing floats
    /// directly would make the reversible-mode oracle both too weak
    /// (`0.0 == -0.0`) and flaky (`NaN != NaN`).
    fn to_bits_u64(self) -> u64;

    /// The value as an `f64`, or `None` for integer types.
    fn as_f64(self) -> Option<f64>;

    /// Exponent of the smallest positive subnormal of this type.
    ///
    /// `0` for integer types, which never reach the checks that use it.
    const MIN_EXPONENT: i32;

    /// Smallest positive *normal* value of this type, as an `f64`.
    ///
    /// `f64::MAX` for integer types, so error-bound checks skip them.
    const MIN_POSITIVE_NORMAL: f64;

    /// Whether the value is subnormal **in its own width**.
    ///
    /// Checking after widening to `f64` is not equivalent and silently misses
    /// every subnormal `f32`: `f32::MIN_POSITIVE / 2` is a perfectly normal
    /// `f64`. Error-bound oracles need the native answer.
    fn is_subnormal_native(self) -> bool;
}

impl FuzzScalar for i32 {
    const SIZE: usize = 4;
    const MIN_POSITIVE_NORMAL: f64 = f64::MAX;
    const MIN_EXPONENT: i32 = 0;

    fn from_ne_chunk(chunk: [u8; 8]) -> Self {
        Self::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])
    }

    fn to_bits_u64(self) -> u64 {
        u64::from(self.cast_unsigned())
    }

    fn as_f64(self) -> Option<f64> {
        None
    }

    fn is_subnormal_native(self) -> bool {
        false
    }
}

impl FuzzScalar for i64 {
    const SIZE: usize = 8;
    const MIN_POSITIVE_NORMAL: f64 = f64::MAX;
    const MIN_EXPONENT: i32 = 0;

    fn from_ne_chunk(chunk: [u8; 8]) -> Self {
        Self::from_ne_bytes(chunk)
    }

    fn to_bits_u64(self) -> u64 {
        self.cast_unsigned()
    }

    fn as_f64(self) -> Option<f64> {
        None
    }

    fn is_subnormal_native(self) -> bool {
        false
    }
}

impl FuzzScalar for f32 {
    const SIZE: usize = 4;
    const MIN_POSITIVE_NORMAL: f64 = f32::MIN_POSITIVE as f64;
    const MIN_EXPONENT: i32 = -149;

    fn from_ne_chunk(chunk: [u8; 8]) -> Self {
        Self::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])
    }

    fn to_bits_u64(self) -> u64 {
        u64::from(self.to_bits())
    }

    fn as_f64(self) -> Option<f64> {
        Some(f64::from(self))
    }

    fn is_subnormal_native(self) -> bool {
        self.is_subnormal()
    }
}

impl FuzzScalar for f64 {
    const SIZE: usize = 8;
    const MIN_POSITIVE_NORMAL: f64 = f64::MIN_POSITIVE;
    const MIN_EXPONENT: i32 = -1074;

    fn from_ne_chunk(chunk: [u8; 8]) -> Self {
        Self::from_ne_bytes(chunk)
    }

    fn to_bits_u64(self) -> u64 {
        self.to_bits()
    }

    fn as_f64(self) -> Option<f64> {
        Some(self)
    }

    fn is_subnormal_native(self) -> bool {
        self.is_subnormal()
    }
}

/// Decode exactly `n` scalars by cycling `payload`.
///
/// Cycling (rather than truncating) means a short payload still produces a
/// full field, so the fuzzer can reach large shapes without having to discover
/// a proportionally large input. An empty payload yields all zeros.
///
/// Note that this deliberately produces **subnormals, NaN and infinities** —
/// the inverse of the `normal_f32()` / `normal_f64()` filters in
/// `tests/proptest/`, which exist only because the C reference's `fwd_cast`
/// overflow is implementation-defined. The targets here have no C oracle, so
/// those values are in scope and are exactly what makes fuzzing worthwhile.
#[must_use]
pub fn decode_scalars<T: FuzzScalar>(payload: &[u8], n: usize) -> Vec<T> {
    if payload.is_empty() {
        return vec![T::default(); n];
    }
    let mut out = Vec::with_capacity(n);
    let mut cursor = 0usize;
    for _ in 0..n {
        let mut chunk = [0u8; 8];
        for byte in chunk.iter_mut().take(T::SIZE) {
            *byte = payload[cursor % payload.len()];
            cursor += 1;
        }
        out.push(T::from_ne_chunk(chunk));
    }
    out
}

/// Scalar width in bytes for a runtime type tag.
#[must_use]
pub fn type_size(ty: ZfpScalarType) -> usize {
    match ty {
        ZfpScalarType::Int32 | ZfpScalarType::Float => 4,
        ZfpScalarType::Int64 | ZfpScalarType::Double => 8,
    }
}
