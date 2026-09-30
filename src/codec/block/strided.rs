//! Strided block entry points.
//!
//! Split out from the parent module so their visibility can be narrowed to the
//! crate when the `ffi` feature is off: they exist to serve the whole-field
//! driver and the C ABI, and no safe Rust caller has a use for them.
//!
//! Every function here is `unsafe` and takes `data` as a raw pointer to the
//! block origin, mirroring the C `zfp_{en,de}code_block_strided_*` API.
//!
//! # Preconditions
//!
//! [`super::encode_block`] and [`super::decode_block`] take a slice and
//! validate its length. The `*_strided` variants take a raw pointer instead:
//! `data` points at the block origin, and the gather/scatter helpers index it
//! as `*data.offset(x*sx + y*sy + z*sz + w*sw)`. Non-unit strides step beyond
//! the block's element count and negative strides step backwards from the
//! origin, so no slice could describe the memory they touch — a slice
//! reference would carry provenance over its own elements only, and indexing
//! outside it is undefined behaviour even when the allocation extends that far.
//!
//! The caller must therefore guarantee that every offset the strides generate
//! is in bounds of a single allocation, and that `data`'s provenance covers it
//! — in practice, by deriving `data` from a pointer to the whole buffer.
//! Callers that cannot should use
//! [`ZfpBitStream::compress`][crate::ZfpBitStream::compress] and
//! [`ZfpBitStream::decompress`][crate::ZfpBitStream::decompress], which
//! validate the field's index span and alignment against its buffer and derive
//! the pointer accordingly.

use super::as_typed_block_mut;
use crate::bitstream::{ZfpBitStreamMutOps, ZfpBitStreamOps};
use crate::config::ZfpConfig;
use crate::types::{ZfpBlockError, ZfpDimensionality, ZfpScalar, ZfpScalarType};

/// Reinterpret a scalar pointer as the concrete type the enclosing match arm has
/// already proven `T` to be.
///
/// A mismatch is unreachable: every caller sits inside a
/// `match (T::SCALAR_TYPE, dims)` arm that pins `T == U`, and `ZfpScalar` is
/// sealed to `{i32, i64, f32, f64}`. The asserts check it in debug builds.
#[inline]
fn cast_ptr<T: ZfpScalar, U: ZfpScalar>(p: *const T) -> *const U {
    debug_assert_eq!(size_of::<T>(), size_of::<U>());
    debug_assert_eq!(align_of::<T>(), align_of::<U>());
    p.cast::<U>()
}

/// Mutable counterpart of [`cast_ptr`].
#[inline]
fn cast_ptr_mut<T: ZfpScalar, U: ZfpScalar>(p: *mut T) -> *mut U {
    debug_assert_eq!(size_of::<T>(), size_of::<U>());
    debug_assert_eq!(align_of::<T>(), align_of::<U>());
    p.cast::<U>()
}

// ---------------------------------------------------------------------------
// Dispatch macros
// ---------------------------------------------------------------------------

/// Expand the 4 scalar type x 4 dimensionality dispatch that every `*_strided*`
/// entry point shares.
///
/// `lengths` is absent for full blocks.
macro_rules! strided_dispatch {
    (
        $bs:ident, $data:ident, $dims:ident, $strides:ident, $cast:ident, $config:ident,
        lengths: [$($len:ident)?],
        d1: [$i1:path, $q1:path, $f1:path, $g1:path $(,)?],
        d2: [$i2:path, $q2:path, $f2:path, $g2:path $(,)?],
        d3: [$i3:path, $q3:path, $f3:path, $g3:path $(,)?],
        d4: [$i4:path, $q4:path, $f4:path, $g4:path $(,)?] $(,)?
    ) => {
        match (T::SCALAR_TYPE, $dims) {
            (ZfpScalarType::I32, ZfpDimensionality::D1) => {
                $i1($bs, $cast::<T, i32>($data), $($len[0],)? $strides[0], $config)
            }
            (ZfpScalarType::I64, ZfpDimensionality::D1) => {
                $q1($bs, $cast::<T, i64>($data), $($len[0],)? $strides[0], $config)
            }
            (ZfpScalarType::F32, ZfpDimensionality::D1) => {
                $f1($bs, $cast::<T, f32>($data), $($len[0],)? $strides[0], $config)
            }
            (ZfpScalarType::F64, ZfpDimensionality::D1) => {
                $g1($bs, $cast::<T, f64>($data), $($len[0],)? $strides[0], $config)
            }
            (ZfpScalarType::I32, ZfpDimensionality::D2) => {
                $i2($bs, $cast::<T, i32>($data), $($len[0], $len[1],)? $strides[0], $strides[1], $config)
            }
            (ZfpScalarType::I64, ZfpDimensionality::D2) => {
                $q2($bs, $cast::<T, i64>($data), $($len[0], $len[1],)? $strides[0], $strides[1], $config)
            }
            (ZfpScalarType::F32, ZfpDimensionality::D2) => {
                $f2($bs, $cast::<T, f32>($data), $($len[0], $len[1],)? $strides[0], $strides[1], $config)
            }
            (ZfpScalarType::F64, ZfpDimensionality::D2) => {
                $g2($bs, $cast::<T, f64>($data), $($len[0], $len[1],)? $strides[0], $strides[1], $config)
            }
            (ZfpScalarType::I32, ZfpDimensionality::D3) => {
                $i3($bs, $cast::<T, i32>($data), $($len[0], $len[1], $len[2],)? $strides[0], $strides[1], $strides[2], $config)
            }
            (ZfpScalarType::I64, ZfpDimensionality::D3) => {
                $q3($bs, $cast::<T, i64>($data), $($len[0], $len[1], $len[2],)? $strides[0], $strides[1], $strides[2], $config)
            }
            (ZfpScalarType::F32, ZfpDimensionality::D3) => {
                $f3($bs, $cast::<T, f32>($data), $($len[0], $len[1], $len[2],)? $strides[0], $strides[1], $strides[2], $config)
            }
            (ZfpScalarType::F64, ZfpDimensionality::D3) => {
                $g3($bs, $cast::<T, f64>($data), $($len[0], $len[1], $len[2],)? $strides[0], $strides[1], $strides[2], $config)
            }
            (ZfpScalarType::I32, ZfpDimensionality::D4) => {
                $i4($bs, $cast::<T, i32>($data), $($len[0], $len[1], $len[2], $len[3],)? $strides[0], $strides[1], $strides[2], $strides[3], $config)
            }
            (ZfpScalarType::I64, ZfpDimensionality::D4) => {
                $q4($bs, $cast::<T, i64>($data), $($len[0], $len[1], $len[2], $len[3],)? $strides[0], $strides[1], $strides[2], $strides[3], $config)
            }
            (ZfpScalarType::F32, ZfpDimensionality::D4) => {
                $f4($bs, $cast::<T, f32>($data), $($len[0], $len[1], $len[2], $len[3],)? $strides[0], $strides[1], $strides[2], $strides[3], $config)
            }
            (ZfpScalarType::F64, ZfpDimensionality::D4) => {
                $g4($bs, $cast::<T, f64>($data), $($len[0], $len[1], $len[2], $len[3],)? $strides[0], $strides[1], $strides[2], $strides[3], $config)
            }
        }
    };
}

/// The block is sized from `dims`, and the enclosing match arm pins `T` to the
/// target type, so reinterpreting it cannot fail. If it did, the block would
/// code nothing.
macro_rules! typed_block {
    ($e:expr) => {
        match $e {
            Ok(block) => block,
            Err(ZfpBlockError) => return 0,
        }
    };
}

/// Expand the reversible dispatch over the contiguous block `with_gathered`
/// produced (encode) or `with_scattered` will scatter (decode). Each
/// dimensionality lists its block length and its coders for `i32`, `i64`,
/// `f32` and `f64`.
macro_rules! reversible_dispatch {
    (
        $bs:ident, $dims:ident, $block:ident, $config:ident,
        $($d:ident($n:literal): [$i:path, $q:path, $f:path, $g:path $(,)?]),+ $(,)?
    ) => {
        match (T::SCALAR_TYPE, $dims) {
            $(
                (ZfpScalarType::I32, ZfpDimensionality::$d) => {
                    $i($bs, typed_block!(as_typed_block_mut::<T, i32, $n>(&mut *$block)), $config)
                }
                (ZfpScalarType::I64, ZfpDimensionality::$d) => {
                    $q($bs, typed_block!(as_typed_block_mut::<T, i64, $n>(&mut *$block)), $config)
                }
                (ZfpScalarType::F32, ZfpDimensionality::$d) => {
                    $f($bs, typed_block!(as_typed_block_mut::<T, f32, $n>(&mut *$block)), $config)
                }
                (ZfpScalarType::F64, ZfpDimensionality::$d) => {
                    $g($bs, typed_block!(as_typed_block_mut::<T, f64, $n>(&mut *$block)), $config)
                }
            )+
        }
    };
}

// ---------------------------------------------------------------------------
// Reversible gather+encode helpers for compress
// ---------------------------------------------------------------------------

/// Run `f` on the 4^d block gathered from strided data.
///
/// `dims` is 1–4, `strides` has effective (non-zero) strides.
/// For partial blocks, `lengths` gives the count per dimension (1–4), and
/// elements outside the field boundary are padded as C's `pad_block` does, by
/// the same gathers the lossy coder uses.
///
/// # Safety
/// Each of the first `dims` entries of `lengths` must be in 1..=4, and `data`
/// must be valid for every offset the strides generate over those lengths. See
/// the [`crate::codec::block`] module documentation.
unsafe fn with_gathered<T: ZfpScalar, R>(
    data: *const T,
    dims: ZfpDimensionality,
    strides: &[isize; 4],
    lengths: [usize; 4],
    f: impl FnOnce(&mut [T]) -> R,
) -> R {
    use crate::codec::encode::{dim1, dim2, dim3, dim4};

    let [sx, sy, sz, sw] = *strides;
    let [lx, ly, lz, lw] = lengths;
    // A whole block has nothing to pad, and its gather has fixed bounds.
    let whole = lengths
        .iter()
        .take(usize::from(dims))
        .all(|&length| length == 4);
    // SAFETY: the caller's contract, which bounds the lengths as the partial
    // gathers require.
    unsafe {
        match dims {
            ZfpDimensionality::D1 if whole => f(&mut dim1::gather_1d(data, sx)),
            ZfpDimensionality::D1 => f(&mut dim1::gather_partial_1d(data, lx, sx)),
            ZfpDimensionality::D2 if whole => f(&mut dim2::gather_2d(data, sx, sy)),
            ZfpDimensionality::D2 => f(&mut dim2::gather_partial_2d(data, lx, ly, sx, sy)),
            ZfpDimensionality::D3 if whole => f(&mut dim3::gather_3d(data, sx, sy, sz)),
            ZfpDimensionality::D3 => f(&mut dim3::gather_partial_3d(data, lx, ly, lz, sx, sy, sz)),
            ZfpDimensionality::D4 if whole => f(&mut dim4::gather_4d(data, sx, sy, sz, sw)),
            ZfpDimensionality::D4 => f(&mut dim4::gather_partial_4d(
                data, lx, ly, lz, lw, sx, sy, sz, sw,
            )),
        }
    }
}

/// Run `f` on a 4^d block, then scatter it into strided data.
///
/// `dims` is 1–4, `strides` has effective (non-zero) strides. Only the
/// `lengths` elements in each dimension are written, by the same scatters the
/// lossy coder uses; padding elements are discarded. `f` must fill the block.
///
/// # Safety
/// Each of the first `dims` entries of `lengths` must be in 1..=4, and `data`
/// must be valid for every offset the strides generate over those lengths. See
/// the [`crate::codec::block`] module documentation.
unsafe fn with_scattered<T: ZfpScalar, R>(
    data: *mut T,
    dims: ZfpDimensionality,
    strides: &[isize; 4],
    lengths: [usize; 4],
    f: impl FnOnce(&mut [T]) -> R,
) -> R {
    use crate::codec::decode::{dim1, dim2, dim3, dim4};

    let [sx, sy, sz, sw] = *strides;
    let [lx, ly, lz, lw] = lengths;
    // A whole block has nothing to discard, and its scatter has fixed bounds.
    let whole = lengths
        .iter()
        .take(usize::from(dims))
        .all(|&length| length == 4);
    // SAFETY: the caller's contract, which bounds the lengths as the partial
    // scatters require.
    unsafe {
        match dims {
            ZfpDimensionality::D1 => {
                let mut block = [T::default(); 4];
                let r = f(&mut block);
                if whole {
                    dim1::scatter_1d(&block, data, sx);
                } else {
                    dim1::scatter_partial_1d(&block, data, lx, sx);
                }
                r
            }
            ZfpDimensionality::D2 => {
                let mut block = [T::default(); 16];
                let r = f(&mut block);
                if whole {
                    dim2::scatter_2d(&block, data, sx, sy);
                } else {
                    dim2::scatter_partial_2d(&block, data, lx, ly, sx, sy);
                }
                r
            }
            ZfpDimensionality::D3 => {
                let mut block = [T::default(); 64];
                let r = f(&mut block);
                if whole {
                    dim3::scatter_3d(&block, data, sx, sy, sz);
                } else {
                    dim3::scatter_partial_3d(&block, data, lx, ly, lz, sx, sy, sz);
                }
                r
            }
            ZfpDimensionality::D4 => {
                let mut block = [T::default(); 256];
                let r = f(&mut block);
                if whole {
                    dim4::scatter_4d(&block, data, sx, sy, sz, sw);
                } else {
                    dim4::scatter_partial_4d(&block, data, lx, ly, lz, lw, sx, sy, sz, sw);
                }
                r
            }
        }
    }
}

/// Reversible encode of a (possibly partial) strided 4^d block; return bits
/// written.
///
/// # Safety
/// `data` must be valid for every offset the strides generate over the
/// block's extent, and every length on an axis below `dims` must be in
/// `1..=4`. See the `codec::block::strided` module documentation.
pub(crate) unsafe fn encode_reversible<T: ZfpScalar>(
    bs: &mut (impl ZfpBitStreamMutOps + ?Sized),
    data: *const T,
    dims: ZfpDimensionality,
    strides: &[isize; 4],
    lengths: [usize; 4],
    config: &ZfpConfig,
) -> usize {
    unsafe {
        use crate::codec::encode::reversible as rev;

        with_gathered(data, dims, strides, lengths, |block: &mut [T]| {
            reversible_dispatch! {
                bs, dims, block, config,
                D1(4): [
                    rev::encode_block_reversible_1d_i32,
                    rev::encode_block_reversible_1d_i64,
                    rev::encode_block_reversible_1d_f32,
                    rev::encode_block_reversible_1d_f64,
                ],
                D2(16): [
                    rev::encode_block_reversible_2d_i32,
                    rev::encode_block_reversible_2d_i64,
                    rev::encode_block_reversible_2d_f32,
                    rev::encode_block_reversible_2d_f64,
                ],
                D3(64): [
                    rev::encode_block_reversible_3d_i32,
                    rev::encode_block_reversible_3d_i64,
                    rev::encode_block_reversible_3d_f32,
                    rev::encode_block_reversible_3d_f64,
                ],
                D4(256): [
                    rev::encode_block_reversible_4d_i32,
                    rev::encode_block_reversible_4d_i64,
                    rev::encode_block_reversible_4d_f32,
                    rev::encode_block_reversible_4d_f64,
                ],
            }
        })
    }
}

/// Reversible decode of a (possibly partial) strided 4^d block; return bits
/// read.
///
/// # Safety
/// `data` must be valid for every offset the strides generate over the
/// block's extent, and every length on an axis below `dims` must be in
/// `1..=4`. See the `codec::block::strided` module documentation.
pub(crate) unsafe fn decode_reversible<T: ZfpScalar>(
    bs: &mut (impl ZfpBitStreamOps + ?Sized),
    data: *mut T,
    dims: ZfpDimensionality,
    strides: &[isize; 4],
    lengths: [usize; 4],
    config: &ZfpConfig,
) -> usize {
    unsafe {
        use crate::codec::decode::reversible as rev;

        with_scattered(data, dims, strides, lengths, |block: &mut [T]| {
            reversible_dispatch! {
                bs, dims, block, config,
                D1(4): [
                    rev::decode_block_reversible_1d_i32,
                    rev::decode_block_reversible_1d_i64,
                    rev::decode_block_reversible_1d_f32,
                    rev::decode_block_reversible_1d_f64,
                ],
                D2(16): [
                    rev::decode_block_reversible_2d_i32,
                    rev::decode_block_reversible_2d_i64,
                    rev::decode_block_reversible_2d_f32,
                    rev::decode_block_reversible_2d_f64,
                ],
                D3(64): [
                    rev::decode_block_reversible_3d_i32,
                    rev::decode_block_reversible_3d_i64,
                    rev::decode_block_reversible_3d_f32,
                    rev::decode_block_reversible_3d_f64,
                ],
                D4(256): [
                    rev::decode_block_reversible_4d_i32,
                    rev::decode_block_reversible_4d_i64,
                    rev::decode_block_reversible_4d_f32,
                    rev::decode_block_reversible_4d_f64,
                ],
            }
        })
    }
}

/// The lengths of a whole 4^d block, for the reversible coder.
fn whole(dims: ZfpDimensionality) -> [usize; 4] {
    std::array::from_fn(|axis| if axis < usize::from(dims) { 4 } else { 0 })
}

/// Check that a partial block has from 1 to 4 values on each of its axes.
#[cfg(feature = "ffi")]
fn check_lengths(dims: ZfpDimensionality, lengths: [usize; 4]) -> Result<(), ZfpBlockError> {
    if lengths
        .iter()
        .take(usize::from(dims))
        .all(|n| (1..=4).contains(n))
    {
        Ok(())
    } else {
        Err(ZfpBlockError)
    }
}

// ---------------------------------------------------------------------------
// Parametric strided block encode: used by ZfpBitStream::compress
// ---------------------------------------------------------------------------

/// Encode a strided 4^d block with explicit stream parameters; return bits written.
///
/// A reversible `config` selects lossless coding.
///
/// # Safety
/// `data` must be valid for every offset the strides generate over the
/// block's extent. See the `codec::block::strided` module documentation.
#[doc(hidden)]
pub unsafe fn encode_block_strided<T: ZfpScalar>(
    bs: &mut (impl ZfpBitStreamMutOps + ?Sized),
    data: *const T,
    dims: ZfpDimensionality,
    strides: &[isize; 4],
    config: &ZfpConfig,
) -> usize {
    if config.is_reversible() {
        // SAFETY: the caller's contract, and whole blocks have valid lengths.
        return unsafe { encode_reversible(bs, data, dims, strides, whole(dims), config) };
    }
    // SAFETY: the caller's contract.
    unsafe { encode_lossy(bs, data, dims, strides, config) }
}

/// [`encode_block_strided`] for a lossy `config`.
///
/// # Safety
/// As for [`encode_block_strided`].
pub(crate) unsafe fn encode_lossy<T: ZfpScalar>(
    bs: &mut (impl ZfpBitStreamMutOps + ?Sized),
    data: *const T,
    dims: ZfpDimensionality,
    strides: &[isize; 4],
    config: &ZfpConfig,
) -> usize {
    unsafe {
        use crate::codec::encode::{dim1, dim2, dim3, dim4};
        strided_dispatch! {
            bs, data, dims, strides, cast_ptr, config,
            lengths: [],
            d1: [
                dim1::encode_block_strided_1d_i32_rate,
                dim1::encode_block_strided_1d_i64_rate,
                dim1::encode_block_strided_1d_f32_rate,
                dim1::encode_block_strided_1d_f64_rate,
            ],
            d2: [
                dim2::encode_block_strided_2d_i32_rate,
                dim2::encode_block_strided_2d_i64_rate,
                dim2::encode_block_strided_2d_f32_rate,
                dim2::encode_block_strided_2d_f64_rate,
            ],
            d3: [
                dim3::encode_block_strided_3d_i32_rate,
                dim3::encode_block_strided_3d_i64_rate,
                dim3::encode_block_strided_3d_f32_rate,
                dim3::encode_block_strided_3d_f64_rate,
            ],
            d4: [
                dim4::encode_block_strided_4d_i32_rate,
                dim4::encode_block_strided_4d_i64_rate,
                dim4::encode_block_strided_4d_f32_rate,
                dim4::encode_block_strided_4d_f64_rate,
            ],
        }
    }
}

/// Encode a partial (boundary) strided block with explicit stream parameters; return bits written.
///
/// A reversible `config` selects lossless coding.
///
/// # Errors
///
/// Returns [`ZfpBlockError`] if a length on an axis below `dims` is not in
/// `1..=4`. Lengths on the other axes are ignored.
///
/// # Safety
/// `data` must be valid for every offset the strides generate over the
/// block's extent. See the `codec::block::strided` module documentation.
#[cfg(feature = "ffi")]
#[doc(hidden)]
pub unsafe fn encode_partial_block_strided<T: ZfpScalar>(
    bs: &mut (impl ZfpBitStreamMutOps + ?Sized),
    data: *const T,
    dims: ZfpDimensionality,
    lengths: [usize; 4],
    strides: &[isize; 4],
    config: &ZfpConfig,
) -> Result<usize, ZfpBlockError> {
    check_lengths(dims, lengths)?;
    // SAFETY: the caller's contract, with the lengths checked.
    Ok(unsafe {
        if config.is_reversible() {
            encode_reversible(bs, data, dims, strides, lengths, config)
        } else {
            encode_lossy_partial(bs, data, dims, lengths, strides, config)
        }
    })
}

/// [`encode_partial_block_strided`] for a lossy `config` and lengths known to
/// be valid.
///
/// # Safety
/// As for [`encode_partial_block_strided`], and every length on an axis below
/// `dims` must be in `1..=4`.
pub(crate) unsafe fn encode_lossy_partial<T: ZfpScalar>(
    bs: &mut (impl ZfpBitStreamMutOps + ?Sized),
    data: *const T,
    dims: ZfpDimensionality,
    lengths: [usize; 4],
    strides: &[isize; 4],
    config: &ZfpConfig,
) -> usize {
    unsafe {
        use crate::codec::encode::{dim1, dim2, dim3, dim4};
        strided_dispatch! {
            bs, data, dims, strides, cast_ptr, config,
            lengths: [lengths],
            d1: [
                dim1::encode_partial_block_strided_1d_i32_rate,
                dim1::encode_partial_block_strided_1d_i64_rate,
                dim1::encode_partial_block_strided_1d_f32_rate,
                dim1::encode_partial_block_strided_1d_f64_rate,
            ],
            d2: [
                dim2::encode_partial_block_strided_2d_i32_rate,
                dim2::encode_partial_block_strided_2d_i64_rate,
                dim2::encode_partial_block_strided_2d_f32_rate,
                dim2::encode_partial_block_strided_2d_f64_rate,
            ],
            d3: [
                dim3::encode_partial_block_strided_3d_i32_rate,
                dim3::encode_partial_block_strided_3d_i64_rate,
                dim3::encode_partial_block_strided_3d_f32_rate,
                dim3::encode_partial_block_strided_3d_f64_rate,
            ],
            d4: [
                dim4::encode_partial_block_strided_4d_i32_rate,
                dim4::encode_partial_block_strided_4d_i64_rate,
                dim4::encode_partial_block_strided_4d_f32_rate,
                dim4::encode_partial_block_strided_4d_f64_rate,
            ],
        }
    }
}

// ---------------------------------------------------------------------------
// Parametric strided block decode: used by ZfpBitStream::decompress
// ---------------------------------------------------------------------------

/// Decode a strided 4^d block with explicit stream parameters; return bits read.
///
/// A reversible `config` selects lossless coding.
///
/// # Safety
/// `data` must be valid for every offset the strides generate over the
/// block's extent. See the `codec::block::strided` module documentation.
#[doc(hidden)]
pub unsafe fn decode_block_strided<T: ZfpScalar>(
    bs: &mut (impl ZfpBitStreamOps + ?Sized),
    data: *mut T,
    dims: ZfpDimensionality,
    strides: &[isize; 4],
    config: &ZfpConfig,
) -> usize {
    if config.is_reversible() {
        // SAFETY: the caller's contract, and whole blocks have valid lengths.
        return unsafe { decode_reversible(bs, data, dims, strides, whole(dims), config) };
    }
    // SAFETY: the caller's contract.
    unsafe { decode_lossy(bs, data, dims, strides, config) }
}

/// [`decode_block_strided`] for a lossy `config`.
///
/// # Safety
/// As for [`decode_block_strided`].
pub(crate) unsafe fn decode_lossy<T: ZfpScalar>(
    bs: &mut (impl ZfpBitStreamOps + ?Sized),
    data: *mut T,
    dims: ZfpDimensionality,
    strides: &[isize; 4],
    config: &ZfpConfig,
) -> usize {
    unsafe {
        use crate::codec::decode::{dim1, dim2, dim3, dim4};
        strided_dispatch! {
            bs, data, dims, strides, cast_ptr_mut, config,
            lengths: [],
            d1: [
                dim1::decode_block_strided_1d_i32_rate,
                dim1::decode_block_strided_1d_i64_rate,
                dim1::decode_block_strided_1d_f32_rate,
                dim1::decode_block_strided_1d_f64_rate,
            ],
            d2: [
                dim2::decode_block_strided_2d_i32_rate,
                dim2::decode_block_strided_2d_i64_rate,
                dim2::decode_block_strided_2d_f32_rate,
                dim2::decode_block_strided_2d_f64_rate,
            ],
            d3: [
                dim3::decode_block_strided_3d_i32_rate,
                dim3::decode_block_strided_3d_i64_rate,
                dim3::decode_block_strided_3d_f32_rate,
                dim3::decode_block_strided_3d_f64_rate,
            ],
            d4: [
                dim4::decode_block_strided_4d_i32_rate,
                dim4::decode_block_strided_4d_i64_rate,
                dim4::decode_block_strided_4d_f32_rate,
                dim4::decode_block_strided_4d_f64_rate,
            ],
        }
    }
}

/// Decode a partial (boundary) strided block with explicit stream parameters; return bits read.
///
/// A reversible `config` selects lossless coding.
///
/// # Errors
///
/// Returns [`ZfpBlockError`] if a length on an axis below `dims` is not in
/// `1..=4`. Lengths on the other axes are ignored.
///
/// # Safety
/// `data` must be valid for every offset the strides generate over the
/// block's extent. See the `codec::block::strided` module documentation.
#[cfg(feature = "ffi")]
#[doc(hidden)]
pub unsafe fn decode_partial_block_strided<T: ZfpScalar>(
    bs: &mut (impl ZfpBitStreamOps + ?Sized),
    data: *mut T,
    dims: ZfpDimensionality,
    lengths: [usize; 4],
    strides: &[isize; 4],
    config: &ZfpConfig,
) -> Result<usize, ZfpBlockError> {
    check_lengths(dims, lengths)?;
    // SAFETY: the caller's contract, with the lengths checked.
    Ok(unsafe {
        if config.is_reversible() {
            decode_reversible(bs, data, dims, strides, lengths, config)
        } else {
            decode_lossy_partial(bs, data, dims, lengths, strides, config)
        }
    })
}

/// [`decode_partial_block_strided`] for a lossy `config` and lengths known to
/// be valid.
///
/// # Safety
/// As for [`decode_partial_block_strided`], and every length on an axis below
/// `dims` must be in `1..=4`.
pub(crate) unsafe fn decode_lossy_partial<T: ZfpScalar>(
    bs: &mut (impl ZfpBitStreamOps + ?Sized),
    data: *mut T,
    dims: ZfpDimensionality,
    lengths: [usize; 4],
    strides: &[isize; 4],
    config: &ZfpConfig,
) -> usize {
    unsafe {
        use crate::codec::decode::{dim1, dim2, dim3, dim4};
        strided_dispatch! {
            bs, data, dims, strides, cast_ptr_mut, config,
            lengths: [lengths],
            d1: [
                dim1::decode_partial_block_strided_1d_i32_rate,
                dim1::decode_partial_block_strided_1d_i64_rate,
                dim1::decode_partial_block_strided_1d_f32_rate,
                dim1::decode_partial_block_strided_1d_f64_rate,
            ],
            d2: [
                dim2::decode_partial_block_strided_2d_i32_rate,
                dim2::decode_partial_block_strided_2d_i64_rate,
                dim2::decode_partial_block_strided_2d_f32_rate,
                dim2::decode_partial_block_strided_2d_f64_rate,
            ],
            d3: [
                dim3::decode_partial_block_strided_3d_i32_rate,
                dim3::decode_partial_block_strided_3d_i64_rate,
                dim3::decode_partial_block_strided_3d_f32_rate,
                dim3::decode_partial_block_strided_3d_f64_rate,
            ],
            d4: [
                dim4::decode_partial_block_strided_4d_i32_rate,
                dim4::decode_partial_block_strided_4d_i64_rate,
                dim4::decode_partial_block_strided_4d_f32_rate,
                dim4::decode_partial_block_strided_4d_f64_rate,
            ],
        }
    }
}

#[cfg(test)]
#[allow(clippy::arithmetic_side_effects, reason = "test offsets are small")]
mod tests {
    use super::*;
    use crate::codec::encode::{dim1, dim2, dim3, dim4};

    /// Where C's `pad_block` takes position `i` of a run of four with `n` real
    /// values from: position 3 repeats position 0, and positions from `n` up to
    /// 2 repeat position `n - 1`.
    fn padded_source(i: usize, n: usize) -> usize {
        if i < n {
            i
        } else if i == 3 {
            0
        } else {
            n - 1
        }
    }

    /// The block C builds for a partial block, one axis at a time: the value at
    /// `[x, y, z, w]` comes from `[pad(x), pad(y), pad(z), pad(w)]`.
    ///
    /// # Safety
    /// `data` must be valid for every offset the strides generate over `lengths`.
    unsafe fn expected(
        data: *const f64,
        dims: ZfpDimensionality,
        strides: [isize; 4],
        lengths: [usize; 4],
    ) -> Vec<f64> {
        (0..dims.block_size())
            .map(|i| {
                let offset: isize = (0..usize::from(dims))
                    .map(|axis| {
                        let coordinate = (i >> (2 * axis)) & 3;
                        padded_source(coordinate, lengths[axis]).cast_signed() * strides[axis]
                    })
                    .sum();
                unsafe { *data.offset(offset) }
            })
            .collect()
    }

    /// Both partial-block gathers, the lossy coder's and the reversible
    /// coder's, build C's padded block for every combination of lengths, along
    /// ascending and descending strides.
    #[test]
    fn partial_gathers_pad_as_c_does_for_every_length() {
        let data: Vec<f64> = (0..2048).map(|i| f64::from(i) + 0.5).collect();
        for strides in [[1isize, 7, 53, 401], [-1, -7, -53, -401]] {
            // Three steps along each axis, which the longest partial block
            // takes, fit the span from the origin: for descending strides the
            // origin is at the top.
            let origin: isize = strides.iter().filter(|&&s| s < 0).map(|&s| -3 * s).sum();
            // SAFETY: `origin` and the steps from it stay inside `data`.
            let ptr = unsafe { data.as_ptr().offset(origin) };
            for dims in [
                ZfpDimensionality::D1,
                ZfpDimensionality::D2,
                ZfpDimensionality::D3,
                ZfpDimensionality::D4,
            ] {
                let rank = usize::from(dims);
                for combination in 0..4usize.pow(u32::from(dims)) {
                    let mut lengths = [0usize; 4];
                    for (axis, length) in lengths.iter_mut().take(rank).enumerate() {
                        *length = ((combination >> (2 * axis)) & 3) + 1;
                    }
                    let [nx, ny, nz, nw] = lengths;
                    let [sx, sy, sz, sw] = strides;
                    let context = format!("{dims:?} {lengths:?} {strides:?}");

                    // SAFETY: as for `origin` above.
                    unsafe {
                        let want = expected(ptr, dims, strides, lengths);

                        let reversible =
                            with_gathered(ptr, dims, &strides, lengths, |block| block.to_vec());
                        assert_eq!(reversible, want, "reversible: {context}");

                        let lossy = match dims {
                            ZfpDimensionality::D1 => dim1::gather_partial_1d(ptr, nx, sx).to_vec(),
                            ZfpDimensionality::D2 => {
                                dim2::gather_partial_2d(ptr, nx, ny, sx, sy).to_vec()
                            }
                            ZfpDimensionality::D3 => {
                                dim3::gather_partial_3d(ptr, nx, ny, nz, sx, sy, sz).to_vec()
                            }
                            ZfpDimensionality::D4 => {
                                dim4::gather_partial_4d(ptr, nx, ny, nz, nw, sx, sy, sz, sw)
                                    .to_vec()
                            }
                        };
                        assert_eq!(lossy, want, "lossy: {context}");
                    }
                }
            }
        }
    }
}
