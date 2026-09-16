//! Strided block entry points.
//!
//! Split out from the parent module so their visibility can be narrowed later:
//! they exist to serve the whole-field driver and the C ABI, and no safe Rust
//! caller has a use for them.

use super::{
    as_typed_block_1d, as_typed_block_1d_mut, as_typed_block_2d, as_typed_block_2d_mut,
    as_typed_block_3d, as_typed_block_3d_mut, as_typed_block_4d, as_typed_block_4d_mut,
};
use crate::bitstream::{ZfpBitStreamMutOps, ZfpBitStreamOps};
use crate::types::{ZfpDimensionality, ZfpScalar, ZfpScalarType};
use bytemuck::{cast_slice, cast_slice_mut};

// ---------------------------------------------------------------------------
// Dispatch macros
// ---------------------------------------------------------------------------

/// Expand the 4 scalar type x 4 dimensionality dispatch that every `*_strided*`
/// entry point shares.
///
/// `lengths` is absent for full blocks. `tail_int` and `tail_float` differ
/// because only the float codecs take `min_exp`.
macro_rules! strided_dispatch {
    (
        $bs:ident, $data:ident, $dims:ident, $strides:ident, $cast:ident,
        lengths: [$($len:ident)?],
        tail_int: [$($ti:expr),* $(,)?],
        tail_float: [$($tf:expr),* $(,)?],
        d1: [$i1:path, $q1:path, $f1:path, $g1:path $(,)?],
        d2: [$i2:path, $q2:path, $f2:path, $g2:path $(,)?],
        d3: [$i3:path, $q3:path, $f3:path, $g3:path $(,)?],
        d4: [$i4:path, $q4:path, $f4:path, $g4:path $(,)?] $(,)?
    ) => {
        match (T::scalar_type(), $dims) {
            (ZfpScalarType::Int32, ZfpDimensionality::D1) => {
                $i1($bs, $cast::<T, i32>($data), $($len[0],)? $strides[0], $($ti,)*)
            }
            (ZfpScalarType::Int64, ZfpDimensionality::D1) => {
                $q1($bs, $cast::<T, i64>($data), $($len[0],)? $strides[0], $($ti,)*)
            }
            (ZfpScalarType::Float, ZfpDimensionality::D1) => {
                $f1($bs, $cast::<T, f32>($data), $($len[0],)? $strides[0], $($tf,)*)
            }
            (ZfpScalarType::Double, ZfpDimensionality::D1) => {
                $g1($bs, $cast::<T, f64>($data), $($len[0],)? $strides[0], $($tf,)*)
            }
            (ZfpScalarType::Int32, ZfpDimensionality::D2) => {
                $i2($bs, $cast::<T, i32>($data), $($len[0], $len[1],)? $strides[0], $strides[1], $($ti,)*)
            }
            (ZfpScalarType::Int64, ZfpDimensionality::D2) => {
                $q2($bs, $cast::<T, i64>($data), $($len[0], $len[1],)? $strides[0], $strides[1], $($ti,)*)
            }
            (ZfpScalarType::Float, ZfpDimensionality::D2) => {
                $f2($bs, $cast::<T, f32>($data), $($len[0], $len[1],)? $strides[0], $strides[1], $($tf,)*)
            }
            (ZfpScalarType::Double, ZfpDimensionality::D2) => {
                $g2($bs, $cast::<T, f64>($data), $($len[0], $len[1],)? $strides[0], $strides[1], $($tf,)*)
            }
            (ZfpScalarType::Int32, ZfpDimensionality::D3) => {
                $i3($bs, $cast::<T, i32>($data), $($len[0], $len[1], $len[2],)? $strides[0], $strides[1], $strides[2], $($ti,)*)
            }
            (ZfpScalarType::Int64, ZfpDimensionality::D3) => {
                $q3($bs, $cast::<T, i64>($data), $($len[0], $len[1], $len[2],)? $strides[0], $strides[1], $strides[2], $($ti,)*)
            }
            (ZfpScalarType::Float, ZfpDimensionality::D3) => {
                $f3($bs, $cast::<T, f32>($data), $($len[0], $len[1], $len[2],)? $strides[0], $strides[1], $strides[2], $($tf,)*)
            }
            (ZfpScalarType::Double, ZfpDimensionality::D3) => {
                $g3($bs, $cast::<T, f64>($data), $($len[0], $len[1], $len[2],)? $strides[0], $strides[1], $strides[2], $($tf,)*)
            }
            (ZfpScalarType::Int32, ZfpDimensionality::D4) => {
                $i4($bs, $cast::<T, i32>($data), $($len[0], $len[1], $len[2], $len[3],)? $strides[0], $strides[1], $strides[2], $strides[3], $($ti,)*)
            }
            (ZfpScalarType::Int64, ZfpDimensionality::D4) => {
                $q4($bs, $cast::<T, i64>($data), $($len[0], $len[1], $len[2], $len[3],)? $strides[0], $strides[1], $strides[2], $strides[3], $($ti,)*)
            }
            (ZfpScalarType::Float, ZfpDimensionality::D4) => {
                $f4($bs, $cast::<T, f32>($data), $($len[0], $len[1], $len[2], $len[3],)? $strides[0], $strides[1], $strides[2], $strides[3], $($tf,)*)
            }
            (ZfpScalarType::Double, ZfpDimensionality::D4) => {
                $g4($bs, $cast::<T, f64>($data), $($len[0], $len[1], $len[2], $len[3],)? $strides[0], $strides[1], $strides[2], $strides[3], $($tf,)*)
            }
        }
    };
}

/// The block is sized from `dims`, so reinterpreting it cannot fail.
macro_rules! typed_block {
    ($e:expr) => {
        $e.unwrap_or_else(|_| unreachable!("block size matches dimensionality"))
    };
}

/// Expand the reversible dispatch over the contiguous block `gather_block`
/// produced (encode) or `scatter_block` will consume (decode).
macro_rules! reversible_dispatch {
    (
        encode $bs:ident, $dims:ident, $block:ident,
        d1: [$i1:path, $q1:path, $f1:path, $g1:path $(,)?],
        d2: [$i2:path, $q2:path, $f2:path, $g2:path $(,)?],
        d3: [$i3:path, $q3:path, $f3:path, $g3:path $(,)?],
        d4: [$i4:path, $q4:path, $f4:path, $g4:path $(,)?] $(,)?
    ) => {
        match (T::scalar_type(), $dims) {
            (ZfpScalarType::Int32, ZfpDimensionality::D1) => {
                $i1($bs, typed_block!(as_typed_block_1d::<T, i32>(&$block)))
            }
            (ZfpScalarType::Int64, ZfpDimensionality::D1) => {
                $q1($bs, typed_block!(as_typed_block_1d::<T, i64>(&$block)))
            }
            (ZfpScalarType::Float, ZfpDimensionality::D1) => {
                $f1($bs, typed_block!(as_typed_block_1d::<T, f32>(&$block)))
            }
            (ZfpScalarType::Double, ZfpDimensionality::D1) => {
                $g1($bs, typed_block!(as_typed_block_1d::<T, f64>(&$block)))
            }
            (ZfpScalarType::Int32, ZfpDimensionality::D2) => {
                $i2($bs, typed_block!(as_typed_block_2d::<T, i32>(&$block)))
            }
            (ZfpScalarType::Int64, ZfpDimensionality::D2) => {
                $q2($bs, typed_block!(as_typed_block_2d::<T, i64>(&$block)))
            }
            (ZfpScalarType::Float, ZfpDimensionality::D2) => {
                $f2($bs, typed_block!(as_typed_block_2d::<T, f32>(&$block)))
            }
            (ZfpScalarType::Double, ZfpDimensionality::D2) => {
                $g2($bs, typed_block!(as_typed_block_2d::<T, f64>(&$block)))
            }
            (ZfpScalarType::Int32, ZfpDimensionality::D3) => {
                $i3($bs, typed_block!(as_typed_block_3d::<T, i32>(&$block)))
            }
            (ZfpScalarType::Int64, ZfpDimensionality::D3) => {
                $q3($bs, typed_block!(as_typed_block_3d::<T, i64>(&$block)))
            }
            (ZfpScalarType::Float, ZfpDimensionality::D3) => {
                $f3($bs, typed_block!(as_typed_block_3d::<T, f32>(&$block)))
            }
            (ZfpScalarType::Double, ZfpDimensionality::D3) => {
                $g3($bs, typed_block!(as_typed_block_3d::<T, f64>(&$block)))
            }
            (ZfpScalarType::Int32, ZfpDimensionality::D4) => {
                $i4($bs, typed_block!(as_typed_block_4d::<T, i32>(&$block)))
            }
            (ZfpScalarType::Int64, ZfpDimensionality::D4) => {
                $q4($bs, typed_block!(as_typed_block_4d::<T, i64>(&$block)))
            }
            (ZfpScalarType::Float, ZfpDimensionality::D4) => {
                $f4($bs, typed_block!(as_typed_block_4d::<T, f32>(&$block)))
            }
            (ZfpScalarType::Double, ZfpDimensionality::D4) => {
                $g4($bs, typed_block!(as_typed_block_4d::<T, f64>(&$block)))
            }
        }
    };
    (
        decode $bs:ident, $dims:ident, $block:ident,
        d1: [$i1:path, $q1:path, $f1:path, $g1:path $(,)?],
        d2: [$i2:path, $q2:path, $f2:path, $g2:path $(,)?],
        d3: [$i3:path, $q3:path, $f3:path, $g3:path $(,)?],
        d4: [$i4:path, $q4:path, $f4:path, $g4:path $(,)?] $(,)?
    ) => {
        match (T::scalar_type(), $dims) {
            (ZfpScalarType::Int32, ZfpDimensionality::D1) => $i1(
                $bs,
                typed_block!(as_typed_block_1d_mut::<T, i32>(&mut $block)),
            ),
            (ZfpScalarType::Int64, ZfpDimensionality::D1) => $q1(
                $bs,
                typed_block!(as_typed_block_1d_mut::<T, i64>(&mut $block)),
            ),
            (ZfpScalarType::Float, ZfpDimensionality::D1) => $f1(
                $bs,
                typed_block!(as_typed_block_1d_mut::<T, f32>(&mut $block)),
            ),
            (ZfpScalarType::Double, ZfpDimensionality::D1) => $g1(
                $bs,
                typed_block!(as_typed_block_1d_mut::<T, f64>(&mut $block)),
            ),
            (ZfpScalarType::Int32, ZfpDimensionality::D2) => $i2(
                $bs,
                typed_block!(as_typed_block_2d_mut::<T, i32>(&mut $block)),
            ),
            (ZfpScalarType::Int64, ZfpDimensionality::D2) => $q2(
                $bs,
                typed_block!(as_typed_block_2d_mut::<T, i64>(&mut $block)),
            ),
            (ZfpScalarType::Float, ZfpDimensionality::D2) => $f2(
                $bs,
                typed_block!(as_typed_block_2d_mut::<T, f32>(&mut $block)),
            ),
            (ZfpScalarType::Double, ZfpDimensionality::D2) => $g2(
                $bs,
                typed_block!(as_typed_block_2d_mut::<T, f64>(&mut $block)),
            ),
            (ZfpScalarType::Int32, ZfpDimensionality::D3) => $i3(
                $bs,
                typed_block!(as_typed_block_3d_mut::<T, i32>(&mut $block)),
            ),
            (ZfpScalarType::Int64, ZfpDimensionality::D3) => $q3(
                $bs,
                typed_block!(as_typed_block_3d_mut::<T, i64>(&mut $block)),
            ),
            (ZfpScalarType::Float, ZfpDimensionality::D3) => $f3(
                $bs,
                typed_block!(as_typed_block_3d_mut::<T, f32>(&mut $block)),
            ),
            (ZfpScalarType::Double, ZfpDimensionality::D3) => $g3(
                $bs,
                typed_block!(as_typed_block_3d_mut::<T, f64>(&mut $block)),
            ),
            (ZfpScalarType::Int32, ZfpDimensionality::D4) => $i4(
                $bs,
                typed_block!(as_typed_block_4d_mut::<T, i32>(&mut $block)),
            ),
            (ZfpScalarType::Int64, ZfpDimensionality::D4) => $q4(
                $bs,
                typed_block!(as_typed_block_4d_mut::<T, i64>(&mut $block)),
            ),
            (ZfpScalarType::Float, ZfpDimensionality::D4) => $f4(
                $bs,
                typed_block!(as_typed_block_4d_mut::<T, f32>(&mut $block)),
            ),
            (ZfpScalarType::Double, ZfpDimensionality::D4) => $g4(
                $bs,
                typed_block!(as_typed_block_4d_mut::<T, f64>(&mut $block)),
            ),
        }
    };
}

// ---------------------------------------------------------------------------
// Strided block encode
// ---------------------------------------------------------------------------

/// Encode a strided 4^d block of scalars; return bits written.
///
/// # Safety
/// `data` must be valid for every offset the strides generate over the
/// block's extent. See the [`crate::codec::block`] module documentation.
pub unsafe fn encode_block_strided<T: ZfpScalar>(
    bs: &mut dyn ZfpBitStreamMutOps,
    data: &[T],
    dims: ZfpDimensionality,
    strides: &[isize],
) -> usize {
    unsafe {
        use crate::codec::encode::{dim1, dim2, dim3, dim4};
        strided_dispatch! {
            bs, data, dims, strides, cast_slice,
            lengths: [],
            tail_int: [],
            tail_float: [],
            d1: [
                dim1::encode_block_strided_1d_i32,
                dim1::encode_block_strided_1d_i64,
                dim1::encode_block_strided_1d_f32,
                dim1::encode_block_strided_1d_f64,
            ],
            d2: [
                dim2::encode_block_strided_2d_i32,
                dim2::encode_block_strided_2d_i64,
                dim2::encode_block_strided_2d_f32,
                dim2::encode_block_strided_2d_f64,
            ],
            d3: [
                dim3::encode_block_strided_3d_i32,
                dim3::encode_block_strided_3d_i64,
                dim3::encode_block_strided_3d_f32,
                dim3::encode_block_strided_3d_f64,
            ],
            d4: [
                dim4::encode_block_strided_4d_i32,
                dim4::encode_block_strided_4d_i64,
                dim4::encode_block_strided_4d_f32,
                dim4::encode_block_strided_4d_f64,
            ],
        }
    }
}

// ---------------------------------------------------------------------------
// Partial strided block encode
// ---------------------------------------------------------------------------

/// Encode a partial (boundary) strided block; return bits written.
///
/// # Safety
/// `data` must be valid for every offset the strides generate over the
/// block's extent. See the [`crate::codec::block`] module documentation.
pub unsafe fn encode_partial_block_strided<T: ZfpScalar>(
    bs: &mut dyn ZfpBitStreamMutOps,
    data: &[T],
    dims: ZfpDimensionality,
    lengths: &[usize],
    strides: &[isize],
) -> usize {
    unsafe {
        use crate::codec::encode::{dim1, dim2, dim3, dim4};
        strided_dispatch! {
            bs, data, dims, strides, cast_slice,
            lengths: [lengths],
            tail_int: [],
            tail_float: [],
            d1: [
                dim1::encode_partial_block_strided_1d_i32,
                dim1::encode_partial_block_strided_1d_i64,
                dim1::encode_partial_block_strided_1d_f32,
                dim1::encode_partial_block_strided_1d_f64,
            ],
            d2: [
                dim2::encode_partial_block_strided_2d_i32,
                dim2::encode_partial_block_strided_2d_i64,
                dim2::encode_partial_block_strided_2d_f32,
                dim2::encode_partial_block_strided_2d_f64,
            ],
            d3: [
                dim3::encode_partial_block_strided_3d_i32,
                dim3::encode_partial_block_strided_3d_i64,
                dim3::encode_partial_block_strided_3d_f32,
                dim3::encode_partial_block_strided_3d_f64,
            ],
            d4: [
                dim4::encode_partial_block_strided_4d_i32,
                dim4::encode_partial_block_strided_4d_i64,
                dim4::encode_partial_block_strided_4d_f32,
                dim4::encode_partial_block_strided_4d_f64,
            ],
        }
    }
}

// ---------------------------------------------------------------------------
// Strided block decode
// ---------------------------------------------------------------------------

/// Decode a strided 4^d block of scalars; return bits read.
#[allow(clippy::cast_possible_wrap, clippy::cast_sign_loss)] // usize↔isize for stride computation
/// Decode a strided 4^d block of scalars; return bits read.
///
/// # Safety
/// `data` must be valid for every offset the strides generate over the
/// block's extent. See the [`crate::codec::block`] module documentation.
pub unsafe fn decode_block_strided<T: ZfpScalar>(
    bs: &mut dyn ZfpBitStreamOps,
    data: &mut [T],
    dims: ZfpDimensionality,
    strides: &[isize],
) -> usize {
    unsafe {
        use crate::codec::decode::{dim1, dim2, dim3, dim4};
        strided_dispatch! {
            bs, data, dims, strides, cast_slice_mut,
            lengths: [],
            tail_int: [],
            tail_float: [],
            d1: [
                dim1::decode_block_strided_1d_i32,
                dim1::decode_block_strided_1d_i64,
                dim1::decode_block_strided_1d_f32,
                dim1::decode_block_strided_1d_f64,
            ],
            d2: [
                dim2::decode_block_strided_2d_i32,
                dim2::decode_block_strided_2d_i64,
                dim2::decode_block_strided_2d_f32,
                dim2::decode_block_strided_2d_f64,
            ],
            d3: [
                dim3::decode_block_strided_3d_i32,
                dim3::decode_block_strided_3d_i64,
                dim3::decode_block_strided_3d_f32,
                dim3::decode_block_strided_3d_f64,
            ],
            d4: [
                dim4::decode_block_strided_4d_i32,
                dim4::decode_block_strided_4d_i64,
                dim4::decode_block_strided_4d_f32,
                dim4::decode_block_strided_4d_f64,
            ],
        }
    }
}

// ---------------------------------------------------------------------------
// Partial strided block decode
// ---------------------------------------------------------------------------

/// Decode a partial (boundary) strided block; return bits read.
///
/// # Safety
/// `data` must be valid for every offset the strides generate over the
/// block's extent. See the [`crate::codec::block`] module documentation.
pub unsafe fn decode_partial_block_strided<T: ZfpScalar>(
    bs: &mut dyn ZfpBitStreamOps,
    data: &mut [T],
    dims: ZfpDimensionality,
    lengths: &[usize],
    strides: &[isize],
) -> usize {
    unsafe {
        use crate::codec::decode::{dim1, dim2, dim3, dim4};
        strided_dispatch! {
            bs, data, dims, strides, cast_slice_mut,
            lengths: [lengths],
            tail_int: [],
            tail_float: [],
            d1: [
                dim1::decode_partial_block_strided_1d_i32,
                dim1::decode_partial_block_strided_1d_i64,
                dim1::decode_partial_block_strided_1d_f32,
                dim1::decode_partial_block_strided_1d_f64,
            ],
            d2: [
                dim2::decode_partial_block_strided_2d_i32,
                dim2::decode_partial_block_strided_2d_i64,
                dim2::decode_partial_block_strided_2d_f32,
                dim2::decode_partial_block_strided_2d_f64,
            ],
            d3: [
                dim3::decode_partial_block_strided_3d_i32,
                dim3::decode_partial_block_strided_3d_i64,
                dim3::decode_partial_block_strided_3d_f32,
                dim3::decode_partial_block_strided_3d_f64,
            ],
            d4: [
                dim4::decode_partial_block_strided_4d_i32,
                dim4::decode_partial_block_strided_4d_i64,
                dim4::decode_partial_block_strided_4d_f32,
                dim4::decode_partial_block_strided_4d_f64,
            ],
        }
    }
}

// ---------------------------------------------------------------------------
// Reversible gather+encode helpers for compress
// ---------------------------------------------------------------------------

/// Gather a 4^d contiguous block from strided data.
///
/// `dims` is 1–4, `strides` has effective (non-zero) strides.
/// For partial blocks, `lengths` gives the count per dimension (≤ 4), and
/// elements outside the field boundary are padded with the nearest value.
#[allow(clippy::cast_possible_wrap, clippy::cast_sign_loss)] // usize↔isize for stride computation
unsafe fn gather_block<T: ZfpScalar>(
    data: &[T],
    dims: ZfpDimensionality,
    strides: &[isize],
    lengths: [usize; 4],
) -> Vec<T> {
    // Map a block index `i` (0..4) with `n` valid elements to the source data index.
    // Mirrors C's `pad_block`: position 3 always comes from position 0; positions
    // n..2 come from position n-1; positions 0..n are the real data.
    fn pad_idx(i: isize, n: isize) -> isize {
        if i < n {
            i
        } else if i == 3 {
            0
        } else {
            n - 1
        }
    }

    let block_size = 4usize.pow(u32::from(dims));
    let mut block = vec![T::default(); block_size];
    match dims {
        ZfpDimensionality::D1 => {
            let sx = strides[0];
            let lx = lengths[0] as isize;
            let p = data.as_ptr();
            for x in 0..4isize {
                let px = pad_idx(x, lx);
                // SAFETY: px is a valid source index within [0, lx-1].
                block[x as usize] = unsafe { *p.offset(px * sx) };
            }
        }
        ZfpDimensionality::D2 => {
            let sx = strides[0];
            let sy = strides[1];
            let lx = lengths[0] as isize;
            let ly = lengths[1] as isize;
            let p = data.as_ptr();
            let mut i = 0;
            for y in 0..4isize {
                let py = pad_idx(y, ly);
                for x in 0..4isize {
                    let px = pad_idx(x, lx);
                    // SAFETY: px, py are valid source indices.
                    block[i] = unsafe { *p.offset(px * sx + py * sy) };
                    i += 1;
                }
            }
        }
        ZfpDimensionality::D3 => {
            let sx = strides[0];
            let sy = strides[1];
            let sz = strides[2];
            let lx = lengths[0] as isize;
            let ly = lengths[1] as isize;
            let lz = lengths[2] as isize;
            let p = data.as_ptr();
            let mut i = 0;
            for z in 0..4isize {
                let pz = pad_idx(z, lz);
                for y in 0..4isize {
                    let py = pad_idx(y, ly);
                    for x in 0..4isize {
                        let px = pad_idx(x, lx);
                        // SAFETY: px, py, pz are valid source indices.
                        block[i] = unsafe { *p.offset(px * sx + py * sy + pz * sz) };
                        i += 1;
                    }
                }
            }
        }
        ZfpDimensionality::D4 => {
            let sx = strides[0];
            let sy = strides[1];
            let sz = strides[2];
            let sw = strides[3];
            let lx = lengths[0] as isize;
            let ly = lengths[1] as isize;
            let lz = lengths[2] as isize;
            let lw = lengths[3] as isize;
            let p = data.as_ptr();
            let mut i = 0;
            for w in 0..4isize {
                let pw = pad_idx(w, lw);
                for z in 0..4isize {
                    let pz = pad_idx(z, lz);
                    for y in 0..4isize {
                        let py = pad_idx(y, ly);
                        for x in 0..4isize {
                            let px = pad_idx(x, lx);
                            // SAFETY: px, py, pz, pw are valid source indices.
                            block[i] = unsafe { *p.offset(px * sx + py * sy + pz * sz + pw * sw) };
                            i += 1;
                        }
                    }
                }
            }
        }
    }
    block
}

/// Scatter a 4^d contiguous block back into strided data.
///
/// Only the `lengths` elements in each dimension are written; padding elements
/// are discarded.
#[allow(clippy::cast_possible_wrap, clippy::cast_sign_loss)] // usize↔isize for stride computation
unsafe fn scatter_block<T: ZfpScalar>(
    block: &[T],
    data: &mut [T],
    dims: ZfpDimensionality,
    strides: &[isize],
    lengths: [usize; 4],
) {
    match dims {
        ZfpDimensionality::D1 => {
            let sx = strides[0];
            let lx = lengths[0];
            let p = data.as_mut_ptr();
            for (x, &v) in block[..lx].iter().enumerate() {
                unsafe { *p.offset(x as isize * sx) = v };
            }
        }
        ZfpDimensionality::D2 => {
            let sx = strides[0];
            let sy = strides[1];
            let lx = lengths[0];
            let ly = lengths[1];
            let p = data.as_mut_ptr();
            let mut i = 0;
            for y in 0..4 {
                for x in 0..4 {
                    if x < lx && y < ly {
                        unsafe { *p.offset(x as isize * sx + y as isize * sy) = block[i] };
                    }
                    i += 1;
                }
            }
        }
        ZfpDimensionality::D3 => {
            let sx = strides[0];
            let sy = strides[1];
            let sz = strides[2];
            let lx = lengths[0];
            let ly = lengths[1];
            let lz = lengths[2];
            let p = data.as_mut_ptr();
            let mut i = 0;
            for z in 0..4 {
                for y in 0..4 {
                    for x in 0..4 {
                        if x < lx && y < ly && z < lz {
                            unsafe {
                                *p.offset(x as isize * sx + y as isize * sy + z as isize * sz) =
                                    block[i];
                            }
                        }
                        i += 1;
                    }
                }
            }
        }
        ZfpDimensionality::D4 => {
            let sx = strides[0];
            let sy = strides[1];
            let sz = strides[2];
            let sw = strides[3];
            let lx = lengths[0];
            let ly = lengths[1];
            let lz = lengths[2];
            let lw = lengths[3];
            let p = data.as_mut_ptr();
            let mut i = 0;
            for w in 0..4 {
                for z in 0..4 {
                    for y in 0..4 {
                        for x in 0..4 {
                            if x < lx && y < ly && z < lz && w < lw {
                                unsafe {
                                    *p.offset(
                                        x as isize * sx
                                            + y as isize * sy
                                            + z as isize * sz
                                            + w as isize * sw,
                                    ) = block[i];
                                }
                            }
                            i += 1;
                        }
                    }
                }
            }
        }
    }
}

/// Reversible encode of a (possibly partial) strided 4^d block; return bits written.
///
/// Used by `ZfpBitStream::compress` in reversible mode.
///
/// # Safety
/// `data` must be valid for every offset the strides generate over the
/// block's extent. See the [`crate::codec::block`] module documentation.
pub unsafe fn encode_block_strided_reversible<T: ZfpScalar>(
    bs: &mut dyn ZfpBitStreamMutOps,
    data: &[T],
    dims: ZfpDimensionality,
    strides: &[isize],
    lengths: [usize; 4],
) -> usize {
    unsafe {
        use crate::codec::encode::reversible as rev;

        let block = gather_block(data, dims, strides, lengths);
        reversible_dispatch! {
            encode bs, dims, block,
            d1: [
                rev::encode_block_reversible_1d_i32,
                rev::encode_block_reversible_1d_i64,
                rev::encode_block_reversible_1d_f32,
                rev::encode_block_reversible_1d_f64,
            ],
            d2: [
                rev::encode_block_reversible_2d_i32,
                rev::encode_block_reversible_2d_i64,
                rev::encode_block_reversible_2d_f32,
                rev::encode_block_reversible_2d_f64,
            ],
            d3: [
                rev::encode_block_reversible_3d_i32,
                rev::encode_block_reversible_3d_i64,
                rev::encode_block_reversible_3d_f32,
                rev::encode_block_reversible_3d_f64,
            ],
            d4: [
                rev::encode_block_reversible_4d_i32,
                rev::encode_block_reversible_4d_i64,
                rev::encode_block_reversible_4d_f32,
                rev::encode_block_reversible_4d_f64,
            ],
        }
    }
}

/// Reversible decode of a (possibly partial) strided 4^d block; return bits read.
///
/// Used by `ZfpBitStream::decompress` in reversible mode.
///
/// # Safety
/// `data` must be valid for every offset the strides generate over the
/// block's extent. See the [`crate::codec::block`] module documentation.
pub unsafe fn decode_block_strided_reversible<T: ZfpScalar>(
    bs: &mut dyn ZfpBitStreamOps,
    data: &mut [T],
    dims: ZfpDimensionality,
    strides: &[isize],
    lengths: [usize; 4],
) -> usize {
    unsafe {
        use crate::codec::decode::reversible as rev;

        let mut block = vec![T::default(); 4usize.pow(u32::from(dims))];
        let bits = reversible_dispatch! {
            decode bs, dims, block,
            d1: [
                rev::decode_block_reversible_1d_i32,
                rev::decode_block_reversible_1d_i64,
                rev::decode_block_reversible_1d_f32,
                rev::decode_block_reversible_1d_f64,
            ],
            d2: [
                rev::decode_block_reversible_2d_i32,
                rev::decode_block_reversible_2d_i64,
                rev::decode_block_reversible_2d_f32,
                rev::decode_block_reversible_2d_f64,
            ],
            d3: [
                rev::decode_block_reversible_3d_i32,
                rev::decode_block_reversible_3d_i64,
                rev::decode_block_reversible_3d_f32,
                rev::decode_block_reversible_3d_f64,
            ],
            d4: [
                rev::decode_block_reversible_4d_i32,
                rev::decode_block_reversible_4d_i64,
                rev::decode_block_reversible_4d_f32,
                rev::decode_block_reversible_4d_f64,
            ],
        };

        scatter_block(&block, data, dims, strides, lengths);
        bits
    }
}

// ---------------------------------------------------------------------------
// Parametric strided block encode: used by ZfpBitStream::compress
// ---------------------------------------------------------------------------

/// Encode a strided 4^d block with explicit stream parameters; return bits written.
///
/// Unlike `encode_block_strided`, this uses the caller-supplied `min_bits`,
/// `max_bits`, `max_prec`, and `min_exp` instead of the lossless defaults.
///
/// # Safety
/// `data` must be valid for every offset the strides generate over the
/// block's extent. See the [`crate::codec::block`] module documentation.
pub unsafe fn encode_block_strided_with_params<T: ZfpScalar>(
    bs: &mut dyn ZfpBitStreamMutOps,
    data: &[T],
    dims: ZfpDimensionality,
    strides: &[isize],
    min_bits: u32,
    max_bits: u32,
    max_prec: u32,
    min_exp: i32,
) -> usize {
    unsafe {
        use crate::codec::encode::{dim1, dim2, dim3, dim4};
        strided_dispatch! {
            bs, data, dims, strides, cast_slice,
            lengths: [],
            tail_int: [min_bits, max_bits, max_prec],
            tail_float: [min_bits, max_bits, max_prec, min_exp],
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
/// # Safety
/// `data` must be valid for every offset the strides generate over the
/// block's extent. See the [`crate::codec::block`] module documentation.
pub unsafe fn encode_partial_block_strided_with_params<T: ZfpScalar>(
    bs: &mut dyn ZfpBitStreamMutOps,
    data: &[T],
    dims: ZfpDimensionality,
    lengths: &[usize],
    strides: &[isize],
    min_bits: u32,
    max_bits: u32,
    max_prec: u32,
    min_exp: i32,
) -> usize {
    unsafe {
        use crate::codec::encode::{dim1, dim2, dim3, dim4};
        strided_dispatch! {
            bs, data, dims, strides, cast_slice,
            lengths: [lengths],
            tail_int: [min_bits, max_bits, max_prec],
            tail_float: [min_bits, max_bits, max_prec, min_exp],
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
/// # Safety
/// `data` must be valid for every offset the strides generate over the
/// block's extent. See the [`crate::codec::block`] module documentation.
pub unsafe fn decode_block_strided_with_params<T: ZfpScalar>(
    bs: &mut dyn ZfpBitStreamOps,
    data: &mut [T],
    dims: ZfpDimensionality,
    strides: &[isize],
    min_bits: u32,
    max_bits: u32,
    max_prec: u32,
    min_exp: i32,
) -> usize {
    unsafe {
        use crate::codec::decode::{dim1, dim2, dim3, dim4};
        strided_dispatch! {
            bs, data, dims, strides, cast_slice_mut,
            lengths: [],
            tail_int: [min_bits, max_bits, max_prec],
            tail_float: [min_bits, max_bits, max_prec, min_exp],
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
/// # Safety
/// `data` must be valid for every offset the strides generate over the
/// block's extent. See the [`crate::codec::block`] module documentation.
pub unsafe fn decode_partial_block_strided_with_params<T: ZfpScalar>(
    bs: &mut dyn ZfpBitStreamOps,
    data: &mut [T],
    dims: ZfpDimensionality,
    lengths: &[usize],
    strides: &[isize],
    min_bits: u32,
    max_bits: u32,
    max_prec: u32,
    min_exp: i32,
) -> usize {
    unsafe {
        use crate::codec::decode::{dim1, dim2, dim3, dim4};
        strided_dispatch! {
            bs, data, dims, strides, cast_slice_mut,
            lengths: [lengths],
            tail_int: [min_bits, max_bits, max_prec],
            tail_float: [min_bits, max_bits, max_prec, min_exp],
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
