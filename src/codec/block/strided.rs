//! Strided block entry points.
//!
//! Split out from the parent module so their visibility can be narrowed to the
//! crate when the `ffi` feature is off: they exist to serve the whole-field
//! driver and the C ABI, and no safe Rust caller has a use for them.
//!
//! Every function here is `unsafe` and takes `data` as a raw pointer to the
//! block origin, mirroring the C `zfp_{en,de}code_block_strided_*` API. See the
//! [`crate::codec::block`] docs for the provenance the caller must supply.

use super::{
    as_typed_block_1d, as_typed_block_1d_mut, as_typed_block_2d, as_typed_block_2d_mut,
    as_typed_block_3d, as_typed_block_3d_mut, as_typed_block_4d, as_typed_block_4d_mut,
};
use crate::bitstream::{ZfpBitStreamMutOps, ZfpBitStreamOps};
use crate::config::ZfpConfig;
use crate::types::{ZFP_MIN_EXP, ZfpDimensionality, ZfpScalar, ZfpScalarType};

/// Reinterpret a scalar pointer as the concrete type the enclosing match arm has
/// already proven `T` to be.
///
/// Replaces the `bytemuck::cast_slice` calls this module used before the
/// signatures became raw pointers. That checked size and alignment at runtime
/// and panicked on a mismatch; the asserts keep the check in debug builds. A
/// mismatch is unreachable: every caller sits inside a
/// `match (T::SCALAR_TYPE, dims)` arm that pins `T == U`, and `ZfpScalar` is
/// sealed to `{i32, i64, f32, f64}`.
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
        encode $bs:ident, $dims:ident, $block:ident, $config:ident,
        d1: [$i1:path, $q1:path, $f1:path, $g1:path $(,)?],
        d2: [$i2:path, $q2:path, $f2:path, $g2:path $(,)?],
        d3: [$i3:path, $q3:path, $f3:path, $g3:path $(,)?],
        d4: [$i4:path, $q4:path, $f4:path, $g4:path $(,)?] $(,)?
    ) => {
        match (T::SCALAR_TYPE, $dims) {
            (ZfpScalarType::I32, ZfpDimensionality::D1) => $i1(
                $bs,
                typed_block!(as_typed_block_1d::<T, i32>(&$block)),
                $config,
            ),
            (ZfpScalarType::I64, ZfpDimensionality::D1) => $q1(
                $bs,
                typed_block!(as_typed_block_1d::<T, i64>(&$block)),
                $config,
            ),
            (ZfpScalarType::F32, ZfpDimensionality::D1) => $f1(
                $bs,
                typed_block!(as_typed_block_1d::<T, f32>(&$block)),
                $config,
            ),
            (ZfpScalarType::F64, ZfpDimensionality::D1) => $g1(
                $bs,
                typed_block!(as_typed_block_1d::<T, f64>(&$block)),
                $config,
            ),
            (ZfpScalarType::I32, ZfpDimensionality::D2) => $i2(
                $bs,
                typed_block!(as_typed_block_2d::<T, i32>(&$block)),
                $config,
            ),
            (ZfpScalarType::I64, ZfpDimensionality::D2) => $q2(
                $bs,
                typed_block!(as_typed_block_2d::<T, i64>(&$block)),
                $config,
            ),
            (ZfpScalarType::F32, ZfpDimensionality::D2) => $f2(
                $bs,
                typed_block!(as_typed_block_2d::<T, f32>(&$block)),
                $config,
            ),
            (ZfpScalarType::F64, ZfpDimensionality::D2) => $g2(
                $bs,
                typed_block!(as_typed_block_2d::<T, f64>(&$block)),
                $config,
            ),
            (ZfpScalarType::I32, ZfpDimensionality::D3) => $i3(
                $bs,
                typed_block!(as_typed_block_3d::<T, i32>(&$block)),
                $config,
            ),
            (ZfpScalarType::I64, ZfpDimensionality::D3) => $q3(
                $bs,
                typed_block!(as_typed_block_3d::<T, i64>(&$block)),
                $config,
            ),
            (ZfpScalarType::F32, ZfpDimensionality::D3) => $f3(
                $bs,
                typed_block!(as_typed_block_3d::<T, f32>(&$block)),
                $config,
            ),
            (ZfpScalarType::F64, ZfpDimensionality::D3) => $g3(
                $bs,
                typed_block!(as_typed_block_3d::<T, f64>(&$block)),
                $config,
            ),
            (ZfpScalarType::I32, ZfpDimensionality::D4) => $i4(
                $bs,
                typed_block!(as_typed_block_4d::<T, i32>(&$block)),
                $config,
            ),
            (ZfpScalarType::I64, ZfpDimensionality::D4) => $q4(
                $bs,
                typed_block!(as_typed_block_4d::<T, i64>(&$block)),
                $config,
            ),
            (ZfpScalarType::F32, ZfpDimensionality::D4) => $f4(
                $bs,
                typed_block!(as_typed_block_4d::<T, f32>(&$block)),
                $config,
            ),
            (ZfpScalarType::F64, ZfpDimensionality::D4) => $g4(
                $bs,
                typed_block!(as_typed_block_4d::<T, f64>(&$block)),
                $config,
            ),
        }
    };
    (
        decode $bs:ident, $dims:ident, $block:ident, $config:ident,
        d1: [$i1:path, $q1:path, $f1:path, $g1:path $(,)?],
        d2: [$i2:path, $q2:path, $f2:path, $g2:path $(,)?],
        d3: [$i3:path, $q3:path, $f3:path, $g3:path $(,)?],
        d4: [$i4:path, $q4:path, $f4:path, $g4:path $(,)?] $(,)?
    ) => {
        match (T::SCALAR_TYPE, $dims) {
            (ZfpScalarType::I32, ZfpDimensionality::D1) => $i1(
                $bs,
                typed_block!(as_typed_block_1d_mut::<T, i32>(&mut $block)),
                $config,
            ),
            (ZfpScalarType::I64, ZfpDimensionality::D1) => $q1(
                $bs,
                typed_block!(as_typed_block_1d_mut::<T, i64>(&mut $block)),
                $config,
            ),
            (ZfpScalarType::F32, ZfpDimensionality::D1) => $f1(
                $bs,
                typed_block!(as_typed_block_1d_mut::<T, f32>(&mut $block)),
                $config,
            ),
            (ZfpScalarType::F64, ZfpDimensionality::D1) => $g1(
                $bs,
                typed_block!(as_typed_block_1d_mut::<T, f64>(&mut $block)),
                $config,
            ),
            (ZfpScalarType::I32, ZfpDimensionality::D2) => $i2(
                $bs,
                typed_block!(as_typed_block_2d_mut::<T, i32>(&mut $block)),
                $config,
            ),
            (ZfpScalarType::I64, ZfpDimensionality::D2) => $q2(
                $bs,
                typed_block!(as_typed_block_2d_mut::<T, i64>(&mut $block)),
                $config,
            ),
            (ZfpScalarType::F32, ZfpDimensionality::D2) => $f2(
                $bs,
                typed_block!(as_typed_block_2d_mut::<T, f32>(&mut $block)),
                $config,
            ),
            (ZfpScalarType::F64, ZfpDimensionality::D2) => $g2(
                $bs,
                typed_block!(as_typed_block_2d_mut::<T, f64>(&mut $block)),
                $config,
            ),
            (ZfpScalarType::I32, ZfpDimensionality::D3) => $i3(
                $bs,
                typed_block!(as_typed_block_3d_mut::<T, i32>(&mut $block)),
                $config,
            ),
            (ZfpScalarType::I64, ZfpDimensionality::D3) => $q3(
                $bs,
                typed_block!(as_typed_block_3d_mut::<T, i64>(&mut $block)),
                $config,
            ),
            (ZfpScalarType::F32, ZfpDimensionality::D3) => $f3(
                $bs,
                typed_block!(as_typed_block_3d_mut::<T, f32>(&mut $block)),
                $config,
            ),
            (ZfpScalarType::F64, ZfpDimensionality::D3) => $g3(
                $bs,
                typed_block!(as_typed_block_3d_mut::<T, f64>(&mut $block)),
                $config,
            ),
            (ZfpScalarType::I32, ZfpDimensionality::D4) => $i4(
                $bs,
                typed_block!(as_typed_block_4d_mut::<T, i32>(&mut $block)),
                $config,
            ),
            (ZfpScalarType::I64, ZfpDimensionality::D4) => $q4(
                $bs,
                typed_block!(as_typed_block_4d_mut::<T, i64>(&mut $block)),
                $config,
            ),
            (ZfpScalarType::F32, ZfpDimensionality::D4) => $f4(
                $bs,
                typed_block!(as_typed_block_4d_mut::<T, f32>(&mut $block)),
                $config,
            ),
            (ZfpScalarType::F64, ZfpDimensionality::D4) => $g4(
                $bs,
                typed_block!(as_typed_block_4d_mut::<T, f64>(&mut $block)),
                $config,
            ),
        }
    };
}

// ---------------------------------------------------------------------------
// Reversible gather+encode helpers for compress
// ---------------------------------------------------------------------------

/// Gather a 4^d contiguous block from strided data.
///
/// `dims` is 1–4, `strides` has effective (non-zero) strides.
/// For partial blocks, `lengths` gives the count per dimension (≤ 4), and
/// elements outside the field boundary are padded with the nearest value.
///
/// # Safety
/// `data` must be valid for every offset the strides generate over the
/// block's extent. See the [`crate::codec::block`] module documentation.
#[allow(clippy::cast_possible_wrap, clippy::cast_sign_loss)] // usize↔isize for stride computation
unsafe fn gather_block<T: ZfpScalar>(
    data: *const T,
    dims: ZfpDimensionality,
    strides: &[isize],
    lengths: [usize; 4],
    block: &mut [T],
) {
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

    debug_assert_eq!(block.len(), dims.block_size());
    match dims {
        ZfpDimensionality::D1 => {
            let sx = strides[0];
            let lx = lengths[0] as isize;
            for x in 0..4isize {
                let px = pad_idx(x, lx);
                // SAFETY: px is a valid source index within [0, lx-1].
                block[x as usize] = unsafe { *data.offset(px * sx) };
            }
        }
        ZfpDimensionality::D2 => {
            let sx = strides[0];
            let sy = strides[1];
            let lx = lengths[0] as isize;
            let ly = lengths[1] as isize;
            let mut i = 0;
            for y in 0..4isize {
                let py = pad_idx(y, ly);
                for x in 0..4isize {
                    let px = pad_idx(x, lx);
                    // SAFETY: px, py are valid source indices.
                    block[i] = unsafe { *data.offset(px * sx + py * sy) };
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
            let mut i = 0;
            for z in 0..4isize {
                let pz = pad_idx(z, lz);
                for y in 0..4isize {
                    let py = pad_idx(y, ly);
                    for x in 0..4isize {
                        let px = pad_idx(x, lx);
                        // SAFETY: px, py, pz are valid source indices.
                        block[i] = unsafe { *data.offset(px * sx + py * sy + pz * sz) };
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
                            block[i] =
                                unsafe { *data.offset(px * sx + py * sy + pz * sz + pw * sw) };
                            i += 1;
                        }
                    }
                }
            }
        }
    }
}

/// Scatter a 4^d contiguous block back into strided data.
///
/// Only the `lengths` elements in each dimension are written; padding elements
/// are discarded.
///
/// # Safety
/// `data` must be valid for every offset the strides generate over the
/// block's extent. See the [`crate::codec::block`] module documentation.
#[allow(clippy::cast_possible_wrap, clippy::cast_sign_loss)] // usize↔isize for stride computation
unsafe fn scatter_block<T: ZfpScalar>(
    block: &[T],
    data: *mut T,
    dims: ZfpDimensionality,
    strides: &[isize],
    lengths: [usize; 4],
) {
    match dims {
        ZfpDimensionality::D1 => {
            let sx = strides[0];
            let lx = lengths[0];
            for (x, &v) in block[..lx].iter().enumerate() {
                unsafe { *data.offset(x as isize * sx) = v };
            }
        }
        ZfpDimensionality::D2 => {
            let sx = strides[0];
            let sy = strides[1];
            let lx = lengths[0];
            let ly = lengths[1];
            let mut i = 0;
            for y in 0..4 {
                for x in 0..4 {
                    if x < lx && y < ly {
                        unsafe { *data.offset(x as isize * sx + y as isize * sy) = block[i] };
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
            let mut i = 0;
            for z in 0..4 {
                for y in 0..4 {
                    for x in 0..4 {
                        if x < lx && y < ly && z < lz {
                            unsafe {
                                *data.offset(x as isize * sx + y as isize * sy + z as isize * sz) =
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
            let mut i = 0;
            for w in 0..4 {
                for z in 0..4 {
                    for y in 0..4 {
                        for x in 0..4 {
                            if x < lx && y < ly && z < lz && w < lw {
                                unsafe {
                                    *data.offset(
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
/// The strided encoders take this path for a reversible `config`.
///
/// # Safety
/// `data` must be valid for every offset the strides generate over the
/// block's extent. See the [`crate::codec::block`] module documentation.
pub unsafe fn encode_block_strided_reversible<T: ZfpScalar>(
    bs: &mut (impl ZfpBitStreamMutOps + ?Sized),
    data: *const T,
    dims: ZfpDimensionality,
    strides: &[isize],
    lengths: [usize; 4],
    config: &ZfpConfig,
) -> usize {
    unsafe {
        use crate::codec::encode::reversible as rev;

        // A 4-D block's worth of stack, rather than an allocation per block.
        let mut buf = [T::default(); 256];
        let block = &mut buf[..dims.block_size()];
        gather_block(data, dims, strides, lengths, block);
        reversible_dispatch! {
            encode bs, dims, block, config,
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
/// The strided decoders take this path for a reversible `config`.
///
/// # Safety
/// `data` must be valid for every offset the strides generate over the
/// block's extent. See the [`crate::codec::block`] module documentation.
pub unsafe fn decode_block_strided_reversible<T: ZfpScalar>(
    bs: &mut (impl ZfpBitStreamOps + ?Sized),
    data: *mut T,
    dims: ZfpDimensionality,
    strides: &[isize],
    lengths: [usize; 4],
    config: &ZfpConfig,
) -> usize {
    unsafe {
        use crate::codec::decode::reversible as rev;

        // A 4-D block's worth of stack, rather than an allocation per block.
        let mut buf = [T::default(); 256];
        let mut block = &mut buf[..dims.block_size()];
        let bits = reversible_dispatch! {
            decode bs, dims, block, config,
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

        scatter_block(block, data, dims, strides, lengths);
        bits
    }
}

/// Whether `config` selects the reversible coder, as C's `REVERSIBLE` does.
#[inline]
fn reversible(config: &ZfpConfig) -> bool {
    config.min_exp() < ZFP_MIN_EXP
}

/// The lengths of a whole 4^d block, for the reversible coder.
fn whole(dims: ZfpDimensionality) -> [usize; 4] {
    std::array::from_fn(|axis| if axis < usize::from(dims) { 4 } else { 0 })
}

/// A partial block's lengths on all four axes, for the reversible coder.
fn four(lengths: &[usize]) -> [usize; 4] {
    std::array::from_fn(|axis| lengths.get(axis).copied().unwrap_or(0))
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
/// block's extent. See the [`crate::codec::block`] module documentation.
pub unsafe fn encode_block_strided<T: ZfpScalar>(
    bs: &mut (impl ZfpBitStreamMutOps + ?Sized),
    data: *const T,
    dims: ZfpDimensionality,
    strides: &[isize],
    config: &ZfpConfig,
) -> usize {
    if reversible(config) {
        // SAFETY: the caller's contract.
        return unsafe {
            encode_block_strided_reversible(bs, data, dims, strides, whole(dims), config)
        };
    }
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
/// # Safety
/// `data` must be valid for every offset the strides generate over the
/// block's extent. See the [`crate::codec::block`] module documentation.
pub unsafe fn encode_partial_block_strided<T: ZfpScalar>(
    bs: &mut (impl ZfpBitStreamMutOps + ?Sized),
    data: *const T,
    dims: ZfpDimensionality,
    lengths: &[usize],
    strides: &[isize],
    config: &ZfpConfig,
) -> usize {
    if reversible(config) {
        // SAFETY: the caller's contract.
        return unsafe {
            encode_block_strided_reversible(bs, data, dims, strides, four(lengths), config)
        };
    }
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
/// block's extent. See the [`crate::codec::block`] module documentation.
pub unsafe fn decode_block_strided<T: ZfpScalar>(
    bs: &mut (impl ZfpBitStreamOps + ?Sized),
    data: *mut T,
    dims: ZfpDimensionality,
    strides: &[isize],
    config: &ZfpConfig,
) -> usize {
    if reversible(config) {
        // SAFETY: the caller's contract.
        return unsafe {
            decode_block_strided_reversible(bs, data, dims, strides, whole(dims), config)
        };
    }
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
/// # Safety
/// `data` must be valid for every offset the strides generate over the
/// block's extent. See the [`crate::codec::block`] module documentation.
pub unsafe fn decode_partial_block_strided<T: ZfpScalar>(
    bs: &mut (impl ZfpBitStreamOps + ?Sized),
    data: *mut T,
    dims: ZfpDimensionality,
    lengths: &[usize],
    strides: &[isize],
    config: &ZfpConfig,
) -> usize {
    if reversible(config) {
        // SAFETY: the caller's contract.
        return unsafe {
            decode_block_strided_reversible(bs, data, dims, strides, four(lengths), config)
        };
    }
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
