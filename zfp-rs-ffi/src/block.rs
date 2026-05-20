//! Block-level encode/decode API: C-level wrappers for single-block operations.
//!
//! Implements all `zfp_encode_block_*` and `zfp_decode_block_*` functions
//! for contiguous (1-D) and strided/partial variants.
//!
//! Dispatches to `zfp_rs::codec::block::{encode_block_strided_with_params, ...}`
//! using the stream's compression parameters.

use crate::abi::zfp_stream;
use zfp_rs::{ZfpBitStreamMutOps, ZfpDimensionality};

/// Get the bitstream and params from a `zfp_stream`.
struct StreamContext<'a> {
    bs: &'a mut dyn ZfpBitStreamMutOps,
    min_bits: u32,
    max_bits: u32,
    max_prec: u32,
    min_exp: i32,
}

fn get_ctx(stream: *mut zfp_stream) -> Option<StreamContext<'static>> {
    if stream.is_null() {
        return None;
    }
    let (min_bits, max_bits, max_prec, min_exp, bitstream) =
        unsafe { crate::stream::stream_params(stream)? };
    let bs = unsafe {
        crate::bitstream_api::get_handle_mut(bitstream)?
            .inner
            .as_ops_mut()
    };
    Some(StreamContext {
        bs,
        min_bits,
        max_bits,
        max_prec,
        min_exp,
    })
}

fn strided_footprint(lengths: &[usize], strides: &[isize]) -> Option<usize> {
    lengths
        .iter()
        .zip(strides)
        .try_fold(1usize, |len, (&length, &stride)| {
            if length == 0 || stride < 0 {
                return None;
            }
            len.checked_add((length - 1).checked_mul(stride.cast_unsigned())?)
        })
}

macro_rules! impl_encode_block_1d {
    ($fn_name:ident, $ty:ty) => {
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $fn_name(stream: *mut zfp_stream, block: *const $ty) -> usize {
            let Some(ctx) = get_ctx(stream) else { return 0 };
            if block.is_null() {
                return 0;
            }
            // SAFETY: caller guarantees block points to 4 elements
            let slice = std::slice::from_raw_parts(block, 4);
            zfp_rs::codec::block::encode_block_strided_with_params::<$ty>(
                ctx.bs,
                slice,
                ZfpDimensionality::D1,
                &[1],
                ctx.min_bits,
                ctx.max_bits,
                ctx.max_prec,
                ctx.min_exp,
            ) as usize
        }
    };
}

impl_encode_block_1d!(zfp_encode_block_int32_1, i32);
impl_encode_block_1d!(zfp_encode_block_int64_1, i64);
impl_encode_block_1d!(zfp_encode_block_float_1, f32);
impl_encode_block_1d!(zfp_encode_block_double_1, f64);

macro_rules! impl_encode_block_2d {
    ($fn_name:ident, $ty:ty) => {
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $fn_name(stream: *mut zfp_stream, block: *const $ty) -> usize {
            let Some(ctx) = get_ctx(stream) else { return 0 };
            if block.is_null() {
                return 0;
            }
            // SAFETY: caller guarantees block points to 16 elements
            let slice = std::slice::from_raw_parts(block, 16);
            zfp_rs::codec::block::encode_block_strided_with_params::<$ty>(
                ctx.bs,
                slice,
                ZfpDimensionality::D2,
                &[1, 4],
                ctx.min_bits,
                ctx.max_bits,
                ctx.max_prec,
                ctx.min_exp,
            ) as usize
        }
    };
}

impl_encode_block_2d!(zfp_encode_block_int32_2, i32);
impl_encode_block_2d!(zfp_encode_block_int64_2, i64);
impl_encode_block_2d!(zfp_encode_block_float_2, f32);
impl_encode_block_2d!(zfp_encode_block_double_2, f64);

macro_rules! impl_encode_block_3d {
    ($fn_name:ident, $ty:ty) => {
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $fn_name(stream: *mut zfp_stream, block: *const $ty) -> usize {
            let Some(ctx) = get_ctx(stream) else { return 0 };
            if block.is_null() {
                return 0;
            }
            // SAFETY: caller guarantees block points to 64 elements
            let slice = std::slice::from_raw_parts(block, 64);
            zfp_rs::codec::block::encode_block_strided_with_params::<$ty>(
                ctx.bs,
                slice,
                ZfpDimensionality::D3,
                &[1, 4, 16],
                ctx.min_bits,
                ctx.max_bits,
                ctx.max_prec,
                ctx.min_exp,
            ) as usize
        }
    };
}

impl_encode_block_3d!(zfp_encode_block_int32_3, i32);
impl_encode_block_3d!(zfp_encode_block_int64_3, i64);
impl_encode_block_3d!(zfp_encode_block_float_3, f32);
impl_encode_block_3d!(zfp_encode_block_double_3, f64);

macro_rules! impl_encode_block_4d {
    ($fn_name:ident, $ty:ty) => {
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $fn_name(stream: *mut zfp_stream, block: *const $ty) -> usize {
            let Some(ctx) = get_ctx(stream) else { return 0 };
            if block.is_null() {
                return 0;
            }
            // SAFETY: caller guarantees block points to 256 elements
            let slice = std::slice::from_raw_parts(block, 256);
            zfp_rs::codec::block::encode_block_strided_with_params::<$ty>(
                ctx.bs,
                slice,
                ZfpDimensionality::D4,
                &[1, 4, 16, 64],
                ctx.min_bits,
                ctx.max_bits,
                ctx.max_prec,
                ctx.min_exp,
            ) as usize
        }
    };
}

impl_encode_block_4d!(zfp_encode_block_int32_4, i32);
impl_encode_block_4d!(zfp_encode_block_int64_4, i64);
impl_encode_block_4d!(zfp_encode_block_float_4, f32);
impl_encode_block_4d!(zfp_encode_block_double_4, f64);

// ===========================================================================
// CONTIGUOUS BLOCK DECODERS
// ===========================================================================

macro_rules! impl_decode_block_1d {
    ($fn_name:ident, $ty:ty) => {
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $fn_name(stream: *mut zfp_stream, block: *mut $ty) -> usize {
            let Some(ctx) = get_ctx(stream) else { return 0 };
            if block.is_null() {
                return 0;
            }
            // SAFETY: caller guarantees block points to 4 elements
            let slice = std::slice::from_raw_parts_mut(block, 4);
            zfp_rs::codec::block::decode_block_strided_with_params::<$ty>(
                ctx.bs,
                slice,
                ZfpDimensionality::D1,
                &[1],
                ctx.min_bits,
                ctx.max_bits,
                ctx.max_prec,
                ctx.min_exp,
            ) as usize
        }
    };
}

impl_decode_block_1d!(zfp_decode_block_int32_1, i32);
impl_decode_block_1d!(zfp_decode_block_int64_1, i64);
impl_decode_block_1d!(zfp_decode_block_float_1, f32);
impl_decode_block_1d!(zfp_decode_block_double_1, f64);

macro_rules! impl_decode_block_2d {
    ($fn_name:ident, $ty:ty) => {
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $fn_name(stream: *mut zfp_stream, block: *mut $ty) -> usize {
            let Some(ctx) = get_ctx(stream) else { return 0 };
            if block.is_null() {
                return 0;
            }
            // SAFETY: caller guarantees block points to 16 elements
            let slice = std::slice::from_raw_parts_mut(block, 16);
            zfp_rs::codec::block::decode_block_strided_with_params::<$ty>(
                ctx.bs,
                slice,
                ZfpDimensionality::D2,
                &[1, 4],
                ctx.min_bits,
                ctx.max_bits,
                ctx.max_prec,
                ctx.min_exp,
            ) as usize
        }
    };
}

impl_decode_block_2d!(zfp_decode_block_int32_2, i32);
impl_decode_block_2d!(zfp_decode_block_int64_2, i64);
impl_decode_block_2d!(zfp_decode_block_float_2, f32);
impl_decode_block_2d!(zfp_decode_block_double_2, f64);

macro_rules! impl_decode_block_3d {
    ($fn_name:ident, $ty:ty) => {
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $fn_name(stream: *mut zfp_stream, block: *mut $ty) -> usize {
            let Some(ctx) = get_ctx(stream) else { return 0 };
            if block.is_null() {
                return 0;
            }
            // SAFETY: caller guarantees block points to 64 elements
            let slice = std::slice::from_raw_parts_mut(block, 64);
            zfp_rs::codec::block::decode_block_strided_with_params::<$ty>(
                ctx.bs,
                slice,
                ZfpDimensionality::D3,
                &[1, 4, 16],
                ctx.min_bits,
                ctx.max_bits,
                ctx.max_prec,
                ctx.min_exp,
            ) as usize
        }
    };
}

impl_decode_block_3d!(zfp_decode_block_int32_3, i32);
impl_decode_block_3d!(zfp_decode_block_int64_3, i64);
impl_decode_block_3d!(zfp_decode_block_float_3, f32);
impl_decode_block_3d!(zfp_decode_block_double_3, f64);

macro_rules! impl_decode_block_4d {
    ($fn_name:ident, $ty:ty) => {
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $fn_name(stream: *mut zfp_stream, block: *mut $ty) -> usize {
            let Some(ctx) = get_ctx(stream) else { return 0 };
            if block.is_null() {
                return 0;
            }
            // SAFETY: caller guarantees block points to 256 elements
            let slice = std::slice::from_raw_parts_mut(block, 256);
            zfp_rs::codec::block::decode_block_strided_with_params::<$ty>(
                ctx.bs,
                slice,
                ZfpDimensionality::D4,
                &[1, 4, 16, 64],
                ctx.min_bits,
                ctx.max_bits,
                ctx.max_prec,
                ctx.min_exp,
            ) as usize
        }
    };
}

impl_decode_block_4d!(zfp_decode_block_int32_4, i32);
impl_decode_block_4d!(zfp_decode_block_int64_4, i64);
impl_decode_block_4d!(zfp_decode_block_float_4, f32);
impl_decode_block_4d!(zfp_decode_block_double_4, f64);

// ===========================================================================
// STRIDED BLOCK ENCODERS
// ===========================================================================

macro_rules! impl_encode_block_strided_1d {
    ($fn_name:ident, $ty:ty) => {
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $fn_name(
            stream: *mut zfp_stream,
            block: *const $ty,
            stride: isize,
        ) -> usize {
            let Some(ctx) = get_ctx(stream) else { return 0 };
            if block.is_null() {
                return 0;
            }
            let slice = std::slice::from_raw_parts(block, 4);
            zfp_rs::codec::block::encode_block_strided_with_params::<$ty>(
                ctx.bs,
                slice,
                ZfpDimensionality::D1,
                &[stride],
                ctx.min_bits,
                ctx.max_bits,
                ctx.max_prec,
                ctx.min_exp,
            ) as usize
        }
    };
}

impl_encode_block_strided_1d!(zfp_encode_block_strided_int32_1, i32);
impl_encode_block_strided_1d!(zfp_encode_block_strided_int64_1, i64);
impl_encode_block_strided_1d!(zfp_encode_block_strided_float_1, f32);
impl_encode_block_strided_1d!(zfp_encode_block_strided_double_1, f64);

macro_rules! impl_encode_block_strided_2d {
    ($fn_name:ident, $ty:ty) => {
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $fn_name(
            stream: *mut zfp_stream,
            block: *const $ty,
            stride_x: isize,
            stride_y: isize,
        ) -> usize {
            let Some(ctx) = get_ctx(stream) else { return 0 };
            if block.is_null() {
                return 0;
            }
            let slice = std::slice::from_raw_parts(block, 16);
            zfp_rs::codec::block::encode_block_strided_with_params::<$ty>(
                ctx.bs,
                slice,
                ZfpDimensionality::D2,
                &[stride_x, stride_y],
                ctx.min_bits,
                ctx.max_bits,
                ctx.max_prec,
                ctx.min_exp,
            ) as usize
        }
    };
}

impl_encode_block_strided_2d!(zfp_encode_block_strided_int32_2, i32);
impl_encode_block_strided_2d!(zfp_encode_block_strided_int64_2, i64);
impl_encode_block_strided_2d!(zfp_encode_block_strided_float_2, f32);
impl_encode_block_strided_2d!(zfp_encode_block_strided_double_2, f64);

macro_rules! impl_encode_block_strided_3d {
    ($fn_name:ident, $ty:ty) => {
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $fn_name(
            stream: *mut zfp_stream,
            block: *const $ty,
            stride_x: isize,
            stride_y: isize,
            stride_z: isize,
        ) -> usize {
            let Some(ctx) = get_ctx(stream) else { return 0 };
            if block.is_null() {
                return 0;
            }
            let slice = std::slice::from_raw_parts(block, 64);
            zfp_rs::codec::block::encode_block_strided_with_params::<$ty>(
                ctx.bs,
                slice,
                ZfpDimensionality::D3,
                &[stride_x, stride_y, stride_z],
                ctx.min_bits,
                ctx.max_bits,
                ctx.max_prec,
                ctx.min_exp,
            ) as usize
        }
    };
}

impl_encode_block_strided_3d!(zfp_encode_block_strided_int32_3, i32);
impl_encode_block_strided_3d!(zfp_encode_block_strided_int64_3, i64);
impl_encode_block_strided_3d!(zfp_encode_block_strided_float_3, f32);
impl_encode_block_strided_3d!(zfp_encode_block_strided_double_3, f64);

macro_rules! impl_encode_block_strided_4d {
    ($fn_name:ident, $ty:ty) => {
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $fn_name(
            stream: *mut zfp_stream,
            block: *const $ty,
            stride_x: isize,
            stride_y: isize,
            stride_z: isize,
            stride_w: isize,
        ) -> usize {
            let Some(ctx) = get_ctx(stream) else { return 0 };
            if block.is_null() {
                return 0;
            }
            let slice = std::slice::from_raw_parts(block, 256);
            zfp_rs::codec::block::encode_block_strided_with_params::<$ty>(
                ctx.bs,
                slice,
                ZfpDimensionality::D4,
                &[stride_x, stride_y, stride_z, stride_w],
                ctx.min_bits,
                ctx.max_bits,
                ctx.max_prec,
                ctx.min_exp,
            ) as usize
        }
    };
}

impl_encode_block_strided_4d!(zfp_encode_block_strided_int32_4, i32);
impl_encode_block_strided_4d!(zfp_encode_block_strided_int64_4, i64);
impl_encode_block_strided_4d!(zfp_encode_block_strided_float_4, f32);
impl_encode_block_strided_4d!(zfp_encode_block_strided_double_4, f64);

// ===========================================================================
// PARTIAL BLOCK ENCODERS (1-D)
// ===========================================================================

macro_rules! impl_encode_partial_block_strided_1d {
    ($fn_name:ident, $ty:ty) => {
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $fn_name(
            stream: *mut zfp_stream,
            block: *const $ty,
            lx: usize,
            stride: isize,
        ) -> usize {
            let Some(ctx) = get_ctx(stream) else { return 0 };
            if block.is_null() || lx == 0 || lx > 4 {
                return 0;
            }
            let slice = std::slice::from_raw_parts(block, lx);
            zfp_rs::codec::block::encode_partial_block_strided_with_params::<$ty>(
                ctx.bs,
                slice,
                ZfpDimensionality::D1,
                &[lx],
                &[stride],
                ctx.min_bits,
                ctx.max_bits,
                ctx.max_prec,
                ctx.min_exp,
            ) as usize
        }
    };
}

impl_encode_partial_block_strided_1d!(zfp_encode_partial_block_strided_int32_1, i32);
impl_encode_partial_block_strided_1d!(zfp_encode_partial_block_strided_int64_1, i64);
impl_encode_partial_block_strided_1d!(zfp_encode_partial_block_strided_float_1, f32);
impl_encode_partial_block_strided_1d!(zfp_encode_partial_block_strided_double_1, f64);

macro_rules! impl_encode_partial_block_strided_2d {
    ($fn_name:ident, $ty:ty) => {
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $fn_name(
            stream: *mut zfp_stream,
            block: *const $ty,
            lx: usize,
            ly: usize,
            stride_x: isize,
            stride_y: isize,
        ) -> usize {
            let Some(ctx) = get_ctx(stream) else { return 0 };
            if block.is_null() || lx == 0 || ly == 0 || lx > 4 || ly > 4 {
                return 0;
            }
            let slice = std::slice::from_raw_parts(block, lx * ly);
            zfp_rs::codec::block::encode_partial_block_strided_with_params::<$ty>(
                ctx.bs,
                slice,
                ZfpDimensionality::D2,
                &[lx, ly],
                &[stride_x, stride_y],
                ctx.min_bits,
                ctx.max_bits,
                ctx.max_prec,
                ctx.min_exp,
            ) as usize
        }
    };
}

impl_encode_partial_block_strided_2d!(zfp_encode_partial_block_strided_int32_2, i32);
impl_encode_partial_block_strided_2d!(zfp_encode_partial_block_strided_int64_2, i64);
impl_encode_partial_block_strided_2d!(zfp_encode_partial_block_strided_float_2, f32);
impl_encode_partial_block_strided_2d!(zfp_encode_partial_block_strided_double_2, f64);

macro_rules! impl_encode_partial_block_strided_3d {
    ($fn_name:ident, $ty:ty) => {
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $fn_name(
            stream: *mut zfp_stream,
            block: *const $ty,
            lx: usize,
            ly: usize,
            lz: usize,
            stride_x: isize,
            stride_y: isize,
            stride_z: isize,
        ) -> usize {
            let Some(ctx) = get_ctx(stream) else { return 0 };
            if block.is_null() || lx == 0 || ly == 0 || lz == 0 || lx > 4 || ly > 4 || lz > 4 {
                return 0;
            }
            let slice = std::slice::from_raw_parts(block, lx * ly * lz);
            zfp_rs::codec::block::encode_partial_block_strided_with_params::<$ty>(
                ctx.bs,
                slice,
                ZfpDimensionality::D3,
                &[lx, ly, lz],
                &[stride_x, stride_y, stride_z],
                ctx.min_bits,
                ctx.max_bits,
                ctx.max_prec,
                ctx.min_exp,
            ) as usize
        }
    };
}

impl_encode_partial_block_strided_3d!(zfp_encode_partial_block_strided_int32_3, i32);
impl_encode_partial_block_strided_3d!(zfp_encode_partial_block_strided_int64_3, i64);
impl_encode_partial_block_strided_3d!(zfp_encode_partial_block_strided_float_3, f32);
impl_encode_partial_block_strided_3d!(zfp_encode_partial_block_strided_double_3, f64);

macro_rules! impl_encode_partial_block_strided_4d {
    ($fn_name:ident, $ty:ty) => {
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $fn_name(
            stream: *mut zfp_stream,
            block: *const $ty,
            lx: usize,
            ly: usize,
            lz: usize,
            lw: usize,
            stride_x: isize,
            stride_y: isize,
            stride_z: isize,
            stride_w: isize,
        ) -> usize {
            let Some(ctx) = get_ctx(stream) else { return 0 };
            if block.is_null()
                || lx == 0
                || ly == 0
                || lz == 0
                || lw == 0
                || lx > 4
                || ly > 4
                || lz > 4
                || lw > 4
            {
                return 0;
            }
            let slice = std::slice::from_raw_parts(block, lx * ly * lz * lw);
            zfp_rs::codec::block::encode_partial_block_strided_with_params::<$ty>(
                ctx.bs,
                slice,
                ZfpDimensionality::D4,
                &[lx, ly, lz, lw],
                &[stride_x, stride_y, stride_z, stride_w],
                ctx.min_bits,
                ctx.max_bits,
                ctx.max_prec,
                ctx.min_exp,
            ) as usize
        }
    };
}

impl_encode_partial_block_strided_4d!(zfp_encode_partial_block_strided_int32_4, i32);
impl_encode_partial_block_strided_4d!(zfp_encode_partial_block_strided_int64_4, i64);
impl_encode_partial_block_strided_4d!(zfp_encode_partial_block_strided_float_4, f32);
impl_encode_partial_block_strided_4d!(zfp_encode_partial_block_strided_double_4, f64);

// ===========================================================================
// STRIDED BLOCK DECODERS
// ===========================================================================

macro_rules! impl_decode_block_strided_1d {
    ($fn_name:ident, $ty:ty) => {
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $fn_name(
            stream: *mut zfp_stream,
            block: *mut $ty,
            stride: isize,
        ) -> usize {
            let Some(ctx) = get_ctx(stream) else { return 0 };
            if block.is_null() {
                return 0;
            }
            let slice = std::slice::from_raw_parts_mut(block, 4);
            zfp_rs::codec::block::decode_block_strided_with_params::<$ty>(
                ctx.bs,
                slice,
                ZfpDimensionality::D1,
                &[stride],
                ctx.min_bits,
                ctx.max_bits,
                ctx.max_prec,
                ctx.min_exp,
            ) as usize
        }
    };
}

impl_decode_block_strided_1d!(zfp_decode_block_strided_int32_1, i32);
impl_decode_block_strided_1d!(zfp_decode_block_strided_int64_1, i64);
impl_decode_block_strided_1d!(zfp_decode_block_strided_float_1, f32);
impl_decode_block_strided_1d!(zfp_decode_block_strided_double_1, f64);

macro_rules! impl_decode_block_strided_2d {
    ($fn_name:ident, $ty:ty) => {
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $fn_name(
            stream: *mut zfp_stream,
            block: *mut $ty,
            stride_x: isize,
            stride_y: isize,
        ) -> usize {
            let Some(ctx) = get_ctx(stream) else { return 0 };
            if block.is_null() {
                return 0;
            }
            let slice = std::slice::from_raw_parts_mut(block, 16);
            zfp_rs::codec::block::decode_block_strided_with_params::<$ty>(
                ctx.bs,
                slice,
                ZfpDimensionality::D2,
                &[stride_x, stride_y],
                ctx.min_bits,
                ctx.max_bits,
                ctx.max_prec,
                ctx.min_exp,
            ) as usize
        }
    };
}

impl_decode_block_strided_2d!(zfp_decode_block_strided_int32_2, i32);
impl_decode_block_strided_2d!(zfp_decode_block_strided_int64_2, i64);
impl_decode_block_strided_2d!(zfp_decode_block_strided_float_2, f32);
impl_decode_block_strided_2d!(zfp_decode_block_strided_double_2, f64);

macro_rules! impl_decode_block_strided_3d {
    ($fn_name:ident, $ty:ty) => {
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $fn_name(
            stream: *mut zfp_stream,
            block: *mut $ty,
            stride_x: isize,
            stride_y: isize,
            stride_z: isize,
        ) -> usize {
            let Some(ctx) = get_ctx(stream) else { return 0 };
            if block.is_null() {
                return 0;
            }
            let slice = std::slice::from_raw_parts_mut(block, 64);
            zfp_rs::codec::block::decode_block_strided_with_params::<$ty>(
                ctx.bs,
                slice,
                ZfpDimensionality::D3,
                &[stride_x, stride_y, stride_z],
                ctx.min_bits,
                ctx.max_bits,
                ctx.max_prec,
                ctx.min_exp,
            ) as usize
        }
    };
}

impl_decode_block_strided_3d!(zfp_decode_block_strided_int32_3, i32);
impl_decode_block_strided_3d!(zfp_decode_block_strided_int64_3, i64);
impl_decode_block_strided_3d!(zfp_decode_block_strided_float_3, f32);
impl_decode_block_strided_3d!(zfp_decode_block_strided_double_3, f64);

macro_rules! impl_decode_block_strided_4d {
    ($fn_name:ident, $ty:ty) => {
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $fn_name(
            stream: *mut zfp_stream,
            block: *mut $ty,
            stride_x: isize,
            stride_y: isize,
            stride_z: isize,
            stride_w: isize,
        ) -> usize {
            let Some(ctx) = get_ctx(stream) else { return 0 };
            if block.is_null() {
                return 0;
            }
            let slice = std::slice::from_raw_parts_mut(block, 256);
            zfp_rs::codec::block::decode_block_strided_with_params::<$ty>(
                ctx.bs,
                slice,
                ZfpDimensionality::D4,
                &[stride_x, stride_y, stride_z, stride_w],
                ctx.min_bits,
                ctx.max_bits,
                ctx.max_prec,
                ctx.min_exp,
            ) as usize
        }
    };
}

impl_decode_block_strided_4d!(zfp_decode_block_strided_int32_4, i32);
impl_decode_block_strided_4d!(zfp_decode_block_strided_int64_4, i64);
impl_decode_block_strided_4d!(zfp_decode_block_strided_float_4, f32);
impl_decode_block_strided_4d!(zfp_decode_block_strided_double_4, f64);

// ===========================================================================
// PARTIAL BLOCK DECODERS
// ===========================================================================

macro_rules! impl_decode_partial_block_strided_1d {
    ($fn_name:ident, $ty:ty) => {
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $fn_name(
            stream: *mut zfp_stream,
            block: *mut $ty,
            lx: usize,
            stride: isize,
        ) -> usize {
            let Some(ctx) = get_ctx(stream) else { return 0 };
            if block.is_null() || lx == 0 || lx > 4 {
                return 0;
            }
            let Some(len) = strided_footprint(&[lx], &[stride]) else {
                return 0;
            };
            let slice = std::slice::from_raw_parts_mut(block, len);
            zfp_rs::codec::block::decode_partial_block_strided_with_params::<$ty>(
                ctx.bs,
                slice,
                ZfpDimensionality::D1,
                &[lx],
                &[stride],
                ctx.min_bits,
                ctx.max_bits,
                ctx.max_prec,
                ctx.min_exp,
            ) as usize
        }
    };
}

impl_decode_partial_block_strided_1d!(zfp_decode_partial_block_strided_int32_1, i32);
impl_decode_partial_block_strided_1d!(zfp_decode_partial_block_strided_int64_1, i64);
impl_decode_partial_block_strided_1d!(zfp_decode_partial_block_strided_float_1, f32);
impl_decode_partial_block_strided_1d!(zfp_decode_partial_block_strided_double_1, f64);

macro_rules! impl_decode_partial_block_strided_2d {
    ($fn_name:ident, $ty:ty) => {
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $fn_name(
            stream: *mut zfp_stream,
            block: *mut $ty,
            lx: usize,
            ly: usize,
            stride_x: isize,
            stride_y: isize,
        ) -> usize {
            let Some(ctx) = get_ctx(stream) else { return 0 };
            if block.is_null() || lx == 0 || ly == 0 || lx > 4 || ly > 4 {
                return 0;
            }
            let Some(len) = strided_footprint(&[lx, ly], &[stride_x, stride_y]) else {
                return 0;
            };
            let slice = std::slice::from_raw_parts_mut(block, len);
            zfp_rs::codec::block::decode_partial_block_strided_with_params::<$ty>(
                ctx.bs,
                slice,
                ZfpDimensionality::D2,
                &[lx, ly],
                &[stride_x, stride_y],
                ctx.min_bits,
                ctx.max_bits,
                ctx.max_prec,
                ctx.min_exp,
            ) as usize
        }
    };
}

impl_decode_partial_block_strided_2d!(zfp_decode_partial_block_strided_int32_2, i32);
impl_decode_partial_block_strided_2d!(zfp_decode_partial_block_strided_int64_2, i64);
impl_decode_partial_block_strided_2d!(zfp_decode_partial_block_strided_float_2, f32);
impl_decode_partial_block_strided_2d!(zfp_decode_partial_block_strided_double_2, f64);

macro_rules! impl_decode_partial_block_strided_3d {
    ($fn_name:ident, $ty:ty) => {
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $fn_name(
            stream: *mut zfp_stream,
            block: *mut $ty,
            lx: usize,
            ly: usize,
            lz: usize,
            stride_x: isize,
            stride_y: isize,
            stride_z: isize,
        ) -> usize {
            let Some(ctx) = get_ctx(stream) else { return 0 };
            if block.is_null() || lx == 0 || ly == 0 || lz == 0 || lx > 4 || ly > 4 || lz > 4 {
                return 0;
            }
            let Some(len) = strided_footprint(&[lx, ly, lz], &[stride_x, stride_y, stride_z])
            else {
                return 0;
            };
            let slice = std::slice::from_raw_parts_mut(block, len);
            zfp_rs::codec::block::decode_partial_block_strided_with_params::<$ty>(
                ctx.bs,
                slice,
                ZfpDimensionality::D3,
                &[lx, ly, lz],
                &[stride_x, stride_y, stride_z],
                ctx.min_bits,
                ctx.max_bits,
                ctx.max_prec,
                ctx.min_exp,
            ) as usize
        }
    };
}

impl_decode_partial_block_strided_3d!(zfp_decode_partial_block_strided_int32_3, i32);
impl_decode_partial_block_strided_3d!(zfp_decode_partial_block_strided_int64_3, i64);
impl_decode_partial_block_strided_3d!(zfp_decode_partial_block_strided_float_3, f32);
impl_decode_partial_block_strided_3d!(zfp_decode_partial_block_strided_double_3, f64);

macro_rules! impl_decode_partial_block_strided_4d {
    ($fn_name:ident, $ty:ty) => {
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $fn_name(
            stream: *mut zfp_stream,
            block: *mut $ty,
            lx: usize,
            ly: usize,
            lz: usize,
            lw: usize,
            stride_x: isize,
            stride_y: isize,
            stride_z: isize,
            stride_w: isize,
        ) -> usize {
            let Some(ctx) = get_ctx(stream) else { return 0 };
            if block.is_null()
                || lx == 0
                || ly == 0
                || lz == 0
                || lw == 0
                || lx > 4
                || ly > 4
                || lz > 4
                || lw > 4
            {
                return 0;
            }
            let Some(len) =
                strided_footprint(&[lx, ly, lz, lw], &[stride_x, stride_y, stride_z, stride_w])
            else {
                return 0;
            };
            let slice = std::slice::from_raw_parts_mut(block, len);
            zfp_rs::codec::block::decode_partial_block_strided_with_params::<$ty>(
                ctx.bs,
                slice,
                ZfpDimensionality::D4,
                &[lx, ly, lz, lw],
                &[stride_x, stride_y, stride_z, stride_w],
                ctx.min_bits,
                ctx.max_bits,
                ctx.max_prec,
                ctx.min_exp,
            ) as usize
        }
    };
}

impl_decode_partial_block_strided_4d!(zfp_decode_partial_block_strided_int32_4, i32);
impl_decode_partial_block_strided_4d!(zfp_decode_partial_block_strided_int64_4, i64);
impl_decode_partial_block_strided_4d!(zfp_decode_partial_block_strided_float_4, f32);
impl_decode_partial_block_strided_4d!(zfp_decode_partial_block_strided_double_4, f64);
