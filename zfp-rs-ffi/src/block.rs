//! Block-level encode/decode API: C-level wrappers for single-block operations.
//!
//! Implements all `zfp_encode_block_*` and `zfp_decode_block_*` functions
//! for contiguous (1-D) and strided/partial variants.
//!
//! Dispatches to `zfp_rs::codec::block::{encode_block_strided, ...}`
//! using the stream's compression parameters.

use crate::abi::zfp_stream;
use crate::bitstream_api::ZfpBitStreamHandleInner;
use zfp_rs::ZfpDimensionality;

/// Evaluate `$body` with `$bs` bound to the stream's bitstream and `$config`
/// to its parameters, or return 0 if either is missing.
///
/// Matching on the handle, rather than coding through `dyn`, dispatches the
/// block codec statically.
macro_rules! with_stream {
    ($stream:ident, |$bs:ident, $config:ident| $body:expr) => {{
        let Some((config, bitstream)) = (unsafe { crate::stream::stream_params($stream) }) else {
            return 0;
        };
        let Some(handle) = (unsafe { crate::bitstream_api::get_handle_mut(bitstream) }) else {
            return 0;
        };
        let $config = &config;
        match &mut handle.inner {
            ZfpBitStreamHandleInner::Owned($bs) => $body,
            ZfpBitStreamHandleInner::BorrowedMut($bs) => $body,
        }
    }};
}

macro_rules! impl_encode_block_1d {
    ($fn_name:ident, $ty:ty) => {
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $fn_name(stream: *mut zfp_stream, block: *const $ty) -> usize {
            if block.is_null() {
                return 0;
            }
            with_stream!(stream, |bs, config| {
                zfp_rs::codec::block::encode_block_strided::<$ty>(
                    bs,
                    block,
                    ZfpDimensionality::D1,
                    &[1, 0, 0, 0],
                    config,
                )
            })
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
            if block.is_null() {
                return 0;
            }
            with_stream!(stream, |bs, config| {
                zfp_rs::codec::block::encode_block_strided::<$ty>(
                    bs,
                    block,
                    ZfpDimensionality::D2,
                    &[1, 4, 0, 0],
                    config,
                )
            })
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
            if block.is_null() {
                return 0;
            }
            with_stream!(stream, |bs, config| {
                zfp_rs::codec::block::encode_block_strided::<$ty>(
                    bs,
                    block,
                    ZfpDimensionality::D3,
                    &[1, 4, 16, 0],
                    config,
                )
            })
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
            if block.is_null() {
                return 0;
            }
            with_stream!(stream, |bs, config| {
                zfp_rs::codec::block::encode_block_strided::<$ty>(
                    bs,
                    block,
                    ZfpDimensionality::D4,
                    &[1, 4, 16, 64],
                    config,
                )
            })
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
            if block.is_null() {
                return 0;
            }
            with_stream!(stream, |bs, config| {
                zfp_rs::codec::block::decode_block_strided::<$ty>(
                    bs,
                    block,
                    ZfpDimensionality::D1,
                    &[1, 0, 0, 0],
                    config,
                )
            })
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
            if block.is_null() {
                return 0;
            }
            with_stream!(stream, |bs, config| {
                zfp_rs::codec::block::decode_block_strided::<$ty>(
                    bs,
                    block,
                    ZfpDimensionality::D2,
                    &[1, 4, 0, 0],
                    config,
                )
            })
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
            if block.is_null() {
                return 0;
            }
            with_stream!(stream, |bs, config| {
                zfp_rs::codec::block::decode_block_strided::<$ty>(
                    bs,
                    block,
                    ZfpDimensionality::D3,
                    &[1, 4, 16, 0],
                    config,
                )
            })
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
            if block.is_null() {
                return 0;
            }
            with_stream!(stream, |bs, config| {
                zfp_rs::codec::block::decode_block_strided::<$ty>(
                    bs,
                    block,
                    ZfpDimensionality::D4,
                    &[1, 4, 16, 64],
                    config,
                )
            })
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
            if block.is_null() {
                return 0;
            }
            with_stream!(stream, |bs, config| {
                zfp_rs::codec::block::encode_block_strided::<$ty>(
                    bs,
                    block,
                    ZfpDimensionality::D1,
                    &[stride, 0, 0, 0],
                    config,
                )
            })
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
            if block.is_null() {
                return 0;
            }
            with_stream!(stream, |bs, config| {
                zfp_rs::codec::block::encode_block_strided::<$ty>(
                    bs,
                    block,
                    ZfpDimensionality::D2,
                    &[stride_x, stride_y, 0, 0],
                    config,
                )
            })
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
            if block.is_null() {
                return 0;
            }
            with_stream!(stream, |bs, config| {
                zfp_rs::codec::block::encode_block_strided::<$ty>(
                    bs,
                    block,
                    ZfpDimensionality::D3,
                    &[stride_x, stride_y, stride_z, 0],
                    config,
                )
            })
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
            if block.is_null() {
                return 0;
            }
            with_stream!(stream, |bs, config| {
                zfp_rs::codec::block::encode_block_strided::<$ty>(
                    bs,
                    block,
                    ZfpDimensionality::D4,
                    &[stride_x, stride_y, stride_z, stride_w],
                    config,
                )
            })
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
            if block.is_null() {
                return 0;
            }
            with_stream!(stream, |bs, config| {
                zfp_rs::codec::block::encode_partial_block_strided::<$ty>(
                    bs,
                    block,
                    ZfpDimensionality::D1,
                    [lx, 0, 0, 0],
                    &[stride, 0, 0, 0],
                    config,
                )
                .unwrap_or(0)
            })
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
            if block.is_null() {
                return 0;
            }
            with_stream!(stream, |bs, config| {
                zfp_rs::codec::block::encode_partial_block_strided::<$ty>(
                    bs,
                    block,
                    ZfpDimensionality::D2,
                    [lx, ly, 0, 0],
                    &[stride_x, stride_y, 0, 0],
                    config,
                )
                .unwrap_or(0)
            })
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
            if block.is_null() {
                return 0;
            }
            with_stream!(stream, |bs, config| {
                zfp_rs::codec::block::encode_partial_block_strided::<$ty>(
                    bs,
                    block,
                    ZfpDimensionality::D3,
                    [lx, ly, lz, 0],
                    &[stride_x, stride_y, stride_z, 0],
                    config,
                )
                .unwrap_or(0)
            })
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
            if block.is_null() {
                return 0;
            }
            with_stream!(stream, |bs, config| {
                zfp_rs::codec::block::encode_partial_block_strided::<$ty>(
                    bs,
                    block,
                    ZfpDimensionality::D4,
                    [lx, ly, lz, lw],
                    &[stride_x, stride_y, stride_z, stride_w],
                    config,
                )
                .unwrap_or(0)
            })
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
            if block.is_null() {
                return 0;
            }
            with_stream!(stream, |bs, config| {
                zfp_rs::codec::block::decode_block_strided::<$ty>(
                    bs,
                    block,
                    ZfpDimensionality::D1,
                    &[stride, 0, 0, 0],
                    config,
                )
            })
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
            if block.is_null() {
                return 0;
            }
            with_stream!(stream, |bs, config| {
                zfp_rs::codec::block::decode_block_strided::<$ty>(
                    bs,
                    block,
                    ZfpDimensionality::D2,
                    &[stride_x, stride_y, 0, 0],
                    config,
                )
            })
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
            if block.is_null() {
                return 0;
            }
            with_stream!(stream, |bs, config| {
                zfp_rs::codec::block::decode_block_strided::<$ty>(
                    bs,
                    block,
                    ZfpDimensionality::D3,
                    &[stride_x, stride_y, stride_z, 0],
                    config,
                )
            })
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
            if block.is_null() {
                return 0;
            }
            with_stream!(stream, |bs, config| {
                zfp_rs::codec::block::decode_block_strided::<$ty>(
                    bs,
                    block,
                    ZfpDimensionality::D4,
                    &[stride_x, stride_y, stride_z, stride_w],
                    config,
                )
            })
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
            if block.is_null() {
                return 0;
            }
            with_stream!(stream, |bs, config| {
                zfp_rs::codec::block::decode_partial_block_strided::<$ty>(
                    bs,
                    block,
                    ZfpDimensionality::D1,
                    [lx, 0, 0, 0],
                    &[stride, 0, 0, 0],
                    config,
                )
                .unwrap_or(0)
            })
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
            if block.is_null() {
                return 0;
            }
            with_stream!(stream, |bs, config| {
                zfp_rs::codec::block::decode_partial_block_strided::<$ty>(
                    bs,
                    block,
                    ZfpDimensionality::D2,
                    [lx, ly, 0, 0],
                    &[stride_x, stride_y, 0, 0],
                    config,
                )
                .unwrap_or(0)
            })
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
            if block.is_null() {
                return 0;
            }
            with_stream!(stream, |bs, config| {
                zfp_rs::codec::block::decode_partial_block_strided::<$ty>(
                    bs,
                    block,
                    ZfpDimensionality::D3,
                    [lx, ly, lz, 0],
                    &[stride_x, stride_y, stride_z, 0],
                    config,
                )
                .unwrap_or(0)
            })
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
            if block.is_null() {
                return 0;
            }
            with_stream!(stream, |bs, config| {
                zfp_rs::codec::block::decode_partial_block_strided::<$ty>(
                    bs,
                    block,
                    ZfpDimensionality::D4,
                    [lx, ly, lz, lw],
                    &[stride_x, stride_y, stride_z, stride_w],
                    config,
                )
                .unwrap_or(0)
            })
        }
    };
}

impl_decode_partial_block_strided_4d!(zfp_decode_partial_block_strided_int32_4, i32);
impl_decode_partial_block_strided_4d!(zfp_decode_partial_block_strided_int64_4, i64);
impl_decode_partial_block_strided_4d!(zfp_decode_partial_block_strided_float_4, f32);
impl_decode_partial_block_strided_4d!(zfp_decode_partial_block_strided_double_4, f64);
