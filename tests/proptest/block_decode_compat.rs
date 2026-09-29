//! Property-based compatibility tests: block_decode_compat.
//!
//! Uses `proptest` + `zfp-sys` to verify byte-for-byte output compatibility
//! with the reference C library for block-level decoding.
//!
//! For each scalar type (i32, i64, f32, f64) and dimensionality (1–4):
//! 1. Encode a random block with C `zfp_encode_block_*` (reference bitstream).
//! 2. Decode the bitstream with both the Rust typed decode function and the C
//!    `zfp_decode_block_*` function.
//! 3. Assert the decoded arrays are identical.
//!
//! Floats include blocks too small for C's encoder, whose scale factor
//! overflows, so zfp-rs must decode C's broken blocks as C does.

#![cfg(feature = "ffi")]

use proptest::prelude::*;
use zfp_rs::ZfpBitStream;
use zfp_rs::ZfpConfig;
use zfp_rs::ZfpDimensionality;
use zfp_rs::codec::block::decode_block_strided;
use zfp_rs::codec::decode::{float as dfloat, integer as dinteger};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const ZFP_MAX_PREC: u32 = 64;
const ZFP_MIN_EXP: i32 = -1074;
const ZFP_MAX_BITS: u32 = 16658;
const ZFP_RATE_PARAM_BITS: u32 = 19;

// ---------------------------------------------------------------------------
// C helper: zfp_stream + bitstream
// ---------------------------------------------------------------------------

struct CZfpBlock {
    zfp: *mut zfp_sys::zfp_stream,
    bs_ptr: *mut zfp_sys::bitstream,
    buf: Vec<u8>,
}

impl CZfpBlock {
    const CAPACITY: usize = 65536;

    fn new(maxbits: u32) -> Self {
        let mut buf = vec![0u8; Self::CAPACITY];
        let bs_ptr = unsafe {
            zfp_sys::stream_open(buf.as_mut_ptr().cast::<std::ffi::c_void>(), Self::CAPACITY)
        };
        assert!(!bs_ptr.is_null());
        let zfp = unsafe { zfp_sys::zfp_stream_open(bs_ptr) };
        assert!(!zfp.is_null());
        unsafe {
            zfp_sys::zfp_stream_set_params(zfp, maxbits, maxbits, ZFP_MAX_PREC, ZFP_MIN_EXP);
        }
        Self { zfp, bs_ptr, buf }
    }

    fn flush(&mut self) {
        unsafe { zfp_sys::stream_flush(self.bs_ptr) };
    }

    fn rewind(&mut self) {
        unsafe { zfp_sys::zfp_stream_rewind(self.zfp) };
    }

    fn encoded_bytes(&self) -> &[u8] {
        let size = unsafe { zfp_sys::stream_size(self.bs_ptr) };
        &self.buf[..size]
    }
}

impl Drop for CZfpBlock {
    fn drop(&mut self) {
        unsafe {
            zfp_sys::zfp_stream_close(self.zfp);
            zfp_sys::stream_close(self.bs_ptr);
        }
    }
}

// ---------------------------------------------------------------------------
// Macro: integer block decode compat
// ---------------------------------------------------------------------------

macro_rules! block_decode_compat_int {
    (
        $test_name:ident,
        $scalar:ty,
        $block_size:expr,
        $c_scalar:ty,
        $rs_decode:path,
        $c_encode:path,
        $c_decode:path
    ) => {
        proptest! {
            #[test]
            fn $test_name(data in prop::collection::vec(
                any::<$scalar>(),
                $block_size..$block_size + 1,
            )) {
                let maxbits = ($block_size as u32) * ZFP_RATE_PARAM_BITS;

                // Encode with C (reference bitstream)
                let mut c = CZfpBlock::new(maxbits);
                unsafe { $c_encode(c.zfp, data.as_ptr().cast::<$c_scalar>()) };
                c.flush();

                // Copy encoded bytes for Rust, then rewind C for C decode
                let c_bytes = c.encoded_bytes().to_vec();
                c.rewind();

                // Decode with Rust.
                let mut rs_bs = ZfpBitStream::from_bytes(&c_bytes);
                let rs_out = $rs_decode(&mut rs_bs, &ZfpConfig::expert(maxbits, maxbits, ZFP_MAX_PREC, ZFP_MIN_EXP).unwrap());

                // Decode with C
                let mut c_out = vec![0 as $scalar; $block_size];
                unsafe { $c_decode(c.zfp, c_out.as_mut_ptr().cast::<$c_scalar>()) };

                prop_assert_eq!(
                    rs_out.as_ref(),
                    c_out.as_slice(),
                    "decoded array mismatch for {}",
                    stringify!($test_name)
                );
            }
        }
    };
}

// ---------------------------------------------------------------------------
// Macro: float block decode compat
// ---------------------------------------------------------------------------

macro_rules! block_decode_compat_float {
    (
        $test_name:ident,
        $scalar:ty,
        $block_size:expr,
        $normal_strategy:expr,
        $rs_decode:path,
        $c_encode:path,
        $c_decode:path
    ) => {
        proptest! {
            #[test]
            fn $test_name(data in prop::collection::vec(
                $normal_strategy,
                $block_size..$block_size + 1,
            )) {
                let maxbits = ($block_size as u32) * ZFP_RATE_PARAM_BITS;

                // Encode with C (reference bitstream)
                let mut c = CZfpBlock::new(maxbits);
                unsafe { $c_encode(c.zfp, data.as_ptr().cast::<$scalar>()) };
                c.flush();

                // Copy encoded bytes for Rust, then rewind C for C decode
                let c_bytes = c.encoded_bytes().to_vec();
                c.rewind();

                // Decode with Rust.
                let mut rs_bs = ZfpBitStream::from_bytes(&c_bytes);
                let rs_out = $rs_decode(&mut rs_bs, &ZfpConfig::expert(maxbits, maxbits, ZFP_MAX_PREC, ZFP_MIN_EXP).unwrap());

                // Decode with C
                let mut c_out = vec![0.0 as $scalar; $block_size];
                unsafe { $c_decode(c.zfp, c_out.as_mut_ptr().cast::<$scalar>()) };

                // Compare bit-for-bit (handles NaN)
                let rs_bits: Vec<_> = rs_out.iter().map(|x| x.to_bits()).collect();
                let c_bits: Vec<_> = c_out.iter().map(|x| x.to_bits()).collect();
                prop_assert_eq!(
                    rs_bits,
                    c_bits,
                    "decoded array mismatch for {}",
                    stringify!($test_name)
                );
            }
        }
    };
}

// ---------------------------------------------------------------------------
// 1-D
// ---------------------------------------------------------------------------

block_decode_compat_int!(
    decode_block_1d_i32,
    i32,
    4,
    zfp_sys::int32,
    dinteger::decode_block_1d_i32,
    zfp_sys::zfp_encode_block_int32_1,
    zfp_sys::zfp_decode_block_int32_1
);
block_decode_compat_int!(
    decode_block_1d_i64,
    i64,
    4,
    zfp_sys::int64,
    dinteger::decode_block_1d_i64,
    zfp_sys::zfp_encode_block_int64_1,
    zfp_sys::zfp_decode_block_int64_1
);
block_decode_compat_float!(
    decode_block_1d_f32,
    f32,
    4,
    any::<f32>(),
    dfloat::decode_block_1d_f32,
    zfp_sys::zfp_encode_block_float_1,
    zfp_sys::zfp_decode_block_float_1
);
block_decode_compat_float!(
    decode_block_1d_f64,
    f64,
    4,
    any::<f64>(),
    dfloat::decode_block_1d_f64,
    zfp_sys::zfp_encode_block_double_1,
    zfp_sys::zfp_decode_block_double_1
);

// ---------------------------------------------------------------------------
// 2-D
// ---------------------------------------------------------------------------

block_decode_compat_int!(
    decode_block_2d_i32,
    i32,
    16,
    zfp_sys::int32,
    dinteger::decode_block_2d_i32,
    zfp_sys::zfp_encode_block_int32_2,
    zfp_sys::zfp_decode_block_int32_2
);
block_decode_compat_int!(
    decode_block_2d_i64,
    i64,
    16,
    zfp_sys::int64,
    dinteger::decode_block_2d_i64,
    zfp_sys::zfp_encode_block_int64_2,
    zfp_sys::zfp_decode_block_int64_2
);
block_decode_compat_float!(
    decode_block_2d_f32,
    f32,
    16,
    any::<f32>(),
    dfloat::decode_block_2d_f32,
    zfp_sys::zfp_encode_block_float_2,
    zfp_sys::zfp_decode_block_float_2
);
block_decode_compat_float!(
    decode_block_2d_f64,
    f64,
    16,
    any::<f64>(),
    dfloat::decode_block_2d_f64,
    zfp_sys::zfp_encode_block_double_2,
    zfp_sys::zfp_decode_block_double_2
);

// ---------------------------------------------------------------------------
// 3-D
// ---------------------------------------------------------------------------

block_decode_compat_int!(
    decode_block_3d_i32,
    i32,
    64,
    zfp_sys::int32,
    dinteger::decode_block_3d_i32,
    zfp_sys::zfp_encode_block_int32_3,
    zfp_sys::zfp_decode_block_int32_3
);
block_decode_compat_int!(
    decode_block_3d_i64,
    i64,
    64,
    zfp_sys::int64,
    dinteger::decode_block_3d_i64,
    zfp_sys::zfp_encode_block_int64_3,
    zfp_sys::zfp_decode_block_int64_3
);
block_decode_compat_float!(
    decode_block_3d_f32,
    f32,
    64,
    any::<f32>(),
    dfloat::decode_block_3d_f32,
    zfp_sys::zfp_encode_block_float_3,
    zfp_sys::zfp_decode_block_float_3
);
block_decode_compat_float!(
    decode_block_3d_f64,
    f64,
    64,
    any::<f64>(),
    dfloat::decode_block_3d_f64,
    zfp_sys::zfp_encode_block_double_3,
    zfp_sys::zfp_decode_block_double_3
);

// ---------------------------------------------------------------------------
// 4-D
// ---------------------------------------------------------------------------

block_decode_compat_int!(
    decode_block_4d_i32,
    i32,
    256,
    zfp_sys::int32,
    dinteger::decode_block_4d_i32,
    zfp_sys::zfp_encode_block_int32_4,
    zfp_sys::zfp_decode_block_int32_4
);
block_decode_compat_int!(
    decode_block_4d_i64,
    i64,
    256,
    zfp_sys::int64,
    dinteger::decode_block_4d_i64,
    zfp_sys::zfp_encode_block_int64_4,
    zfp_sys::zfp_decode_block_int64_4
);
block_decode_compat_float!(
    decode_block_4d_f32,
    f32,
    256,
    any::<f32>(),
    dfloat::decode_block_4d_f32,
    zfp_sys::zfp_encode_block_float_4,
    zfp_sys::zfp_decode_block_float_4
);
block_decode_compat_float!(
    decode_block_4d_f64,
    f64,
    256,
    any::<f64>(),
    dfloat::decode_block_4d_f64,
    zfp_sys::zfp_encode_block_double_4,
    zfp_sys::zfp_decode_block_double_4
);

// ---------------------------------------------------------------------------
// Reversible streams through the strided dispatcher, which zfp-rs-ffi's
// `zfp_decode_block_*` call: as in C, a reversible config selects the
// lossless coder, so a block C encoded decodes exactly.
// ---------------------------------------------------------------------------

macro_rules! reversible_block_decode_compat {
    (
        $test_name:ident,
        $scalar:ty,
        $strategy:expr,
        $dims:expr,
        $strides:expr,
        $c_encode:path,
        $c_decode:path $(,)?
    ) => {
        proptest! {
            #[test]
            fn $test_name(data in prop::collection::vec(
                $strategy,
                $dims.block_size()..=$dims.block_size(),
            )) {
                let mut c = CZfpBlock::new(ZFP_MAX_BITS);
                unsafe { zfp_sys::zfp_stream_set_reversible(c.zfp) };
                unsafe { $c_encode(c.zfp, data.as_ptr()) };
                c.flush();
                c.rewind();
                let mut c_out = vec![<$scalar>::default(); $dims.block_size()];
                let c_bits = unsafe { $c_decode(c.zfp, c_out.as_mut_ptr()) };

                let mut rs_bs = ZfpBitStream::from_bytes(c.encoded_bytes());
                let mut rs_out = vec![<$scalar>::default(); $dims.block_size()];
                let rs_bits = unsafe {
                    decode_block_strided::<$scalar>(
                        &mut rs_bs,
                        rs_out.as_mut_ptr(),
                        $dims,
                        &$strides,
                        &ZfpConfig::reversible(),
                    )
                };

                prop_assert_eq!(rs_bits, c_bits);
                prop_assert_eq!(
                    bytemuck::cast_slice::<$scalar, u8>(&rs_out),
                    bytemuck::cast_slice::<$scalar, u8>(&data)
                );
                prop_assert_eq!(
                    bytemuck::cast_slice::<$scalar, u8>(&rs_out),
                    bytemuck::cast_slice::<$scalar, u8>(&c_out)
                );
            }
        }
    };
}

reversible_block_decode_compat!(
    reversible_decode_block_1d_i32,
    i32,
    any::<i32>(),
    ZfpDimensionality::D1,
    [1],
    zfp_sys::zfp_encode_block_int32_1,
    zfp_sys::zfp_decode_block_int32_1,
);
reversible_block_decode_compat!(
    reversible_decode_block_2d_i64,
    i64,
    any::<i64>(),
    ZfpDimensionality::D2,
    [1, 4],
    zfp_sys::zfp_encode_block_int64_2,
    zfp_sys::zfp_decode_block_int64_2,
);
reversible_block_decode_compat!(
    reversible_decode_block_3d_f32,
    f32,
    any::<f32>(),
    ZfpDimensionality::D3,
    [1, 4, 16],
    zfp_sys::zfp_encode_block_float_3,
    zfp_sys::zfp_decode_block_float_3,
);
reversible_block_decode_compat!(
    reversible_decode_block_4d_f64,
    f64,
    any::<f64>(),
    ZfpDimensionality::D4,
    [1, 4, 16, 64],
    zfp_sys::zfp_encode_block_double_4,
    zfp_sys::zfp_decode_block_double_4,
);
