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
//! Subnormal floats are excluded for the same reason as in block_encode_compat:
//! the fwd_cast scale-factor overflow produces implementation-defined behaviour
//! in C and Rust's saturating `as` cast produces a different bit pattern.

#![cfg(feature = "ffi")]

use proptest::prelude::*;
use zfp_rs::ZfpBitStream;
use zfp_rs::ZfpRounding;
use zfp_rs::codec::decode::{float as dfloat, integer as dinteger};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const ZFP_MAX_PREC: u32 = 64;
const ZFP_MIN_EXP: i32 = -1074;
const ZFP_RATE_PARAM_BITS: u32 = 19;

// ---------------------------------------------------------------------------
// Strategies for normal floats
// ---------------------------------------------------------------------------

fn normal_f32() -> impl Strategy<Value = f32> {
    any::<f32>().prop_filter("must be normal or zero", |f| !f.is_subnormal())
}

fn normal_f64() -> impl Strategy<Value = f64> {
    any::<f64>().prop_filter("must be normal or zero", |f| !f.is_subnormal())
}

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
                let rs_out = $rs_decode(&mut rs_bs, maxbits, maxbits, ZFP_MAX_PREC, ZfpRounding::Never);

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
                let rs_out = $rs_decode(&mut rs_bs, maxbits, maxbits, ZFP_MAX_PREC, ZFP_MIN_EXP, ZfpRounding::Never);

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
    normal_f32(),
    dfloat::decode_block_1d_f32,
    zfp_sys::zfp_encode_block_float_1,
    zfp_sys::zfp_decode_block_float_1
);
block_decode_compat_float!(
    decode_block_1d_f64,
    f64,
    4,
    normal_f64(),
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
    normal_f32(),
    dfloat::decode_block_2d_f32,
    zfp_sys::zfp_encode_block_float_2,
    zfp_sys::zfp_decode_block_float_2
);
block_decode_compat_float!(
    decode_block_2d_f64,
    f64,
    16,
    normal_f64(),
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
    normal_f32(),
    dfloat::decode_block_3d_f32,
    zfp_sys::zfp_encode_block_float_3,
    zfp_sys::zfp_decode_block_float_3
);
block_decode_compat_float!(
    decode_block_3d_f64,
    f64,
    64,
    normal_f64(),
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
    normal_f32(),
    dfloat::decode_block_4d_f32,
    zfp_sys::zfp_encode_block_float_4,
    zfp_sys::zfp_decode_block_float_4
);
block_decode_compat_float!(
    decode_block_4d_f64,
    f64,
    256,
    normal_f64(),
    dfloat::decode_block_4d_f64,
    zfp_sys::zfp_encode_block_double_4,
    zfp_sys::zfp_decode_block_double_4
);
