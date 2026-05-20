//! Property-based compatibility tests: block_encode_compat.
//!
//! Uses `proptest` + `zfp-sys` to verify byte-for-byte output compatibility
//! with the reference C library for block-level encoding.
//!
//! For each scalar type (i32, i64, f32, f64) and each dimensionality (1–4),
//! generates a random block, encodes it with both the Rust typed encode function
//! and the C `zfp_encode_block_*` function, and asserts the bitstreams are identical.
//!
//! The Rust typed functions and the C block functions take the same parameters:
//! `minbits`, `maxbits`, `maxprec` (and `minexp` for floats). We use fixed-rate
//! mode: `minbits = maxbits = block_size * ZFP_RATE_PARAM_BITS`.

#![cfg(feature = "ffi")]

use proptest::prelude::*;
use zfp_rs::ZfpBitStream;
use zfp_rs::codec::encode::{float as efloat, integer as einteger};

// ---------------------------------------------------------------------------
// Proptest strategies for normal (non-subnormal) floats
//
// Subnormal inputs cause overflow in fwd_cast (scale factor 2^(62-emax) with
// emax = -1022 clamp → s = 2^1084 which overflows f64 to infinity). The C
// library comment in encodef.c warns about this and the ZFP_WITH_DAZ flag
// exists precisely to treat subnormals as zero. Since C's cast of infinity
// to integer is implementation-defined while Rust's `as` saturates, the two
// implementations diverge on subnormal inputs. We exclude them.
// ---------------------------------------------------------------------------

fn normal_f32() -> impl Strategy<Value = f32> {
    any::<f32>().prop_filter("must be normal or zero", |f| !f.is_subnormal())
}

fn normal_f64() -> impl Strategy<Value = f64> {
    any::<f64>().prop_filter("must be normal or zero", |f| !f.is_subnormal())
}

// ---------------------------------------------------------------------------
// Constants matching the C test suite (universalConsts.h)
// ---------------------------------------------------------------------------

const ZFP_MAX_PREC: u32 = 64;
const ZFP_MIN_EXP: i32 = -1074;
// Fixed-rate: 19 bits per element; MAXBITS = block_size * 19
const ZFP_RATE_PARAM_BITS: u32 = 19;

// ---------------------------------------------------------------------------
// C-side helper: zfp_stream + bitstream with the given fixed-rate params
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
        // minbits = maxbits for fixed-rate; maxprec = 64; minexp = -1074
        unsafe {
            zfp_sys::zfp_stream_set_params(zfp, maxbits, maxbits, ZFP_MAX_PREC, ZFP_MIN_EXP);
        }
        Self { zfp, bs_ptr, buf }
    }

    fn flush(&mut self) {
        unsafe { zfp_sys::stream_flush(self.bs_ptr) };
    }

    /// Committed bytes after flush.
    fn as_bytes(&self) -> &[u8] {
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
// Macro: generate a proptest for one integer (scalar, dims, block_size) combo
// ---------------------------------------------------------------------------

macro_rules! block_encode_compat_int {
    (
        $test_name:ident,
        $scalar:ty,
        $block_size:expr,
        $array_ty:ty,
        $rs_fn:path,
        $c_fn:path
    ) => {
        proptest! {
            #[test]
            fn $test_name(data in prop::collection::vec(
                any::<$scalar>(),
                $block_size..$block_size + 1,
            )) {
                let maxbits = ($block_size as u32) * ZFP_RATE_PARAM_BITS;
                let block: &[$scalar; $block_size] = data.as_slice().try_into().unwrap();

                // Rust side
                let mut rs_bs = ZfpBitStream::new(CZfpBlock::CAPACITY);
                $rs_fn(&mut rs_bs, block, maxbits, maxbits, ZFP_MAX_PREC);
                rs_bs.flush();

                // C side
                let mut c = CZfpBlock::new(maxbits);
                unsafe { $c_fn(c.zfp, data.as_ptr().cast::<$array_ty>()) };
                c.flush();

                prop_assert_eq!(
                    rs_bs.as_bytes(),
                    c.as_bytes(),
                    "bitstream mismatch for {}",
                    stringify!($test_name)
                );
            }
        }
    };
}

// ---------------------------------------------------------------------------
// Macro: generate a proptest for one float (scalar, dims, block_size) combo
// ---------------------------------------------------------------------------

macro_rules! block_encode_compat_float {
    (
        $test_name:ident,
        $scalar:ty,
        $block_size:expr,
        $c_scalar:ty,
        $normal_strategy:expr,
        $rs_fn:path,
        $c_fn:path
    ) => {
        proptest! {
            #[test]
            fn $test_name(data in prop::collection::vec(
                $normal_strategy,
                $block_size..$block_size + 1,
            )) {
                let maxbits = ($block_size as u32) * ZFP_RATE_PARAM_BITS;
                let block: &[$scalar; $block_size] = data.as_slice().try_into().unwrap();

                // Rust side
                let mut rs_bs = ZfpBitStream::new(CZfpBlock::CAPACITY);
                $rs_fn(&mut rs_bs, block, maxbits, maxbits, ZFP_MAX_PREC, ZFP_MIN_EXP);
                rs_bs.flush();

                // C side
                let mut c = CZfpBlock::new(maxbits);
                unsafe { $c_fn(c.zfp, data.as_ptr().cast::<$c_scalar>()) };
                c.flush();

                prop_assert_eq!(
                    rs_bs.as_bytes(),
                    c.as_bytes(),
                    "bitstream mismatch for {}",
                    stringify!($test_name)
                );
            }
        }
    };
}

// ---------------------------------------------------------------------------
// 1-D (block_size = 4)
// ---------------------------------------------------------------------------

block_encode_compat_int!(
    encode_block_1d_i32,
    i32,
    4,
    zfp_sys::int32,
    einteger::encode_block_1d_i32,
    zfp_sys::zfp_encode_block_int32_1
);
block_encode_compat_int!(
    encode_block_1d_i64,
    i64,
    4,
    zfp_sys::int64,
    einteger::encode_block_1d_i64,
    zfp_sys::zfp_encode_block_int64_1
);
block_encode_compat_float!(
    encode_block_1d_f32,
    f32,
    4,
    f32,
    normal_f32(),
    efloat::encode_block_1d_f32,
    zfp_sys::zfp_encode_block_float_1
);
block_encode_compat_float!(
    encode_block_1d_f64,
    f64,
    4,
    f64,
    normal_f64(),
    efloat::encode_block_1d_f64,
    zfp_sys::zfp_encode_block_double_1
);

// ---------------------------------------------------------------------------
// 2-D (block_size = 16)
// ---------------------------------------------------------------------------

block_encode_compat_int!(
    encode_block_2d_i32,
    i32,
    16,
    zfp_sys::int32,
    einteger::encode_block_2d_i32,
    zfp_sys::zfp_encode_block_int32_2
);
block_encode_compat_int!(
    encode_block_2d_i64,
    i64,
    16,
    zfp_sys::int64,
    einteger::encode_block_2d_i64,
    zfp_sys::zfp_encode_block_int64_2
);
block_encode_compat_float!(
    encode_block_2d_f32,
    f32,
    16,
    f32,
    normal_f32(),
    efloat::encode_block_2d_f32,
    zfp_sys::zfp_encode_block_float_2
);
block_encode_compat_float!(
    encode_block_2d_f64,
    f64,
    16,
    f64,
    normal_f64(),
    efloat::encode_block_2d_f64,
    zfp_sys::zfp_encode_block_double_2
);

// ---------------------------------------------------------------------------
// 3-D (block_size = 64)
// ---------------------------------------------------------------------------

block_encode_compat_int!(
    encode_block_3d_i32,
    i32,
    64,
    zfp_sys::int32,
    einteger::encode_block_3d_i32,
    zfp_sys::zfp_encode_block_int32_3
);
block_encode_compat_int!(
    encode_block_3d_i64,
    i64,
    64,
    zfp_sys::int64,
    einteger::encode_block_3d_i64,
    zfp_sys::zfp_encode_block_int64_3
);
block_encode_compat_float!(
    encode_block_3d_f32,
    f32,
    64,
    f32,
    normal_f32(),
    efloat::encode_block_3d_f32,
    zfp_sys::zfp_encode_block_float_3
);
block_encode_compat_float!(
    encode_block_3d_f64,
    f64,
    64,
    f64,
    normal_f64(),
    efloat::encode_block_3d_f64,
    zfp_sys::zfp_encode_block_double_3
);

// ---------------------------------------------------------------------------
// 4-D (block_size = 256)
// ---------------------------------------------------------------------------

block_encode_compat_int!(
    encode_block_4d_i32,
    i32,
    256,
    zfp_sys::int32,
    einteger::encode_block_4d_i32,
    zfp_sys::zfp_encode_block_int32_4
);
block_encode_compat_int!(
    encode_block_4d_i64,
    i64,
    256,
    zfp_sys::int64,
    einteger::encode_block_4d_i64,
    zfp_sys::zfp_encode_block_int64_4
);
block_encode_compat_float!(
    encode_block_4d_f32,
    f32,
    256,
    f32,
    normal_f32(),
    efloat::encode_block_4d_f32,
    zfp_sys::zfp_encode_block_float_4
);
block_encode_compat_float!(
    encode_block_4d_f64,
    f64,
    256,
    f64,
    normal_f64(),
    efloat::encode_block_4d_f64,
    zfp_sys::zfp_encode_block_double_4
);
