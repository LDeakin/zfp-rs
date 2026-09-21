//! Byte-exactness against a C zfp built with `ZFP_ROUNDING_MODE=ZFP_ROUND_FIRST`
//! and `ZFP_WITH_TIGHT_ERROR=ON`, which is what `zfp-sys/round-tight-error` does.
//!
//! Cargo unifies `zfp-sys` features across a build, so a rounding-enabled C zfp
//! and a stock one cannot coexist in one invocation. This target is gated behind
//! `c-round-tight-error` and must be selected with `--test c_rounding`, leaving
//! the stock-zfp targets built against stock zfp.
//!
//! `zfp-sys` exposes only the coupled build, so this covers `First` alone.
//! Every mode, `Last` included, is cross-validated through `zfp-rs-ffi`, whose
//! `ffi_compat` suite builds the in-repo `zfp` with matching CMake defines.
#![cfg(feature = "c-round-tight-error")]
#![expect(unsafe_op_in_unsafe_fn)]

use proptest::prelude::*;
use std::ffi::c_void;
use zfp_rs::{
    ZfpBitStream, ZfpConfig, ZfpDimensionality, ZfpField, ZfpRounding, ZfpScalarType,
    ZfpStreamAlignment,
};

/// The mode `zfp-sys/round-tight-error` compiles the C library for.
const ROUNDING: ZfpRounding = ZfpRounding::First { tight_error: true };

const CAPACITY: usize = 1 << 22;

#[derive(Clone, Copy, Debug)]
#[expect(clippy::enum_variant_names, reason = "these are zfp's mode names")]
enum Mode {
    FixedRate(u32),
    FixedPrecision(u32),
    FixedAccuracy(i32),
}

/// Rounding only bites where `prec < intprec`, so cover a spread of precisions.
const MODES: [Mode; 6] = [
    Mode::FixedRate(64),
    Mode::FixedRate(512),
    Mode::FixedPrecision(3),
    Mode::FixedPrecision(17),
    Mode::FixedAccuracy(-40),
    Mode::FixedAccuracy(-4),
];

fn rust_config(mode: Mode, ty: ZfpScalarType, dims: ZfpDimensionality) -> ZfpConfig {
    let config = match mode {
        Mode::FixedRate(bits) => ZfpConfig::fixed_rate(
            f64::from(bits) / f64::from(1u32 << (2 * u32::from(dims))),
            ty,
            dims,
            ZfpStreamAlignment::None,
        ),
        Mode::FixedPrecision(p) => ZfpConfig::fixed_precision(p),
        Mode::FixedAccuracy(e) => ZfpConfig::fixed_accuracy(libm::ldexp(1.0, e)),
    };
    config.with_rounding(ROUNDING)
}

unsafe fn apply_mode_c(zfp: *mut zfp_sys::zfp_stream, mode: Mode, ty: zfp_sys::zfp_type, d: u32) {
    match mode {
        Mode::FixedRate(bits) => {
            let rate = f64::from(bits) / f64::from(1u32 << (2 * d));
            zfp_sys::zfp_stream_set_rate(zfp, rate, ty, d, 0);
        }
        Mode::FixedPrecision(p) => {
            zfp_sys::zfp_stream_set_precision(zfp, p);
        }
        Mode::FixedAccuracy(e) => {
            zfp_sys::zfp_stream_set_accuracy(zfp, libm::ldexp(1.0, e));
        }
    }
}

struct CStream {
    zfp: *mut zfp_sys::zfp_stream,
    bs_ptr: *mut zfp_sys::bitstream,
    buf: Vec<u8>,
}

impl CStream {
    fn new() -> Self {
        let mut buf = vec![0u8; CAPACITY];
        let bs_ptr = unsafe { zfp_sys::stream_open(buf.as_mut_ptr().cast::<c_void>(), CAPACITY) };
        assert!(!bs_ptr.is_null());
        let zfp = unsafe { zfp_sys::zfp_stream_open(bs_ptr) };
        assert!(!zfp.is_null());
        Self { zfp, bs_ptr, buf }
    }

    fn as_bytes(&self) -> &[u8] {
        let size = unsafe { zfp_sys::stream_size(self.bs_ptr) };
        &self.buf[..size]
    }
}

impl Drop for CStream {
    fn drop(&mut self) {
        unsafe {
            zfp_sys::zfp_stream_close(self.zfp);
            zfp_sys::stream_close(self.bs_ptr);
        }
    }
}

/// Compress `data` with the C library; return the committed bytes.
unsafe fn c_compress(
    data: *const c_void,
    c_ty: zfp_sys::zfp_type,
    lens: &[usize],
    mode: Mode,
) -> Vec<u8> {
    let c = CStream::new();
    let d = u32::try_from(lens.len()).unwrap();
    apply_mode_c(c.zfp, mode, c_ty, d);
    let data = data.cast_mut();
    let field = match *lens {
        [nx] => zfp_sys::zfp_field_1d(data, c_ty, nx),
        [nx, ny] => zfp_sys::zfp_field_2d(data, c_ty, nx, ny),
        [nx, ny, nz] => zfp_sys::zfp_field_3d(data, c_ty, nx, ny, nz),
        [nx, ny, nz, nw] => zfp_sys::zfp_field_4d(data, c_ty, nx, ny, nz, nw),
        _ => unreachable!("1-4 dimensions"),
    };
    assert!(!field.is_null());
    let n = zfp_sys::zfp_compress(c.zfp, field);
    zfp_sys::zfp_field_free(field);
    assert!(n > 0, "C zfp_compress returned 0");
    zfp_sys::stream_flush(c.bs_ptr);
    c.as_bytes().to_vec()
}

/// Negative control: proves the linked C library really is the rounding build.
///
/// Without it, a `zfp-sys` built without `round-tight-error` would make every
/// test in this file pass vacuously against stock zfp.
#[test]
fn stock_rounding_does_not_match_the_c_library() {
    let data: Vec<f64> = (0..64).map(|i| f64::from(i) * 0.37 - 7.5).collect();
    let lens = [4usize, 4, 4];
    let mode = Mode::FixedAccuracy(-20);

    let mut bs = ZfpBitStream::new(CAPACITY);
    let never = ZfpConfig::fixed_accuracy(libm::ldexp(1.0, -20));
    bs.compress(&never, &ZfpField::new(&data[..], lens))
        .unwrap();
    let never_bytes = bs.as_bytes().to_vec();

    let c_bytes = unsafe {
        c_compress(
            data.as_ptr().cast::<c_void>(),
            zfp_sys::zfp_type_zfp_type_double,
            &lens,
            mode,
        )
    };
    assert_ne!(
        never_bytes, c_bytes,
        "zfp-sys was not built with round-tight-error; every test here is vacuous"
    );

    // ...and the rounding config does match, so the difference is the rounding.
    let mut bs = ZfpBitStream::new(CAPACITY);
    bs.compress(
        &never.with_rounding(ROUNDING),
        &ZfpField::new(&data[..], lens),
    )
    .unwrap();
    assert_eq!(bs.as_bytes(), &c_bytes[..]);
}

macro_rules! compat {
    ($name:ident, $scalar:ty, $rs_ty:expr, $c_ty:expr, $lens:expr, $strategy:expr) => {
        proptest! {
            #[test]
            fn $name(data in prop::collection::vec($strategy, $lens.iter().product::<usize>())) {
                let lens = $lens;
                let dims = ZfpDimensionality::try_from(u32::try_from(lens.len()).unwrap()).unwrap();
                for mode in MODES {
                    let mut bs = ZfpBitStream::new(CAPACITY);
                    bs.compress(&rust_config(mode, $rs_ty, dims), &ZfpField::new(&data[..], lens))
                        .unwrap();
                    let rs_bytes = bs.as_bytes().to_vec();
                    let c_bytes = unsafe {
                        c_compress(data.as_ptr().cast::<c_void>(), $c_ty, &lens, mode)
                    };
                    prop_assert_eq!(rs_bytes, c_bytes, "mode={:?}", mode);
                }
            }
        }
    };
}

// Subnormals are excluded for the same reason as in `tests/proptest`: they
// overflow `fwd_cast`, where C's cast is implementation-defined.
fn normal_f32() -> impl Strategy<Value = f32> {
    any::<f32>().prop_filter("normal or zero", |f| !f.is_subnormal())
}

fn normal_f64() -> impl Strategy<Value = f64> {
    any::<f64>().prop_filter("normal or zero", |f| !f.is_subnormal())
}

compat!(
    rounding_1d_i32,
    i32,
    ZfpScalarType::Int32,
    zfp_sys::zfp_type_zfp_type_int32,
    [37usize],
    any::<i32>()
);
compat!(
    rounding_1d_i64,
    i64,
    ZfpScalarType::Int64,
    zfp_sys::zfp_type_zfp_type_int64,
    [37usize],
    any::<i64>()
);
compat!(
    rounding_1d_f32,
    f32,
    ZfpScalarType::Float,
    zfp_sys::zfp_type_zfp_type_float,
    [37usize],
    normal_f32()
);
compat!(
    rounding_1d_f64,
    f64,
    ZfpScalarType::Double,
    zfp_sys::zfp_type_zfp_type_double,
    [37usize],
    normal_f64()
);

compat!(
    rounding_2d_i32,
    i32,
    ZfpScalarType::Int32,
    zfp_sys::zfp_type_zfp_type_int32,
    [7usize, 5],
    any::<i32>()
);
compat!(
    rounding_2d_f32,
    f32,
    ZfpScalarType::Float,
    zfp_sys::zfp_type_zfp_type_float,
    [7usize, 5],
    normal_f32()
);
compat!(
    rounding_2d_f64,
    f64,
    ZfpScalarType::Double,
    zfp_sys::zfp_type_zfp_type_double,
    [7usize, 5],
    normal_f64()
);

compat!(
    rounding_3d_i64,
    i64,
    ZfpScalarType::Int64,
    zfp_sys::zfp_type_zfp_type_int64,
    [5usize, 3, 6],
    any::<i64>()
);
compat!(
    rounding_3d_f32,
    f32,
    ZfpScalarType::Float,
    zfp_sys::zfp_type_zfp_type_float,
    [5usize, 3, 6],
    normal_f32()
);
compat!(
    rounding_3d_f64,
    f64,
    ZfpScalarType::Double,
    zfp_sys::zfp_type_zfp_type_double,
    [5usize, 3, 6],
    normal_f64()
);

// 4-D exercises the `*_many_ints_*` decoders (block size 256 > 64).
compat!(
    rounding_4d_i32,
    i32,
    ZfpScalarType::Int32,
    zfp_sys::zfp_type_zfp_type_int32,
    [5usize, 3, 2, 6],
    any::<i32>()
);
compat!(
    rounding_4d_f32,
    f32,
    ZfpScalarType::Float,
    zfp_sys::zfp_type_zfp_type_float,
    [5usize, 3, 2, 6],
    normal_f32()
);
compat!(
    rounding_4d_f64,
    f64,
    ZfpScalarType::Double,
    zfp_sys::zfp_type_zfp_type_double,
    [5usize, 3, 2, 6],
    normal_f64()
);
