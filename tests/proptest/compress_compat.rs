//! Property-based compatibility tests: compress_compat.
//!
//! Uses `proptest` + `zfp-sys` to verify byte-for-byte output compatibility
//! with the reference C library for full-field compression via
//! `ZfpConfig::compress` vs `zfp_compress`.
//!
//! For each scalar type (i32, i64, f32, f64) and dimensionality (1–4), generates
//! a random field with a random compression mode and asserts that the Rust and C
//! compressed bitstreams are identical.
#![expect(unsafe_op_in_unsafe_fn)]

use proptest::prelude::*;
use zfp_rs::{
    ZfpBitStream, ZfpConfig, ZfpDimensionality, ZfpExecution, ZfpField, types::ZfpScalarType,
};

// ---------------------------------------------------------------------------
// Proptest strategies for normal (non-subnormal) floats
//
// Subnormal inputs overflow fwd_cast (scale factor 2^1084 → infinity) so the
// C's implementation-defined cast diverges from Rust's saturating `as`.
// ---------------------------------------------------------------------------

fn normal_f32() -> impl Strategy<Value = f32> {
    any::<f32>().prop_filter("must be normal or zero", |f| !f.is_subnormal())
}

fn normal_f64() -> impl Strategy<Value = f64> {
    any::<f64>().prop_filter("must be normal or zero", |f| !f.is_subnormal())
}

// ---------------------------------------------------------------------------
// Execution strategy
// ---------------------------------------------------------------------------

fn execution_strategy() -> impl Strategy<Value = ZfpExecution> {
    #[cfg(feature = "rayon")]
    {
        prop_oneof![
            Just(ZfpExecution::Serial),
            (0u32..=4u32, 0u32..=64u32).prop_map(|(threads, chunk_size)| {
                ZfpExecution::Rayon {
                    threads,
                    chunk_size,
                }
            }),
        ]
    }
    #[cfg(not(feature = "rayon"))]
    {
        Just(ZfpExecution::Serial)
    }
}

// ---------------------------------------------------------------------------
// Compression mode
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
enum Mode {
    FixedRate(u32),      // bits per block (1..=2048)
    FixedPrecision(u32), // precision (1..=64)
    FixedAccuracy(i32),  // min_exp (-1074..=843)
    Reversible,
}

fn mode_strategy() -> impl Strategy<Value = Mode> {
    prop_oneof![
        (1u32..=2048u32).prop_map(Mode::FixedRate),
        (1u32..=64u32).prop_map(Mode::FixedPrecision),
        (-1074i32..=843i32).prop_map(Mode::FixedAccuracy),
        Just(Mode::Reversible),
    ]
}

fn apply_mode_rust(
    config: &mut ZfpConfig,
    mode: &Mode,
    ty: ZfpScalarType,
    dims: ZfpDimensionality,
) {
    match *mode {
        Mode::FixedRate(bits) => {
            *config = ZfpConfig::fixed_rate(
                f64::from(bits) / f64::from(1u32 << (2 * u32::from(dims))),
                ty,
                dims,
                zfp_rs::ZfpStreamAlignment::None,
            );
        }
        Mode::FixedPrecision(p) => {
            *config = ZfpConfig::fixed_precision(p);
        }
        Mode::FixedAccuracy(e) => {
            *config = ZfpConfig::fixed_accuracy(libm::ldexp(1.0, e));
        }
        Mode::Reversible => {
            *config = ZfpConfig::reversible();
        }
    }
}

unsafe fn apply_mode_c(
    zfp: *mut zfp_sys::zfp_stream,
    mode: &Mode,
    c_type: zfp_sys::zfp_type,
    dims: u32,
) {
    match *mode {
        Mode::FixedRate(bits) => {
            zfp_sys::zfp_stream_set_rate(
                zfp,
                f64::from(bits) / f64::from(1u32 << (2 * dims)),
                c_type,
                dims,
                0,
            );
        }
        Mode::FixedPrecision(p) => {
            zfp_sys::zfp_stream_set_precision(zfp, p);
        }
        Mode::FixedAccuracy(e) => {
            zfp_sys::zfp_stream_set_accuracy(zfp, libm::ldexp(1.0, e));
        }
        Mode::Reversible => {
            zfp_sys::zfp_stream_set_reversible(zfp);
        }
    }
}

// ---------------------------------------------------------------------------
// C-side helper: zfp_stream + bitstream for full-field compress
// ---------------------------------------------------------------------------

struct CStream {
    zfp: *mut zfp_sys::zfp_stream,
    bs_ptr: *mut zfp_sys::bitstream,
    buf: Vec<u8>,
}

impl CStream {
    // 4 MB — large enough for all field sizes we generate
    const CAPACITY: usize = 1 << 22;

    fn new() -> Self {
        let mut buf = vec![0u8; Self::CAPACITY];
        let bs_ptr = unsafe {
            zfp_sys::stream_open(buf.as_mut_ptr().cast::<std::ffi::c_void>(), Self::CAPACITY)
        };
        assert!(!bs_ptr.is_null());
        let zfp = unsafe { zfp_sys::zfp_stream_open(bs_ptr) };
        assert!(!zfp.is_null());
        Self { zfp, bs_ptr, buf }
    }

    fn flush(&mut self) {
        unsafe { zfp_sys::stream_flush(self.bs_ptr) };
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

// ---------------------------------------------------------------------------
// 1D integer tests
// ---------------------------------------------------------------------------

macro_rules! compress_compat_1d_int {
    ($test_name:ident, $scalar:ty, $zfp_type:expr, $c_type:expr) => {
        proptest! {
            #[test]
            fn $test_name(
                nx in 1usize..=64usize,
                mode in mode_strategy(),
                exec in execution_strategy(),
                data in prop::collection::vec(any::<$scalar>(), 1..=64),
            ) {
                let nx = nx.min(data.len());
                let data = &data[..nx];

                let field = ZfpField::new(data, [nx]);
                let mut rs = ZfpConfig::new();
                apply_mode_rust(&mut rs, &mode, $zfp_type, ZfpDimensionality::D1);
                let mut bs = ZfpBitStream::new(CStream::CAPACITY);
                bs.compress_with_execution(&rs, &field, exec).unwrap();

                let rs_bytes = bs.as_bytes().to_vec();

                let mut c = CStream::new();
                unsafe { apply_mode_c(c.zfp, &mode, $c_type, 1) };
                let c_field = unsafe {
                    zfp_sys::zfp_field_1d(data.as_ptr() as *mut std::ffi::c_void, $c_type, nx)
                };
                assert!(!c_field.is_null());
                let n = unsafe { zfp_sys::zfp_compress(c.zfp, c_field) };
                unsafe { zfp_sys::zfp_field_free(c_field) };
                prop_assert!(n > 0, "C zfp_compress returned 0");
                c.flush();
                let c_bytes = c.as_bytes().to_vec();

                prop_assert_eq!(rs_bytes, c_bytes, "mode={:?} nx={}", mode, nx);
            }
        }
    };
}

compress_compat_1d_int!(
    compress_1d_i32,
    i32,
    ZfpScalarType::Int32,
    zfp_sys::zfp_type_zfp_type_int32
);
compress_compat_1d_int!(
    compress_1d_i64,
    i64,
    ZfpScalarType::Int64,
    zfp_sys::zfp_type_zfp_type_int64
);

// ---------------------------------------------------------------------------
// 1D float tests
// ---------------------------------------------------------------------------

macro_rules! compress_compat_1d_float {
    ($test_name:ident, $scalar:ty, $zfp_type:expr, $c_type:expr, $normal:expr) => {
        proptest! {
            #[test]
            fn $test_name(
                nx in 1usize..=64usize,
                mode in mode_strategy(),
                exec in execution_strategy(),
                data in prop::collection::vec($normal, 1..=64),
            ) {
                let nx = nx.min(data.len());
                let data = &data[..nx];

                let field = ZfpField::new(data, [nx]);
                let mut rs = ZfpConfig::new();
                apply_mode_rust(&mut rs, &mode, $zfp_type, ZfpDimensionality::D1);
                let mut bs = ZfpBitStream::new(CStream::CAPACITY);
                bs.compress_with_execution(&rs, &field, exec).unwrap();

                let rs_bytes = bs.as_bytes().to_vec();

                let mut c = CStream::new();
                unsafe { apply_mode_c(c.zfp, &mode, $c_type, 1) };
                let c_field = unsafe {
                    zfp_sys::zfp_field_1d(data.as_ptr() as *mut std::ffi::c_void, $c_type, nx)
                };
                assert!(!c_field.is_null());
                let n = unsafe { zfp_sys::zfp_compress(c.zfp, c_field) };
                unsafe { zfp_sys::zfp_field_free(c_field) };
                prop_assert!(n > 0, "C zfp_compress returned 0");
                c.flush();
                let c_bytes = c.as_bytes().to_vec();

                prop_assert_eq!(rs_bytes, c_bytes, "mode={:?} nx={}", mode, nx);
            }
        }
    };
}

compress_compat_1d_float!(
    compress_1d_f32,
    f32,
    ZfpScalarType::Float,
    zfp_sys::zfp_type_zfp_type_float,
    normal_f32()
);
compress_compat_1d_float!(
    compress_1d_f64,
    f64,
    ZfpScalarType::Double,
    zfp_sys::zfp_type_zfp_type_double,
    normal_f64()
);

// ---------------------------------------------------------------------------
// 2D integer tests
// ---------------------------------------------------------------------------

macro_rules! compress_compat_2d_int {
    ($test_name:ident, $scalar:ty, $zfp_type:expr, $c_type:expr) => {
        proptest! {
            #[test]
            fn $test_name(
                nx in 1usize..=16usize,
                ny in 1usize..=16usize,
                mode in mode_strategy(),
                exec in execution_strategy(),
                data in prop::collection::vec(any::<$scalar>(), 1..=256),
            ) {
                let n = (nx * ny).min(data.len());
                let nx = if n < nx { 1 } else { nx };
                let ny = if n / nx == 0 { 1 } else { n / nx };
                let data = &data[..nx * ny];

                let field = ZfpField::new(data, [nx, ny]);
                let mut rs = ZfpConfig::new();
                apply_mode_rust(&mut rs, &mode, $zfp_type, ZfpDimensionality::D2);
                let mut bs = ZfpBitStream::new(CStream::CAPACITY);
                bs.compress_with_execution(&rs, &field, exec).unwrap();

                let rs_bytes = bs.as_bytes().to_vec();

                let mut c = CStream::new();
                unsafe { apply_mode_c(c.zfp, &mode, $c_type, 2) };
                let c_field = unsafe {
                    zfp_sys::zfp_field_2d(data.as_ptr() as *mut std::ffi::c_void, $c_type, nx, ny)
                };
                assert!(!c_field.is_null());
                let n_written = unsafe { zfp_sys::zfp_compress(c.zfp, c_field) };
                unsafe { zfp_sys::zfp_field_free(c_field) };
                prop_assert!(n_written > 0, "C zfp_compress returned 0");
                c.flush();
                let c_bytes = c.as_bytes().to_vec();

                prop_assert_eq!(rs_bytes, c_bytes, "mode={:?} nx={} ny={}", mode, nx, ny);
            }
        }
    };
}

compress_compat_2d_int!(
    compress_2d_i32,
    i32,
    ZfpScalarType::Int32,
    zfp_sys::zfp_type_zfp_type_int32
);
compress_compat_2d_int!(
    compress_2d_i64,
    i64,
    ZfpScalarType::Int64,
    zfp_sys::zfp_type_zfp_type_int64
);

// ---------------------------------------------------------------------------
// 2D float tests
// ---------------------------------------------------------------------------

macro_rules! compress_compat_2d_float {
    ($test_name:ident, $scalar:ty, $zfp_type:expr, $c_type:expr, $normal:expr) => {
        proptest! {
            #[test]
            fn $test_name(
                nx in 1usize..=16usize,
                ny in 1usize..=16usize,
                mode in mode_strategy(),
                exec in execution_strategy(),
                data in prop::collection::vec($normal, 1..=256),
            ) {
                let n = (nx * ny).min(data.len());
                let nx = if n < nx { 1 } else { nx };
                let ny = if n / nx == 0 { 1 } else { n / nx };
                let data = &data[..nx * ny];

                let field = ZfpField::new(data, [nx, ny]);
                let mut rs = ZfpConfig::new();
                apply_mode_rust(&mut rs, &mode, $zfp_type, ZfpDimensionality::D2);
                let mut bs = ZfpBitStream::new(CStream::CAPACITY);
                bs.compress_with_execution(&rs, &field, exec).unwrap();

                let rs_bytes = bs.as_bytes().to_vec();

                let mut c = CStream::new();
                unsafe { apply_mode_c(c.zfp, &mode, $c_type, 2) };
                let c_field = unsafe {
                    zfp_sys::zfp_field_2d(data.as_ptr() as *mut std::ffi::c_void, $c_type, nx, ny)
                };
                assert!(!c_field.is_null());
                let n_written = unsafe { zfp_sys::zfp_compress(c.zfp, c_field) };
                unsafe { zfp_sys::zfp_field_free(c_field) };
                prop_assert!(n_written > 0, "C zfp_compress returned 0");
                c.flush();
                let c_bytes = c.as_bytes().to_vec();

                prop_assert_eq!(rs_bytes, c_bytes, "mode={:?} nx={} ny={}", mode, nx, ny);
            }
        }
    };
}

compress_compat_2d_float!(
    compress_2d_f32,
    f32,
    ZfpScalarType::Float,
    zfp_sys::zfp_type_zfp_type_float,
    normal_f32()
);
compress_compat_2d_float!(
    compress_2d_f64,
    f64,
    ZfpScalarType::Double,
    zfp_sys::zfp_type_zfp_type_double,
    normal_f64()
);

// ---------------------------------------------------------------------------
// 3D integer tests
// ---------------------------------------------------------------------------

macro_rules! compress_compat_3d_int {
    ($test_name:ident, $scalar:ty, $zfp_type:expr, $c_type:expr) => {
        proptest! {
            #[test]
            fn $test_name(
                nx in 1usize..=8usize,
                ny in 1usize..=8usize,
                nz in 1usize..=8usize,
                mode in mode_strategy(),
                exec in execution_strategy(),
                data in prop::collection::vec(any::<$scalar>(), 1..=512),
            ) {
                let n = (nx * ny * nz).min(data.len());
                let nx = if n < nx { 1 } else { nx };
                let ny = if n / nx < ny { (n / nx).max(1) } else { ny };
                let nz = if n / (nx * ny) == 0 { 1 } else { n / (nx * ny) };
                let data = &data[..nx * ny * nz];

                let field = ZfpField::new(data, [nx, ny, nz]);
                let mut rs = ZfpConfig::new();
                apply_mode_rust(&mut rs, &mode, $zfp_type, ZfpDimensionality::D3);
                let mut bs = ZfpBitStream::new(CStream::CAPACITY);
                bs.compress_with_execution(&rs, &field, exec).unwrap();

                let rs_bytes = bs.as_bytes().to_vec();

                let mut c = CStream::new();
                unsafe { apply_mode_c(c.zfp, &mode, $c_type, 3) };
                let c_field = unsafe {
                    zfp_sys::zfp_field_3d(data.as_ptr() as *mut std::ffi::c_void, $c_type, nx, ny, nz)
                };
                assert!(!c_field.is_null());
                let n_written = unsafe { zfp_sys::zfp_compress(c.zfp, c_field) };
                unsafe { zfp_sys::zfp_field_free(c_field) };
                prop_assert!(n_written > 0, "C zfp_compress returned 0");
                c.flush();
                let c_bytes = c.as_bytes().to_vec();

                prop_assert_eq!(rs_bytes, c_bytes, "mode={:?} nx={} ny={} nz={}", mode, nx, ny, nz);
            }
        }
    };
}

compress_compat_3d_int!(
    compress_3d_i32,
    i32,
    ZfpScalarType::Int32,
    zfp_sys::zfp_type_zfp_type_int32
);
compress_compat_3d_int!(
    compress_3d_i64,
    i64,
    ZfpScalarType::Int64,
    zfp_sys::zfp_type_zfp_type_int64
);

// ---------------------------------------------------------------------------
// 3D float tests
// ---------------------------------------------------------------------------

macro_rules! compress_compat_3d_float {
    ($test_name:ident, $scalar:ty, $zfp_type:expr, $c_type:expr, $normal:expr) => {
        proptest! {
            #[test]
            fn $test_name(
                nx in 1usize..=8usize,
                ny in 1usize..=8usize,
                nz in 1usize..=8usize,
                mode in mode_strategy(),
                exec in execution_strategy(),
                data in prop::collection::vec($normal, 1..=512),
            ) {
                let n = (nx * ny * nz).min(data.len());
                let nx = if n < nx { 1 } else { nx };
                let ny = if n / nx < ny { (n / nx).max(1) } else { ny };
                let nz = if n / (nx * ny) == 0 { 1 } else { n / (nx * ny) };
                let data = &data[..nx * ny * nz];

                let field = ZfpField::new(data, [nx, ny, nz]);
                let mut rs = ZfpConfig::new();
                apply_mode_rust(&mut rs, &mode, $zfp_type, ZfpDimensionality::D3);
                let mut bs = ZfpBitStream::new(CStream::CAPACITY);
                bs.compress_with_execution(&rs, &field, exec).unwrap();

                let rs_bytes = bs.as_bytes().to_vec();

                let mut c = CStream::new();
                unsafe { apply_mode_c(c.zfp, &mode, $c_type, 3) };
                let c_field = unsafe {
                    zfp_sys::zfp_field_3d(data.as_ptr() as *mut std::ffi::c_void, $c_type, nx, ny, nz)
                };
                assert!(!c_field.is_null());
                let n_written = unsafe { zfp_sys::zfp_compress(c.zfp, c_field) };
                unsafe { zfp_sys::zfp_field_free(c_field) };
                prop_assert!(n_written > 0, "C zfp_compress returned 0");
                c.flush();
                let c_bytes = c.as_bytes().to_vec();

                prop_assert_eq!(rs_bytes, c_bytes, "mode={:?} nx={} ny={} nz={}", mode, nx, ny, nz);
            }
        }
    };
}

compress_compat_3d_float!(
    compress_3d_f32,
    f32,
    ZfpScalarType::Float,
    zfp_sys::zfp_type_zfp_type_float,
    normal_f32()
);
compress_compat_3d_float!(
    compress_3d_f64,
    f64,
    ZfpScalarType::Double,
    zfp_sys::zfp_type_zfp_type_double,
    normal_f64()
);

// ---------------------------------------------------------------------------
// 4D integer tests
// ---------------------------------------------------------------------------

macro_rules! compress_compat_4d_int {
    ($test_name:ident, $scalar:ty, $zfp_type:expr, $c_type:expr) => {
        proptest! {
            #[test]
            fn $test_name(
                nx in 1usize..=4usize,
                ny in 1usize..=4usize,
                nz in 1usize..=4usize,
                nw in 1usize..=4usize,
                mode in mode_strategy(),
                exec in execution_strategy(),
                data in prop::collection::vec(any::<$scalar>(), 1..=256),
            ) {
                let n = (nx * ny * nz * nw).min(data.len());
                let nx = if n < nx { 1 } else { nx };
                let ny = if n / nx < ny { (n / nx).max(1) } else { ny };
                let nz = if n / (nx * ny) < nz { (n / (nx * ny)).max(1) } else { nz };
                let nw = if n / (nx * ny * nz) == 0 { 1 } else { n / (nx * ny * nz) };
                let data = &data[..nx * ny * nz * nw];

                let field = ZfpField::new(data, [nx, ny, nz, nw]);
                let mut rs = ZfpConfig::new();
                apply_mode_rust(&mut rs, &mode, $zfp_type, ZfpDimensionality::D4);
                let mut bs = ZfpBitStream::new(CStream::CAPACITY);
                bs.compress_with_execution(&rs, &field, exec).unwrap();

                let rs_bytes = bs.as_bytes().to_vec();

                let mut c = CStream::new();
                unsafe { apply_mode_c(c.zfp, &mode, $c_type, 4) };
                let c_field = unsafe {
                    zfp_sys::zfp_field_4d(data.as_ptr() as *mut std::ffi::c_void, $c_type, nx, ny, nz, nw)
                };
                assert!(!c_field.is_null());
                let n_written = unsafe { zfp_sys::zfp_compress(c.zfp, c_field) };
                unsafe { zfp_sys::zfp_field_free(c_field) };
                prop_assert!(n_written > 0, "C zfp_compress returned 0");
                c.flush();
                let c_bytes = c.as_bytes().to_vec();

                prop_assert_eq!(rs_bytes, c_bytes, "mode={:?} nx={} ny={} nz={} nw={}", mode, nx, ny, nz, nw);
            }
        }
    };
}

compress_compat_4d_int!(
    compress_4d_i32,
    i32,
    ZfpScalarType::Int32,
    zfp_sys::zfp_type_zfp_type_int32
);
compress_compat_4d_int!(
    compress_4d_i64,
    i64,
    ZfpScalarType::Int64,
    zfp_sys::zfp_type_zfp_type_int64
);

// ---------------------------------------------------------------------------
// 4D float tests
// ---------------------------------------------------------------------------

macro_rules! compress_compat_4d_float {
    ($test_name:ident, $scalar:ty, $zfp_type:expr, $c_type:expr, $normal:expr) => {
        proptest! {
            #[test]
            fn $test_name(
                nx in 1usize..=4usize,
                ny in 1usize..=4usize,
                nz in 1usize..=4usize,
                nw in 1usize..=4usize,
                mode in mode_strategy(),
                exec in execution_strategy(),
                data in prop::collection::vec($normal, 1..=256),
            ) {
                let n = (nx * ny * nz * nw).min(data.len());
                let nx = if n < nx { 1 } else { nx };
                let ny = if n / nx < ny { (n / nx).max(1) } else { ny };
                let nz = if n / (nx * ny) < nz { (n / (nx * ny)).max(1) } else { nz };
                let nw = if n / (nx * ny * nz) == 0 { 1 } else { n / (nx * ny * nz) };
                let data = &data[..nx * ny * nz * nw];

                let field = ZfpField::new(data, [nx, ny, nz, nw]);
                let mut rs = ZfpConfig::new();
                apply_mode_rust(&mut rs, &mode, $zfp_type, ZfpDimensionality::D4);
                let mut bs = ZfpBitStream::new(CStream::CAPACITY);
                bs.compress_with_execution(&rs, &field, exec).unwrap();

                let rs_bytes = bs.as_bytes().to_vec();

                let mut c = CStream::new();
                unsafe { apply_mode_c(c.zfp, &mode, $c_type, 4) };
                let c_field = unsafe {
                    zfp_sys::zfp_field_4d(data.as_ptr() as *mut std::ffi::c_void, $c_type, nx, ny, nz, nw)
                };
                assert!(!c_field.is_null());
                let n_written = unsafe { zfp_sys::zfp_compress(c.zfp, c_field) };
                unsafe { zfp_sys::zfp_field_free(c_field) };
                prop_assert!(n_written > 0, "C zfp_compress returned 0");
                c.flush();
                let c_bytes = c.as_bytes().to_vec();

                prop_assert_eq!(rs_bytes, c_bytes, "mode={:?} nx={} ny={} nz={} nw={}", mode, nx, ny, nz, nw);
            }
        }
    };
}

compress_compat_4d_float!(
    compress_4d_f32,
    f32,
    ZfpScalarType::Float,
    zfp_sys::zfp_type_zfp_type_float,
    normal_f32()
);
compress_compat_4d_float!(
    compress_4d_f64,
    f64,
    ZfpScalarType::Double,
    zfp_sys::zfp_type_zfp_type_double,
    normal_f64()
);
