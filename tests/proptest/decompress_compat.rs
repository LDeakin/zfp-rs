//! Property-based compatibility tests: decompress_compat.
//!
//! Uses `proptest` + `zfp-sys` to verify byte-for-byte output compatibility
//! with the reference C library for full-field decompression via
//! `ZfpConfig::decompress` vs `zfp_decompress`.
//!
//! Strategy: compress with C (`zfp_compress`) to get a trusted bitstream, then
//! decode with both Rust and C and assert the decoded arrays are identical.
#![expect(unsafe_op_in_unsafe_fn)]

use proptest::prelude::*;
use zfp_rs::{
    ZfpBitStream, ZfpConfig, ZfpDimensionality, ZfpExecution, ZfpFieldMut, types::ZfpScalarType,
};

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
// Normal (non-subnormal) float strategies
// ---------------------------------------------------------------------------

fn normal_f32() -> impl Strategy<Value = f32> {
    any::<f32>().prop_filter("must be normal or zero", |f| !f.is_subnormal())
}

fn normal_f64() -> impl Strategy<Value = f64> {
    any::<f64>().prop_filter("must be normal or zero", |f| !f.is_subnormal())
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
// C-side helper: zfp_stream + bitstream for full-field compress/decompress
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

    fn rewind(&mut self) {
        unsafe { zfp_sys::zfp_stream_rewind(self.zfp) };
    }

    fn compressed_bytes(&self) -> &[u8] {
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

/// Build a `ZfpBitStream` for decompression from a compressed byte slice.
///
/// Adds one extra zero word (8 bytes) of padding so that decoder lookahead
/// reads past the last compressed word don't panic. This matches the C
/// library's behavior where the output buffer is always larger than the
/// compressed data.
fn bs_for_decompress(compressed: &[u8]) -> ZfpBitStream {
    let mut buf = compressed.to_vec();
    buf.extend_from_slice(&[0u8; 8]);
    ZfpBitStream::from_bytes(&buf)
}

// ---------------------------------------------------------------------------
// 1D integer tests
// ---------------------------------------------------------------------------

macro_rules! decompress_compat_1d_int {
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

                // Compress with C to get trusted reference bitstream.
                let mut c = CStream::new();
                unsafe { apply_mode_c(c.zfp, &mode, $c_type, 1) };
                let c_field_in = unsafe {
                    zfp_sys::zfp_field_1d(data.as_ptr() as *mut std::ffi::c_void, $c_type, nx)
                };
                assert!(!c_field_in.is_null());
                let n = unsafe { zfp_sys::zfp_compress(c.zfp, c_field_in) };
                unsafe { zfp_sys::zfp_field_free(c_field_in) };
                prop_assume!(n > 0);
                c.flush();
                let compressed = c.compressed_bytes().to_vec();

                // Decode with Rust.
                let mut rs_out = vec![<$scalar>::default(); nx];
                {
                    let mut rs = ZfpConfig::new();
                    apply_mode_rust(&mut rs, &mode, $zfp_type, ZfpDimensionality::D1);
                    let mut bs = bs_for_decompress(&compressed);
                    let mut field_mut = ZfpFieldMut::new(&mut rs_out, [nx]);
                    bs.decompress_with_execution(&rs, &mut field_mut, exec).unwrap();
                }

                // Decode with C.
                let mut c_out = vec![<$scalar>::default(); nx];
                c.rewind();
                let c_field_out = unsafe {
                    zfp_sys::zfp_field_1d(c_out.as_mut_ptr().cast::<std::ffi::c_void>(), $c_type, nx)
                };
                assert!(!c_field_out.is_null());
                let n2 = unsafe { zfp_sys::zfp_decompress(c.zfp, c_field_out) };
                unsafe { zfp_sys::zfp_field_free(c_field_out) };
                prop_assert!(n2 > 0, "C zfp_decompress returned 0");

                prop_assert_eq!(rs_out, c_out, "mode={:?} nx={}", mode, nx);
            }
        }
    };
}

decompress_compat_1d_int!(
    decompress_1d_i32,
    i32,
    ZfpScalarType::Int32,
    zfp_sys::zfp_type_zfp_type_int32
);
decompress_compat_1d_int!(
    decompress_1d_i64,
    i64,
    ZfpScalarType::Int64,
    zfp_sys::zfp_type_zfp_type_int64
);

// ---------------------------------------------------------------------------
// 1D float tests
// ---------------------------------------------------------------------------

macro_rules! decompress_compat_1d_float {
    ($test_name:ident, $scalar:ty, $zfp_type:expr, $c_type:expr, $to_bits:expr, $normal:expr) => {
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

                // Compress with C.
                let mut c = CStream::new();
                unsafe { apply_mode_c(c.zfp, &mode, $c_type, 1) };
                let c_field_in = unsafe {
                    zfp_sys::zfp_field_1d(data.as_ptr() as *mut std::ffi::c_void, $c_type, nx)
                };
                assert!(!c_field_in.is_null());
                let n = unsafe { zfp_sys::zfp_compress(c.zfp, c_field_in) };
                unsafe { zfp_sys::zfp_field_free(c_field_in) };
                prop_assume!(n > 0);
                c.flush();
                let compressed = c.compressed_bytes().to_vec();

                // Decode with Rust.
                let mut rs_out = vec![<$scalar>::default(); nx];
                {
                    let mut rs = ZfpConfig::new();
                    apply_mode_rust(&mut rs, &mode, $zfp_type, ZfpDimensionality::D1);
                    let mut bs = bs_for_decompress(&compressed);
                    let mut field_mut = ZfpFieldMut::new(&mut rs_out, [nx]);
                    bs.decompress_with_execution(&rs, &mut field_mut, exec).unwrap();
                }

                // Decode with C.
                let mut c_out = vec![<$scalar>::default(); nx];
                c.rewind();
                let c_field_out = unsafe {
                    zfp_sys::zfp_field_1d(c_out.as_mut_ptr().cast::<std::ffi::c_void>(), $c_type, nx)
                };
                assert!(!c_field_out.is_null());
                let n2 = unsafe { zfp_sys::zfp_decompress(c.zfp, c_field_out) };
                unsafe { zfp_sys::zfp_field_free(c_field_out) };
                prop_assert!(n2 > 0, "C zfp_decompress returned 0");

                // Compare float outputs bit-for-bit (handles NaN identity).
                let rs_bits: Vec<_> = rs_out.iter().map($to_bits).collect();
                let c_bits: Vec<_> = c_out.iter().map($to_bits).collect();
                prop_assert_eq!(rs_bits, c_bits, "mode={:?} nx={}", mode, nx);
            }
        }
    };
}

decompress_compat_1d_float!(
    decompress_1d_f32,
    f32,
    ZfpScalarType::Float,
    zfp_sys::zfp_type_zfp_type_float,
    |f: &f32| f.to_bits(),
    normal_f32()
);
decompress_compat_1d_float!(
    decompress_1d_f64,
    f64,
    ZfpScalarType::Double,
    zfp_sys::zfp_type_zfp_type_double,
    |f: &f64| f.to_bits(),
    normal_f64()
);

// ---------------------------------------------------------------------------
// 2D integer tests
// ---------------------------------------------------------------------------

macro_rules! decompress_compat_2d_int {
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

                let mut c = CStream::new();
                unsafe { apply_mode_c(c.zfp, &mode, $c_type, 2) };
                let c_field_in = unsafe {
                    zfp_sys::zfp_field_2d(data.as_ptr() as *mut std::ffi::c_void, $c_type, nx, ny)
                };
                assert!(!c_field_in.is_null());
                let n = unsafe { zfp_sys::zfp_compress(c.zfp, c_field_in) };
                unsafe { zfp_sys::zfp_field_free(c_field_in) };
                prop_assume!(n > 0);
                c.flush();
                let compressed = c.compressed_bytes().to_vec();

                let mut rs_out = vec![<$scalar>::default(); nx * ny];
                {
                    let mut rs = ZfpConfig::new();
                    apply_mode_rust(&mut rs, &mode, $zfp_type, ZfpDimensionality::D2);
                    let mut bs = bs_for_decompress(&compressed);
                    let mut field_mut = ZfpFieldMut::new(&mut rs_out, [nx, ny]);
                    bs.decompress_with_execution(&rs, &mut field_mut, exec).unwrap();
                }

                let mut c_out = vec![<$scalar>::default(); nx * ny];
                c.rewind();
                let c_field_out = unsafe {
                    zfp_sys::zfp_field_2d(c_out.as_mut_ptr().cast::<std::ffi::c_void>(), $c_type, nx, ny)
                };
                assert!(!c_field_out.is_null());
                let n2 = unsafe { zfp_sys::zfp_decompress(c.zfp, c_field_out) };
                unsafe { zfp_sys::zfp_field_free(c_field_out) };
                prop_assert!(n2 > 0, "C zfp_decompress returned 0");

                prop_assert_eq!(rs_out, c_out, "mode={:?} nx={} ny={}", mode, nx, ny);
            }
        }
    };
}

decompress_compat_2d_int!(
    decompress_2d_i32,
    i32,
    ZfpScalarType::Int32,
    zfp_sys::zfp_type_zfp_type_int32
);
decompress_compat_2d_int!(
    decompress_2d_i64,
    i64,
    ZfpScalarType::Int64,
    zfp_sys::zfp_type_zfp_type_int64
);

// ---------------------------------------------------------------------------
// 2D float tests
// ---------------------------------------------------------------------------

macro_rules! decompress_compat_2d_float {
    ($test_name:ident, $scalar:ty, $zfp_type:expr, $c_type:expr, $to_bits:expr, $normal:expr) => {
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

                let mut c = CStream::new();
                unsafe { apply_mode_c(c.zfp, &mode, $c_type, 2) };
                let c_field_in = unsafe {
                    zfp_sys::zfp_field_2d(data.as_ptr() as *mut std::ffi::c_void, $c_type, nx, ny)
                };
                assert!(!c_field_in.is_null());
                let n = unsafe { zfp_sys::zfp_compress(c.zfp, c_field_in) };
                unsafe { zfp_sys::zfp_field_free(c_field_in) };
                prop_assume!(n > 0);
                c.flush();
                let compressed = c.compressed_bytes().to_vec();

                let mut rs_out = vec![<$scalar>::default(); nx * ny];
                {
                    let mut rs = ZfpConfig::new();
                    apply_mode_rust(&mut rs, &mode, $zfp_type, ZfpDimensionality::D2);
                    let mut bs = bs_for_decompress(&compressed);
                    let mut field_mut = ZfpFieldMut::new(&mut rs_out, [nx, ny]);
                    bs.decompress_with_execution(&rs, &mut field_mut, exec).unwrap();
                }

                let mut c_out = vec![<$scalar>::default(); nx * ny];
                c.rewind();
                let c_field_out = unsafe {
                    zfp_sys::zfp_field_2d(c_out.as_mut_ptr().cast::<std::ffi::c_void>(), $c_type, nx, ny)
                };
                assert!(!c_field_out.is_null());
                let n2 = unsafe { zfp_sys::zfp_decompress(c.zfp, c_field_out) };
                unsafe { zfp_sys::zfp_field_free(c_field_out) };
                prop_assert!(n2 > 0, "C zfp_decompress returned 0");

                let rs_bits: Vec<_> = rs_out.iter().map($to_bits).collect();
                let c_bits: Vec<_> = c_out.iter().map($to_bits).collect();
                prop_assert_eq!(rs_bits, c_bits, "mode={:?} nx={} ny={}", mode, nx, ny);
            }
        }
    };
}

decompress_compat_2d_float!(
    decompress_2d_f32,
    f32,
    ZfpScalarType::Float,
    zfp_sys::zfp_type_zfp_type_float,
    |f: &f32| f.to_bits(),
    normal_f32()
);
decompress_compat_2d_float!(
    decompress_2d_f64,
    f64,
    ZfpScalarType::Double,
    zfp_sys::zfp_type_zfp_type_double,
    |f: &f64| f.to_bits(),
    normal_f64()
);

// ---------------------------------------------------------------------------
// 3D integer tests
// ---------------------------------------------------------------------------

macro_rules! decompress_compat_3d_int {
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

                let mut c = CStream::new();
                unsafe { apply_mode_c(c.zfp, &mode, $c_type, 3) };
                let c_field_in = unsafe {
                    zfp_sys::zfp_field_3d(data.as_ptr() as *mut std::ffi::c_void, $c_type, nx, ny, nz)
                };
                assert!(!c_field_in.is_null());
                let n = unsafe { zfp_sys::zfp_compress(c.zfp, c_field_in) };
                unsafe { zfp_sys::zfp_field_free(c_field_in) };
                prop_assume!(n > 0);
                c.flush();
                let compressed = c.compressed_bytes().to_vec();

                let mut rs_out = vec![<$scalar>::default(); nx * ny * nz];
                {
                    let mut rs = ZfpConfig::new();
                    apply_mode_rust(&mut rs, &mode, $zfp_type, ZfpDimensionality::D3);
                    let mut bs = bs_for_decompress(&compressed);
                    let mut field_mut = ZfpFieldMut::new(&mut rs_out, [nx, ny, nz]);
                    bs.decompress_with_execution(&rs, &mut field_mut, exec).unwrap();
                }

                let mut c_out = vec![<$scalar>::default(); nx * ny * nz];
                c.rewind();
                let c_field_out = unsafe {
                    zfp_sys::zfp_field_3d(c_out.as_mut_ptr().cast::<std::ffi::c_void>(), $c_type, nx, ny, nz)
                };
                assert!(!c_field_out.is_null());
                let n2 = unsafe { zfp_sys::zfp_decompress(c.zfp, c_field_out) };
                unsafe { zfp_sys::zfp_field_free(c_field_out) };
                prop_assert!(n2 > 0, "C zfp_decompress returned 0");

                prop_assert_eq!(rs_out, c_out, "mode={:?} nx={} ny={} nz={}", mode, nx, ny, nz);
            }
        }
    };
}

decompress_compat_3d_int!(
    decompress_3d_i32,
    i32,
    ZfpScalarType::Int32,
    zfp_sys::zfp_type_zfp_type_int32
);
decompress_compat_3d_int!(
    decompress_3d_i64,
    i64,
    ZfpScalarType::Int64,
    zfp_sys::zfp_type_zfp_type_int64
);

// ---------------------------------------------------------------------------
// 3D float tests
// ---------------------------------------------------------------------------

macro_rules! decompress_compat_3d_float {
    ($test_name:ident, $scalar:ty, $zfp_type:expr, $c_type:expr, $to_bits:expr, $normal:expr) => {
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

                let mut c = CStream::new();
                unsafe { apply_mode_c(c.zfp, &mode, $c_type, 3) };
                let c_field_in = unsafe {
                    zfp_sys::zfp_field_3d(data.as_ptr() as *mut std::ffi::c_void, $c_type, nx, ny, nz)
                };
                assert!(!c_field_in.is_null());
                let n = unsafe { zfp_sys::zfp_compress(c.zfp, c_field_in) };
                unsafe { zfp_sys::zfp_field_free(c_field_in) };
                prop_assume!(n > 0);
                c.flush();
                let compressed = c.compressed_bytes().to_vec();

                let mut rs_out = vec![<$scalar>::default(); nx * ny * nz];
                {
                    let mut rs = ZfpConfig::new();
                    apply_mode_rust(&mut rs, &mode, $zfp_type, ZfpDimensionality::D3);
                    let mut bs = bs_for_decompress(&compressed);
                    let mut field_mut = ZfpFieldMut::new(&mut rs_out, [nx, ny, nz]);
                    bs.decompress_with_execution(&rs, &mut field_mut, exec).unwrap();
                }

                let mut c_out = vec![<$scalar>::default(); nx * ny * nz];
                c.rewind();
                let c_field_out = unsafe {
                    zfp_sys::zfp_field_3d(c_out.as_mut_ptr().cast::<std::ffi::c_void>(), $c_type, nx, ny, nz)
                };
                assert!(!c_field_out.is_null());
                let n2 = unsafe { zfp_sys::zfp_decompress(c.zfp, c_field_out) };
                unsafe { zfp_sys::zfp_field_free(c_field_out) };
                prop_assert!(n2 > 0, "C zfp_decompress returned 0");

                let rs_bits: Vec<_> = rs_out.iter().map($to_bits).collect();
                let c_bits: Vec<_> = c_out.iter().map($to_bits).collect();
                prop_assert_eq!(rs_bits, c_bits, "mode={:?} nx={} ny={} nz={}", mode, nx, ny, nz);
            }
        }
    };
}

decompress_compat_3d_float!(
    decompress_3d_f32,
    f32,
    ZfpScalarType::Float,
    zfp_sys::zfp_type_zfp_type_float,
    |f: &f32| f.to_bits(),
    normal_f32()
);
decompress_compat_3d_float!(
    decompress_3d_f64,
    f64,
    ZfpScalarType::Double,
    zfp_sys::zfp_type_zfp_type_double,
    |f: &f64| f.to_bits(),
    normal_f64()
);

// ---------------------------------------------------------------------------
// 4D integer tests
// ---------------------------------------------------------------------------

macro_rules! decompress_compat_4d_int {
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

                let mut c = CStream::new();
                unsafe { apply_mode_c(c.zfp, &mode, $c_type, 4) };
                let c_field_in = unsafe {
                    zfp_sys::zfp_field_4d(data.as_ptr() as *mut std::ffi::c_void, $c_type, nx, ny, nz, nw)
                };
                assert!(!c_field_in.is_null());
                let n = unsafe { zfp_sys::zfp_compress(c.zfp, c_field_in) };
                unsafe { zfp_sys::zfp_field_free(c_field_in) };
                prop_assume!(n > 0);
                c.flush();
                let compressed = c.compressed_bytes().to_vec();

                let mut rs_out = vec![<$scalar>::default(); nx * ny * nz * nw];
                {
                    let mut rs = ZfpConfig::new();
                    apply_mode_rust(&mut rs, &mode, $zfp_type, ZfpDimensionality::D4);
                    let mut bs = bs_for_decompress(&compressed);
                    let mut field_mut = ZfpFieldMut::new(&mut rs_out, [nx, ny, nz, nw]);
                    bs.decompress_with_execution(&rs, &mut field_mut, exec).unwrap();
                }

                let mut c_out = vec![<$scalar>::default(); nx * ny * nz * nw];
                c.rewind();
                let c_field_out = unsafe {
                    zfp_sys::zfp_field_4d(c_out.as_mut_ptr().cast::<std::ffi::c_void>(), $c_type, nx, ny, nz, nw)
                };
                assert!(!c_field_out.is_null());
                let n2 = unsafe { zfp_sys::zfp_decompress(c.zfp, c_field_out) };
                unsafe { zfp_sys::zfp_field_free(c_field_out) };
                prop_assert!(n2 > 0, "C zfp_decompress returned 0");

                prop_assert_eq!(rs_out, c_out, "mode={:?} nx={} ny={} nz={} nw={}", mode, nx, ny, nz, nw);
            }
        }
    };
}

decompress_compat_4d_int!(
    decompress_4d_i32,
    i32,
    ZfpScalarType::Int32,
    zfp_sys::zfp_type_zfp_type_int32
);
decompress_compat_4d_int!(
    decompress_4d_i64,
    i64,
    ZfpScalarType::Int64,
    zfp_sys::zfp_type_zfp_type_int64
);

// ---------------------------------------------------------------------------
// 4D float tests
// ---------------------------------------------------------------------------

macro_rules! decompress_compat_4d_float {
    ($test_name:ident, $scalar:ty, $zfp_type:expr, $c_type:expr, $to_bits:expr, $normal:expr) => {
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

                let mut c = CStream::new();
                unsafe { apply_mode_c(c.zfp, &mode, $c_type, 4) };
                let c_field_in = unsafe {
                    zfp_sys::zfp_field_4d(data.as_ptr() as *mut std::ffi::c_void, $c_type, nx, ny, nz, nw)
                };
                assert!(!c_field_in.is_null());
                let n = unsafe { zfp_sys::zfp_compress(c.zfp, c_field_in) };
                unsafe { zfp_sys::zfp_field_free(c_field_in) };
                prop_assume!(n > 0);
                c.flush();
                let compressed = c.compressed_bytes().to_vec();

                let mut rs_out = vec![<$scalar>::default(); nx * ny * nz * nw];
                {
                    let mut rs = ZfpConfig::new();
                    apply_mode_rust(&mut rs, &mode, $zfp_type, ZfpDimensionality::D4);
                    let mut bs = bs_for_decompress(&compressed);
                    let mut field_mut = ZfpFieldMut::new(&mut rs_out, [nx, ny, nz, nw]);
                    bs.decompress_with_execution(&rs, &mut field_mut, exec).unwrap();
                }

                let mut c_out = vec![<$scalar>::default(); nx * ny * nz * nw];
                c.rewind();
                let c_field_out = unsafe {
                    zfp_sys::zfp_field_4d(c_out.as_mut_ptr().cast::<std::ffi::c_void>(), $c_type, nx, ny, nz, nw)
                };
                assert!(!c_field_out.is_null());
                let n2 = unsafe { zfp_sys::zfp_decompress(c.zfp, c_field_out) };
                unsafe { zfp_sys::zfp_field_free(c_field_out) };
                prop_assert!(n2 > 0, "C zfp_decompress returned 0");

                let rs_bits: Vec<_> = rs_out.iter().map($to_bits).collect();
                let c_bits: Vec<_> = c_out.iter().map($to_bits).collect();
                prop_assert_eq!(rs_bits, c_bits, "mode={:?} nx={} ny={} nz={} nw={}", mode, nx, ny, nz, nw);
            }
        }
    };
}

decompress_compat_4d_float!(
    decompress_4d_f32,
    f32,
    ZfpScalarType::Float,
    zfp_sys::zfp_type_zfp_type_float,
    |f: &f32| f.to_bits(),
    normal_f32()
);
decompress_compat_4d_float!(
    decompress_4d_f64,
    f64,
    ZfpScalarType::Double,
    zfp_sys::zfp_type_zfp_type_double,
    |f: &f64| f.to_bits(),
    normal_f64()
);
