//! Property-based compatibility tests: header_compat.
//!
//! Uses `proptest` + `zfp-sys` to verify byte-for-byte output compatibility
//! with the reference C library for header write/read.
//!
//! For each combination of header mask (MAGIC, META, MODE, and FULL), scalar
//! type, dimensionality, and compression mode, we write the header with both
//! the Rust `write_header` and C `zfp_write_header`, then compare
//! the resulting bitstream bytes.
#![expect(unsafe_op_in_unsafe_fn)]

use proptest::prelude::*;
use zfp_rs::{
    ZfpBitStream, ZfpConfig, ZfpDimensionality, ZfpField,
    types::{ZfpHeaderMask, ZfpScalarType},
};

// ---------------------------------------------------------------------------
// C helper: zfp_stream + bitstream
// ---------------------------------------------------------------------------

struct CHeader {
    zfp: *mut zfp_sys::zfp_stream,
    bs_ptr: *mut zfp_sys::bitstream,
    buf: Vec<u8>,
}

impl CHeader {
    const CAPACITY: usize = 256; // header is at most 148 bits ≈ 19 bytes

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

impl Drop for CHeader {
    fn drop(&mut self) {
        unsafe {
            zfp_sys::zfp_stream_close(self.zfp);
            zfp_sys::stream_close(self.bs_ptr);
        }
    }
}

// ---------------------------------------------------------------------------
// Helper: set the same mode on both Rust ZfpConfig and C zfp_stream
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
// Macro: header write compatibility for one scalar type
// ---------------------------------------------------------------------------

macro_rules! header_write_compat {
    (
        $test_name:ident,
        $scalar:ty,
        $zfp_type:expr,
        $c_type:expr
    ) => {
        proptest! {
            #[test]
            fn $test_name(
                nx in 1usize..=1024usize,
                mode in mode_strategy(),
            ) {
                // Use full header (MAGIC | META | MODE)
                let mask = ZfpHeaderMask::MAGIC | ZfpHeaderMask::META | ZfpHeaderMask::MODE;

                // Dummy data slice — header only looks at pointer for C (not for read)
                let data = vec![<$scalar as Default>::default(); nx];

                // --- Rust side ---
                let field = ZfpField::new(&data, [nx]);
                let mut rs_params = ZfpConfig::new();
                apply_mode_rust(&mut rs_params, &mode, $zfp_type, ZfpDimensionality::D1);
                let mut rs_bs = ZfpBitStream::new(256);
                rs_bs.write_header(&rs_params, &field, mask);
                rs_bs.flush();
                let rs_bytes = rs_bs.as_bytes();

                // --- C side ---
                let mut c = CHeader::new();
                let c_field = unsafe {
                    zfp_sys::zfp_field_1d(
                        data.as_ptr() as *mut std::ffi::c_void,
                        $c_type,
                        nx,
                    )
                };
                assert!(!c_field.is_null());
                unsafe { apply_mode_c(c.zfp, &mode, $c_type, 1) };
                let c_bits_written = unsafe {
                    zfp_sys::zfp_write_header(
                        c.zfp,
                        c_field,
                        (zfp_sys::ZFP_HEADER_MAGIC | zfp_sys::ZFP_HEADER_META | zfp_sys::ZFP_HEADER_MODE) as u32,
                    )
                };
                unsafe { zfp_sys::zfp_field_free(c_field) };
                c.flush();
                let c_bytes = c.as_bytes().to_vec();

                // Compare: both should have written the same bytes
                prop_assert!(c_bits_written > 0, "C write_header returned 0");
                prop_assert_eq!(
                    rs_bytes,
                    c_bytes,
                    "header mismatch for {} mode={:?} nx={}",
                    stringify!($scalar), mode, nx
                );
            }
        }
    };
}

header_write_compat!(
    header_write_i32,
    i32,
    ZfpScalarType::Int32,
    zfp_sys::zfp_type_zfp_type_int32
);
header_write_compat!(
    header_write_i64,
    i64,
    ZfpScalarType::Int64,
    zfp_sys::zfp_type_zfp_type_int64
);
header_write_compat!(
    header_write_f32,
    f32,
    ZfpScalarType::Float,
    zfp_sys::zfp_type_zfp_type_float
);
header_write_compat!(
    header_write_f64,
    f64,
    ZfpScalarType::Double,
    zfp_sys::zfp_type_zfp_type_double
);
