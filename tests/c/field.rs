#![allow(clippy::cast_possible_truncation)] // usize→u32 for test constants
#![allow(clippy::doc_markdown)] // C constants in test docs
//! Port of `zfp/tests/src/misc/testZfpField{1-4}{d,f}.c` (8 files).
//!
//! Each C file sets DIMS, ZFP_TYPE, SCALAR, NX, SX (etc.) and includes
//! `zfpFieldBase.c`.  We replicate that pattern with Rust generics/macros.

// ---------------------------------------------------------------------------
// Macro: generate the 8 field tests for a given (dimensionality, scalar type)
// ---------------------------------------------------------------------------

/// Generate an `nx * ny * nz * nw` capacity vec (unused dims collapse to 1).
macro_rules! field_tests {
    // 1-D arm
    (mod=$mod_name:ident, ty=$ty:ty, dims=1, nx=$nx:expr, sx=$sx:expr) => {
        mod $mod_name {
            use std::mem::size_of;
            use zfp_rs::ZfpField;

            #[test]
            fn given_contiguous_data_is_contiguous_returns_true() {
                let data = vec![<$ty>::default(); $nx];
                let field = ZfpField::new(&data, [$nx]);
                assert!(field.is_contiguous());
            }

            #[test]
            fn given_noncontiguous_data_is_contiguous_returns_false() {
                let n = ($sx) * ($nx - 1) + 1;
                let data = vec![<$ty>::default(); n];
                let field = ZfpField::new_strided(&data, [$nx], [$sx as isize]);
                assert!(!field.is_contiguous());
            }

            #[test]
            fn when_no_field_data_field_begin_returns_null() {
                let field = ZfpField::new(&[] as &[$ty], [$nx]);
                assert!(field.begin().is_none());
            }

            #[test]
            fn when_contiguous_data_field_begins_at_data_pointer() {
                let data = vec![<$ty>::default(); $nx];
                let field = ZfpField::new(&data, [$nx]);
                assert_eq!(field.begin().unwrap(), data.as_ptr().cast());
            }

            #[test]
            fn when_noncontiguous_data_with_negative_stride_field_begins_at_correct_location() {
                let n = ($sx) * ($nx - 1) + 1;
                let data = vec![<$ty>::default(); n];
                let field = ZfpField::new_strided(&data, [$nx], [-($sx as isize)]);
                // `data` covers the whole strided span starting at its lowest
                // address, so `begin` is the slice start whatever the stride
                // signs; the element at index 0 is the *last* one here.
                assert_eq!(field.begin().unwrap(), data.as_ptr().cast());
            }

            #[test]
            fn given_field_precision_correct() {
                let field = ZfpField::new(&[] as &[$ty], [0]);
                assert_eq!(field.precision(), (size_of::<$ty>() * 8) as u32);
            }

            #[test]
            fn given_contiguous_data_field_size_bytes_correct() {
                let data = vec![<$ty>::default(); $nx];
                let field = ZfpField::new(&data, [$nx]);
                assert_eq!(field.size_bytes(), $nx * size_of::<$ty>());
            }

            #[test]
            fn given_noncontiguous_data_field_size_bytes_correct() {
                let n = ($sx) * ($nx - 1) + 1;
                let data = vec![<$ty>::default(); n];
                let field = ZfpField::new_strided(&data, [$nx], [$sx as isize]);
                assert_eq!(field.size_bytes(), n * size_of::<$ty>());
            }
        }
    };

    // 2-D arm
    (mod=$mod_name:ident, ty=$ty:ty, dims=2,
     nx=$nx:expr, ny=$ny:expr, sx=$sx:expr, sy=$sy:expr) => {
        mod $mod_name {
            use std::mem::size_of;
            use zfp_rs::ZfpField;

            #[test]
            fn given_contiguous_data_is_contiguous_returns_true() {
                let data = vec![<$ty>::default(); $nx * $ny];
                let field = ZfpField::new(&data, [$nx, $ny]);
                assert!(field.is_contiguous());
            }

            #[test]
            fn given_noncontiguous_data_is_contiguous_returns_false() {
                let n = ($sx) * ($nx - 1) + ($sy) * ($ny - 1) + 1;
                let data = vec![<$ty>::default(); n];
                let field = ZfpField::new_strided(&data, [$nx, $ny], [$sx as isize, $sy as isize]);
                assert!(!field.is_contiguous());
            }

            #[test]
            fn when_no_field_data_field_begin_returns_null() {
                let field = ZfpField::new(&[] as &[$ty], [$nx, $ny]);
                assert!(field.begin().is_none());
            }

            #[test]
            fn when_contiguous_data_field_begins_at_data_pointer() {
                let data = vec![<$ty>::default(); $nx * $ny];
                let field = ZfpField::new(&data, [$nx, $ny]);
                assert_eq!(field.begin().unwrap(), data.as_ptr().cast());
            }

            #[test]
            fn when_noncontiguous_data_with_negative_stride_field_begins_at_correct_location() {
                let n = ($sx) * ($nx - 1) + ($sy) * ($ny - 1) + 1;
                let data = vec![<$ty>::default(); n];
                let field =
                    ZfpField::new_strided(&data, [$nx, $ny], [-($sx as isize), -($sy as isize)]);
                // `begin` is the slice start: `data` covers the whole strided
                // span from its lowest address.
                assert_eq!(field.begin().unwrap(), data.as_ptr().cast());
            }

            #[test]
            fn given_field_precision_correct() {
                let field = ZfpField::new(&[] as &[$ty], [0]);
                assert_eq!(field.precision(), (size_of::<$ty>() * 8) as u32);
            }

            #[test]
            fn given_contiguous_data_field_size_bytes_correct() {
                let data = vec![<$ty>::default(); $nx * $ny];
                let field = ZfpField::new(&data, [$nx, $ny]);
                assert_eq!(field.size_bytes(), $nx * $ny * size_of::<$ty>());
            }

            #[test]
            fn given_noncontiguous_data_field_size_bytes_correct() {
                let n = ($sx) * ($nx - 1) + ($sy) * ($ny - 1) + 1;
                let data = vec![<$ty>::default(); n];
                let field = ZfpField::new_strided(&data, [$nx, $ny], [$sx as isize, $sy as isize]);
                assert_eq!(field.size_bytes(), n * size_of::<$ty>());
            }
        }
    };

    // 3-D arm
    (mod=$mod_name:ident, ty=$ty:ty, dims=3,
     nx=$nx:expr, ny=$ny:expr, nz=$nz:expr,
     sx=$sx:expr, sy=$sy:expr, sz=$sz:expr) => {
        mod $mod_name {
            use std::mem::size_of;
            use zfp_rs::ZfpField;

            #[test]
            fn given_contiguous_data_is_contiguous_returns_true() {
                let data = vec![<$ty>::default(); $nx * $ny * $nz];
                let field = ZfpField::new(&data, [$nx, $ny, $nz]);
                assert!(field.is_contiguous());
            }

            #[test]
            fn given_noncontiguous_data_is_contiguous_returns_false() {
                let n = ($sx) * ($nx - 1) + ($sy) * ($ny - 1) + ($sz) * ($nz - 1) + 1;
                let data = vec![<$ty>::default(); n];
                let field = ZfpField::new_strided(
                    &data,
                    [$nx, $ny, $nz],
                    [$sx as isize, $sy as isize, $sz as isize],
                );
                assert!(!field.is_contiguous());
            }

            #[test]
            fn when_no_field_data_field_begin_returns_null() {
                let field = ZfpField::new(&[] as &[$ty], [$nx, $ny, $nz]);
                assert!(field.begin().is_none());
            }

            #[test]
            fn when_contiguous_data_field_begins_at_data_pointer() {
                let data = vec![<$ty>::default(); $nx * $ny * $nz];
                let field = ZfpField::new(&data, [$nx, $ny, $nz]);
                assert_eq!(field.begin().unwrap(), data.as_ptr().cast());
            }

            #[test]
            fn when_noncontiguous_data_with_negative_stride_field_begins_at_correct_location() {
                let n = ($sx) * ($nx - 1) + ($sy) * ($ny - 1) + ($sz) * ($nz - 1) + 1;
                let data = vec![<$ty>::default(); n];
                let field = ZfpField::new_strided(
                    &data,
                    [$nx, $ny, $nz],
                    [-($sx as isize), -($sy as isize), -($sz as isize)],
                );
                // `begin` is the slice start: `data` covers the whole strided
                // span from its lowest address.
                assert_eq!(field.begin().unwrap(), data.as_ptr().cast());
            }

            #[test]
            fn given_field_precision_correct() {
                let field = ZfpField::new(&[] as &[$ty], [0]);
                assert_eq!(field.precision(), (size_of::<$ty>() * 8) as u32);
            }

            #[test]
            fn given_contiguous_data_field_size_bytes_correct() {
                let data = vec![<$ty>::default(); $nx * $ny * $nz];
                let field = ZfpField::new(&data, [$nx, $ny, $nz]);
                assert_eq!(field.size_bytes(), $nx * $ny * $nz * size_of::<$ty>());
            }

            #[test]
            fn given_noncontiguous_data_field_size_bytes_correct() {
                let n = ($sx) * ($nx - 1) + ($sy) * ($ny - 1) + ($sz) * ($nz - 1) + 1;
                let data = vec![<$ty>::default(); n];
                let field = ZfpField::new_strided(
                    &data,
                    [$nx, $ny, $nz],
                    [$sx as isize, $sy as isize, $sz as isize],
                );
                assert_eq!(field.size_bytes(), n * size_of::<$ty>());
            }
        }
    };

    // 4-D arm
    (mod=$mod_name:ident, ty=$ty:ty, dims=4,
     nx=$nx:expr, ny=$ny:expr, nz=$nz:expr, nw=$nw:expr,
     sx=$sx:expr, sy=$sy:expr, sz=$sz:expr, sw=$sw:expr) => {
        mod $mod_name {
            use std::mem::size_of;
            use zfp_rs::ZfpField;

            #[test]
            fn given_contiguous_data_is_contiguous_returns_true() {
                let data = vec![<$ty>::default(); $nx * $ny * $nz * $nw];
                let field = ZfpField::new(&data, [$nx, $ny, $nz, $nw]);
                assert!(field.is_contiguous());
            }

            #[test]
            fn given_noncontiguous_data_is_contiguous_returns_false() {
                let n = ($sx) * ($nx - 1)
                    + ($sy) * ($ny - 1)
                    + ($sz) * ($nz - 1)
                    + ($sw) * ($nw - 1)
                    + 1;
                let data = vec![<$ty>::default(); n];
                let field = ZfpField::new_strided(
                    &data,
                    [$nx, $ny, $nz, $nw],
                    [$sx as isize, $sy as isize, $sz as isize, $sw as isize],
                );
                assert!(!field.is_contiguous());
            }

            #[test]
            fn when_no_field_data_field_begin_returns_null() {
                let field = ZfpField::new(&[] as &[$ty], [$nx, $ny, $nz, $nw]);
                assert!(field.begin().is_none());
            }

            #[test]
            fn when_contiguous_data_field_begins_at_data_pointer() {
                let data = vec![<$ty>::default(); $nx * $ny * $nz * $nw];
                let field = ZfpField::new(&data, [$nx, $ny, $nz, $nw]);
                assert_eq!(field.begin().unwrap(), data.as_ptr().cast());
            }

            #[test]
            fn when_noncontiguous_data_with_negative_stride_field_begins_at_correct_location() {
                let n = ($sx) * ($nx - 1)
                    + ($sy) * ($ny - 1)
                    + ($sz) * ($nz - 1)
                    + ($sw) * ($nw - 1)
                    + 1;
                let data = vec![<$ty>::default(); n];
                let field = ZfpField::new_strided(
                    &data,
                    [$nx, $ny, $nz, $nw],
                    [
                        -($sx as isize),
                        -($sy as isize),
                        -($sz as isize),
                        -($sw as isize),
                    ],
                );
                // `begin` is the slice start: `data` covers the whole strided
                // span from its lowest address.
                assert_eq!(field.begin().unwrap(), data.as_ptr().cast());
            }

            #[test]
            fn given_field_precision_correct() {
                let field = ZfpField::new(&[] as &[$ty], [0]);
                assert_eq!(field.precision(), (size_of::<$ty>() * 8) as u32);
            }

            #[test]
            fn given_contiguous_data_field_size_bytes_correct() {
                let data = vec![<$ty>::default(); $nx * $ny * $nz * $nw];
                let field = ZfpField::new(&data, [$nx, $ny, $nz, $nw]);
                assert_eq!(field.size_bytes(), $nx * $ny * $nz * $nw * size_of::<$ty>());
            }

            #[test]
            fn given_noncontiguous_data_field_size_bytes_correct() {
                let n = ($sx) * ($nx - 1)
                    + ($sy) * ($ny - 1)
                    + ($sz) * ($nz - 1)
                    + ($sw) * ($nw - 1)
                    + 1;
                let data = vec![<$ty>::default(); n];
                let field = ZfpField::new_strided(
                    &data,
                    [$nx, $ny, $nz, $nw],
                    [$sx as isize, $sy as isize, $sz as isize, $sw as isize],
                );
                assert_eq!(field.size_bytes(), n * size_of::<$ty>());
            }
        }
    };
}

// ---------------------------------------------------------------------------
// 1-D double  (testZfpField1d.c)
// ---------------------------------------------------------------------------
field_tests!(mod = dim1_double, ty = f64, dims = 1, nx = 20, sx = 2);

// ---------------------------------------------------------------------------
// 1-D float   (testZfpField1f.c)
// ---------------------------------------------------------------------------
field_tests!(mod = dim1_float, ty = f32, dims = 1, nx = 20, sx = 2);

// ---------------------------------------------------------------------------
// 2-D double  (testZfpField2d.c)
// ---------------------------------------------------------------------------
field_tests!(mod = dim2_double, ty = f64, dims = 2, nx = 20, ny = 21, sx = 2, sy = 3);

// ---------------------------------------------------------------------------
// 2-D float   (testZfpField2f.c)
// ---------------------------------------------------------------------------
field_tests!(mod = dim2_float, ty = f32, dims = 2, nx = 20, ny = 21, sx = 2, sy = 3);

// ---------------------------------------------------------------------------
// 3-D double  (testZfpField3d.c)
// ---------------------------------------------------------------------------
field_tests!(mod = dim3_double, ty = f64, dims = 3, nx = 20, ny = 21, nz = 12, sx = 2, sy = 3, sz = 4);

// ---------------------------------------------------------------------------
// 3-D float   (testZfpField3f.c)
// ---------------------------------------------------------------------------
field_tests!(mod = dim3_float, ty = f32, dims = 3, nx = 20, ny = 21, nz = 12, sx = 2, sy = 3, sz = 4);

// ---------------------------------------------------------------------------
// 4-D double  (testZfpField4d.c)
// ---------------------------------------------------------------------------
field_tests!(mod = dim4_double, ty = f64, dims = 4, nx = 20, ny = 21, nz = 12, nw = 6, sx = 2, sy = 3, sz = 4, sw = 2);

// ---------------------------------------------------------------------------
// 4-D float   (testZfpField4f.c)
// ---------------------------------------------------------------------------
field_tests!(mod = dim4_float, ty = f32, dims = 4, nx = 20, ny = 21, nz = 12, nw = 6, sx = 2, sy = 3, sz = 4, sw = 2);
