#![expect(unsafe_op_in_unsafe_fn)]

use proptest::prelude::*;
use std::ffi::c_void;
use std::mem::size_of;
use zfp_rs_ffi as ffi;

#[allow(non_camel_case_types, non_upper_case_globals)]
mod zfp_sys {
    use super::{c_void, ffi};
    use libloading::Library;
    use std::path::{Path, PathBuf};
    use std::process::Command;
    use std::sync::OnceLock;

    pub type bitstream = c_void;
    pub type zfp_field = ffi::zfp_field;
    pub type zfp_stream = ffi::zfp_stream;
    pub type zfp_type = ffi::zfp_type;

    pub const zfp_type_zfp_type_int32: zfp_type = ffi::zfp_type_zfp_type_int32;
    pub const zfp_type_zfp_type_int64: zfp_type = ffi::zfp_type_zfp_type_int64;
    pub const zfp_type_zfp_type_float: zfp_type = ffi::zfp_type_zfp_type_float;
    pub const zfp_type_zfp_type_double: zfp_type = ffi::zfp_type_zfp_type_double;

    struct Api {
        _lib: Library,
        stream_close: unsafe extern "C" fn(*mut bitstream),
        stream_flush: unsafe extern "C" fn(*mut bitstream) -> usize,
        stream_open: unsafe extern "C" fn(*mut c_void, usize) -> *mut bitstream,
        stream_size: unsafe extern "C" fn(*const bitstream) -> usize,
        zfp_compress: unsafe extern "C" fn(*mut zfp_stream, *const zfp_field) -> usize,
        zfp_decompress: unsafe extern "C" fn(*mut zfp_stream, *mut zfp_field) -> usize,
        zfp_field_1d: unsafe extern "C" fn(*mut c_void, zfp_type, usize) -> *mut zfp_field,
        zfp_field_2d: unsafe extern "C" fn(*mut c_void, zfp_type, usize, usize) -> *mut zfp_field,
        zfp_field_3d:
            unsafe extern "C" fn(*mut c_void, zfp_type, usize, usize, usize) -> *mut zfp_field,
        zfp_field_4d: unsafe extern "C" fn(
            *mut c_void,
            zfp_type,
            usize,
            usize,
            usize,
            usize,
        ) -> *mut zfp_field,
        zfp_field_free: unsafe extern "C" fn(*mut zfp_field),
        zfp_field_set_stride_1d: unsafe extern "C" fn(*mut zfp_field, isize),
        zfp_field_set_stride_2d: unsafe extern "C" fn(*mut zfp_field, isize, isize),
        zfp_field_set_stride_3d: unsafe extern "C" fn(*mut zfp_field, isize, isize, isize),
        zfp_field_set_stride_4d: unsafe extern "C" fn(*mut zfp_field, isize, isize, isize, isize),
        zfp_stream_close: unsafe extern "C" fn(*mut zfp_stream),
        zfp_stream_mode: unsafe extern "C" fn(*const zfp_stream) -> u64,
        zfp_stream_open: unsafe extern "C" fn(*mut bitstream) -> *mut zfp_stream,
        zfp_stream_rewind: unsafe extern "C" fn(*mut zfp_stream),
        zfp_stream_set_accuracy: unsafe extern "C" fn(*mut zfp_stream, f64) -> f64,
        zfp_stream_set_bit_stream: unsafe extern "C" fn(*mut zfp_stream, *mut bitstream),
        zfp_stream_set_params: unsafe extern "C" fn(*mut zfp_stream, u32, u32, u32, i32) -> i32,
        zfp_stream_set_precision: unsafe extern "C" fn(*mut zfp_stream, u32) -> u32,
        zfp_stream_set_rate: unsafe extern "C" fn(*mut zfp_stream, f64, zfp_type, u32, i32) -> f64,
        zfp_stream_set_reversible: unsafe extern "C" fn(*mut zfp_stream),
        zfp_write_header: unsafe extern "C" fn(*mut zfp_stream, *const zfp_field, u32) -> usize,
        zfp_encode_block_int32_1: unsafe extern "C" fn(*mut zfp_stream, *const i32) -> usize,
        zfp_encode_block_int64_1: unsafe extern "C" fn(*mut zfp_stream, *const i64) -> usize,
        zfp_encode_block_float_1: unsafe extern "C" fn(*mut zfp_stream, *const f32) -> usize,
        zfp_encode_block_double_1: unsafe extern "C" fn(*mut zfp_stream, *const f64) -> usize,
        zfp_encode_block_int32_2: unsafe extern "C" fn(*mut zfp_stream, *const i32) -> usize,
        zfp_encode_block_int64_2: unsafe extern "C" fn(*mut zfp_stream, *const i64) -> usize,
        zfp_encode_block_float_2: unsafe extern "C" fn(*mut zfp_stream, *const f32) -> usize,
        zfp_encode_block_double_2: unsafe extern "C" fn(*mut zfp_stream, *const f64) -> usize,
        zfp_encode_block_int32_3: unsafe extern "C" fn(*mut zfp_stream, *const i32) -> usize,
        zfp_encode_block_int64_3: unsafe extern "C" fn(*mut zfp_stream, *const i64) -> usize,
        zfp_encode_block_float_3: unsafe extern "C" fn(*mut zfp_stream, *const f32) -> usize,
        zfp_encode_block_double_3: unsafe extern "C" fn(*mut zfp_stream, *const f64) -> usize,
        zfp_encode_block_int32_4: unsafe extern "C" fn(*mut zfp_stream, *const i32) -> usize,
        zfp_encode_block_int64_4: unsafe extern "C" fn(*mut zfp_stream, *const i64) -> usize,
        zfp_encode_block_float_4: unsafe extern "C" fn(*mut zfp_stream, *const f32) -> usize,
        zfp_encode_block_double_4: unsafe extern "C" fn(*mut zfp_stream, *const f64) -> usize,
        zfp_decode_partial_block_strided_int32_1:
            unsafe extern "C" fn(*mut zfp_stream, *mut i32, usize, isize) -> usize,
        zfp_decode_partial_block_strided_int64_1:
            unsafe extern "C" fn(*mut zfp_stream, *mut i64, usize, isize) -> usize,
        zfp_decode_partial_block_strided_float_1:
            unsafe extern "C" fn(*mut zfp_stream, *mut f32, usize, isize) -> usize,
        zfp_decode_partial_block_strided_double_1:
            unsafe extern "C" fn(*mut zfp_stream, *mut f64, usize, isize) -> usize,
        zfp_decode_partial_block_strided_int32_2:
            unsafe extern "C" fn(*mut zfp_stream, *mut i32, usize, usize, isize, isize) -> usize,
        zfp_decode_partial_block_strided_int64_2:
            unsafe extern "C" fn(*mut zfp_stream, *mut i64, usize, usize, isize, isize) -> usize,
        zfp_decode_partial_block_strided_float_2:
            unsafe extern "C" fn(*mut zfp_stream, *mut f32, usize, usize, isize, isize) -> usize,
        zfp_decode_partial_block_strided_double_2:
            unsafe extern "C" fn(*mut zfp_stream, *mut f64, usize, usize, isize, isize) -> usize,
        zfp_decode_partial_block_strided_int32_3: unsafe extern "C" fn(
            *mut zfp_stream,
            *mut i32,
            usize,
            usize,
            usize,
            isize,
            isize,
            isize,
        ) -> usize,
        zfp_decode_partial_block_strided_int64_3: unsafe extern "C" fn(
            *mut zfp_stream,
            *mut i64,
            usize,
            usize,
            usize,
            isize,
            isize,
            isize,
        ) -> usize,
        zfp_decode_partial_block_strided_float_3: unsafe extern "C" fn(
            *mut zfp_stream,
            *mut f32,
            usize,
            usize,
            usize,
            isize,
            isize,
            isize,
        ) -> usize,
        zfp_decode_partial_block_strided_double_3: unsafe extern "C" fn(
            *mut zfp_stream,
            *mut f64,
            usize,
            usize,
            usize,
            isize,
            isize,
            isize,
        ) -> usize,
        zfp_decode_partial_block_strided_int32_4: unsafe extern "C" fn(
            *mut zfp_stream,
            *mut i32,
            usize,
            usize,
            usize,
            usize,
            isize,
            isize,
            isize,
            isize,
        ) -> usize,
        zfp_decode_partial_block_strided_int64_4: unsafe extern "C" fn(
            *mut zfp_stream,
            *mut i64,
            usize,
            usize,
            usize,
            usize,
            isize,
            isize,
            isize,
            isize,
        ) -> usize,
        zfp_decode_partial_block_strided_float_4: unsafe extern "C" fn(
            *mut zfp_stream,
            *mut f32,
            usize,
            usize,
            usize,
            usize,
            isize,
            isize,
            isize,
            isize,
        ) -> usize,
        zfp_decode_partial_block_strided_double_4: unsafe extern "C" fn(
            *mut zfp_stream,
            *mut f64,
            usize,
            usize,
            usize,
            usize,
            isize,
            isize,
            isize,
            isize,
        ) -> usize,
    }

    static API: OnceLock<Api> = OnceLock::new();

    fn api() -> &'static Api {
        API.get_or_init(|| unsafe {
            let lib_path = build_shared_zfp();
            let lib = Library::new(&lib_path)
                .unwrap_or_else(|err| panic!("failed to load {}: {err}", lib_path.display()));
            Api {
                stream_close: sym(&lib, b"stream_close\0"),
                stream_flush: sym(&lib, b"stream_flush\0"),
                stream_open: sym(&lib, b"stream_open\0"),
                stream_size: sym(&lib, b"stream_size\0"),
                zfp_compress: sym(&lib, b"zfp_compress\0"),
                zfp_decompress: sym(&lib, b"zfp_decompress\0"),
                zfp_field_1d: sym(&lib, b"zfp_field_1d\0"),
                zfp_field_2d: sym(&lib, b"zfp_field_2d\0"),
                zfp_field_3d: sym(&lib, b"zfp_field_3d\0"),
                zfp_field_4d: sym(&lib, b"zfp_field_4d\0"),
                zfp_field_free: sym(&lib, b"zfp_field_free\0"),
                zfp_field_set_stride_1d: sym(&lib, b"zfp_field_set_stride_1d\0"),
                zfp_field_set_stride_2d: sym(&lib, b"zfp_field_set_stride_2d\0"),
                zfp_field_set_stride_3d: sym(&lib, b"zfp_field_set_stride_3d\0"),
                zfp_field_set_stride_4d: sym(&lib, b"zfp_field_set_stride_4d\0"),
                zfp_stream_close: sym(&lib, b"zfp_stream_close\0"),
                zfp_stream_mode: sym(&lib, b"zfp_stream_mode\0"),
                zfp_stream_open: sym(&lib, b"zfp_stream_open\0"),
                zfp_stream_rewind: sym(&lib, b"zfp_stream_rewind\0"),
                zfp_stream_set_accuracy: sym(&lib, b"zfp_stream_set_accuracy\0"),
                zfp_stream_set_bit_stream: sym(&lib, b"zfp_stream_set_bit_stream\0"),
                zfp_stream_set_params: sym(&lib, b"zfp_stream_set_params\0"),
                zfp_stream_set_precision: sym(&lib, b"zfp_stream_set_precision\0"),
                zfp_stream_set_rate: sym(&lib, b"zfp_stream_set_rate\0"),
                zfp_stream_set_reversible: sym(&lib, b"zfp_stream_set_reversible\0"),
                zfp_write_header: sym(&lib, b"zfp_write_header\0"),
                zfp_encode_block_int32_1: sym(&lib, b"zfp_encode_block_int32_1\0"),
                zfp_encode_block_int64_1: sym(&lib, b"zfp_encode_block_int64_1\0"),
                zfp_encode_block_float_1: sym(&lib, b"zfp_encode_block_float_1\0"),
                zfp_encode_block_double_1: sym(&lib, b"zfp_encode_block_double_1\0"),
                zfp_encode_block_int32_2: sym(&lib, b"zfp_encode_block_int32_2\0"),
                zfp_encode_block_int64_2: sym(&lib, b"zfp_encode_block_int64_2\0"),
                zfp_encode_block_float_2: sym(&lib, b"zfp_encode_block_float_2\0"),
                zfp_encode_block_double_2: sym(&lib, b"zfp_encode_block_double_2\0"),
                zfp_encode_block_int32_3: sym(&lib, b"zfp_encode_block_int32_3\0"),
                zfp_encode_block_int64_3: sym(&lib, b"zfp_encode_block_int64_3\0"),
                zfp_encode_block_float_3: sym(&lib, b"zfp_encode_block_float_3\0"),
                zfp_encode_block_double_3: sym(&lib, b"zfp_encode_block_double_3\0"),
                zfp_encode_block_int32_4: sym(&lib, b"zfp_encode_block_int32_4\0"),
                zfp_encode_block_int64_4: sym(&lib, b"zfp_encode_block_int64_4\0"),
                zfp_encode_block_float_4: sym(&lib, b"zfp_encode_block_float_4\0"),
                zfp_encode_block_double_4: sym(&lib, b"zfp_encode_block_double_4\0"),
                zfp_decode_partial_block_strided_int32_1: sym(
                    &lib,
                    b"zfp_decode_partial_block_strided_int32_1\0",
                ),
                zfp_decode_partial_block_strided_int64_1: sym(
                    &lib,
                    b"zfp_decode_partial_block_strided_int64_1\0",
                ),
                zfp_decode_partial_block_strided_float_1: sym(
                    &lib,
                    b"zfp_decode_partial_block_strided_float_1\0",
                ),
                zfp_decode_partial_block_strided_double_1: sym(
                    &lib,
                    b"zfp_decode_partial_block_strided_double_1\0",
                ),
                zfp_decode_partial_block_strided_int32_2: sym(
                    &lib,
                    b"zfp_decode_partial_block_strided_int32_2\0",
                ),
                zfp_decode_partial_block_strided_int64_2: sym(
                    &lib,
                    b"zfp_decode_partial_block_strided_int64_2\0",
                ),
                zfp_decode_partial_block_strided_float_2: sym(
                    &lib,
                    b"zfp_decode_partial_block_strided_float_2\0",
                ),
                zfp_decode_partial_block_strided_double_2: sym(
                    &lib,
                    b"zfp_decode_partial_block_strided_double_2\0",
                ),
                zfp_decode_partial_block_strided_int32_3: sym(
                    &lib,
                    b"zfp_decode_partial_block_strided_int32_3\0",
                ),
                zfp_decode_partial_block_strided_int64_3: sym(
                    &lib,
                    b"zfp_decode_partial_block_strided_int64_3\0",
                ),
                zfp_decode_partial_block_strided_float_3: sym(
                    &lib,
                    b"zfp_decode_partial_block_strided_float_3\0",
                ),
                zfp_decode_partial_block_strided_double_3: sym(
                    &lib,
                    b"zfp_decode_partial_block_strided_double_3\0",
                ),
                zfp_decode_partial_block_strided_int32_4: sym(
                    &lib,
                    b"zfp_decode_partial_block_strided_int32_4\0",
                ),
                zfp_decode_partial_block_strided_int64_4: sym(
                    &lib,
                    b"zfp_decode_partial_block_strided_int64_4\0",
                ),
                zfp_decode_partial_block_strided_float_4: sym(
                    &lib,
                    b"zfp_decode_partial_block_strided_float_4\0",
                ),
                zfp_decode_partial_block_strided_double_4: sym(
                    &lib,
                    b"zfp_decode_partial_block_strided_double_4\0",
                ),
                _lib: lib,
            }
        })
    }

    unsafe fn sym<T: Copy>(lib: &Library, name: &'static [u8]) -> T {
        *lib.get::<T>(name)
            .unwrap_or_else(|err| panic!("failed to load symbol {}: {err}", symbol_name(name)))
    }

    fn symbol_name(name: &'static [u8]) -> &'static str {
        std::str::from_utf8(name.strip_suffix(b"\0").unwrap_or(name)).unwrap_or("<invalid utf8>")
    }

    fn build_shared_zfp() -> PathBuf {
        let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
        let workspace = manifest_dir.parent().expect("workspace root");
        let build_dir = workspace.join("target").join("zfp-rs-ffi-c-ref-shared");
        run(
            Command::new("cmake")
                .arg("-S")
                .arg(workspace.join("zfp"))
                .arg("-B")
                .arg(&build_dir)
                .arg("-DBUILD_SHARED_LIBS=ON")
                .arg("-DBUILD_TESTING=OFF")
                .arg("-DBUILD_UTILITIES=OFF")
                .arg("-DBUILD_EXAMPLES=OFF")
                .arg("-DZFP_WITH_OPENMP=OFF"),
            "configuring shared upstream zfp",
        );
        run(
            Command::new("cmake")
                .arg("--build")
                .arg(&build_dir)
                .arg("--target")
                .arg("zfp"),
            "building shared upstream zfp",
        );
        shared_library_path(&build_dir)
    }

    fn run(command: &mut Command, description: &str) {
        let output = command
            .output()
            .unwrap_or_else(|err| panic!("failed {description}: {err}"));
        if !output.status.success() {
            panic!(
                "failed {description}\nstatus: {}\nstdout:\n{}\nstderr:\n{}",
                output.status,
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
        }
    }

    fn shared_library_path(build_dir: &Path) -> PathBuf {
        if cfg!(target_os = "macos") {
            build_dir.join("lib").join("libzfp.dylib")
        } else if cfg!(target_os = "windows") {
            build_dir.join("bin").join("zfp.dll")
        } else {
            build_dir.join("lib").join("libzfp.so")
        }
    }

    macro_rules! wrap {
        ($name:ident($($arg:ident: $ty:ty),*) -> $ret:ty) => {
            pub unsafe extern "C" fn $name($($arg: $ty),*) -> $ret {
                (api().$name)($($arg),*)
            }
        };
        ($name:ident($($arg:ident: $ty:ty),*)) => {
            pub unsafe extern "C" fn $name($($arg: $ty),*) {
                (api().$name)($($arg),*)
            }
        };
    }

    wrap!(stream_close(stream: *mut bitstream));
    wrap!(stream_flush(stream: *mut bitstream) -> usize);
    wrap!(stream_open(buffer: *mut c_void, bytes: usize) -> *mut bitstream);
    wrap!(stream_size(stream: *const bitstream) -> usize);
    wrap!(zfp_compress(stream: *mut zfp_stream, field: *const zfp_field) -> usize);
    wrap!(zfp_decompress(stream: *mut zfp_stream, field: *mut zfp_field) -> usize);
    wrap!(zfp_field_1d(data: *mut c_void, ty: zfp_type, nx: usize) -> *mut zfp_field);
    wrap!(zfp_field_2d(data: *mut c_void, ty: zfp_type, nx: usize, ny: usize) -> *mut zfp_field);
    wrap!(zfp_field_3d(data: *mut c_void, ty: zfp_type, nx: usize, ny: usize, nz: usize) -> *mut zfp_field);
    wrap!(zfp_field_4d(data: *mut c_void, ty: zfp_type, nx: usize, ny: usize, nz: usize, nw: usize) -> *mut zfp_field);
    wrap!(zfp_field_free(field: *mut zfp_field));
    wrap!(zfp_field_set_stride_1d(field: *mut zfp_field, sx: isize));
    wrap!(zfp_field_set_stride_2d(field: *mut zfp_field, sx: isize, sy: isize));
    wrap!(zfp_field_set_stride_3d(field: *mut zfp_field, sx: isize, sy: isize, sz: isize));
    wrap!(zfp_field_set_stride_4d(field: *mut zfp_field, sx: isize, sy: isize, sz: isize, sw: isize));
    wrap!(zfp_stream_close(stream: *mut zfp_stream));
    wrap!(zfp_stream_mode(stream: *const zfp_stream) -> u64);
    wrap!(zfp_stream_open(bs: *mut bitstream) -> *mut zfp_stream);
    wrap!(zfp_stream_rewind(stream: *mut zfp_stream));
    wrap!(zfp_stream_set_accuracy(stream: *mut zfp_stream, tolerance: f64) -> f64);
    wrap!(zfp_stream_set_bit_stream(stream: *mut zfp_stream, bs: *mut bitstream));
    wrap!(zfp_stream_set_params(stream: *mut zfp_stream, minbits: u32, maxbits: u32, maxprec: u32, minexp: i32) -> i32);
    wrap!(zfp_stream_set_precision(stream: *mut zfp_stream, precision: u32) -> u32);
    wrap!(zfp_stream_set_rate(stream: *mut zfp_stream, rate: f64, ty: zfp_type, dims: u32, align: i32) -> f64);
    wrap!(zfp_stream_set_reversible(stream: *mut zfp_stream));
    wrap!(zfp_write_header(stream: *mut zfp_stream, field: *const zfp_field, mask: u32) -> usize);
    wrap!(zfp_encode_block_int32_1(stream: *mut zfp_stream, block: *const i32) -> usize);
    wrap!(zfp_encode_block_int64_1(stream: *mut zfp_stream, block: *const i64) -> usize);
    wrap!(zfp_encode_block_float_1(stream: *mut zfp_stream, block: *const f32) -> usize);
    wrap!(zfp_encode_block_double_1(stream: *mut zfp_stream, block: *const f64) -> usize);
    wrap!(zfp_encode_block_int32_2(stream: *mut zfp_stream, block: *const i32) -> usize);
    wrap!(zfp_encode_block_int64_2(stream: *mut zfp_stream, block: *const i64) -> usize);
    wrap!(zfp_encode_block_float_2(stream: *mut zfp_stream, block: *const f32) -> usize);
    wrap!(zfp_encode_block_double_2(stream: *mut zfp_stream, block: *const f64) -> usize);
    wrap!(zfp_encode_block_int32_3(stream: *mut zfp_stream, block: *const i32) -> usize);
    wrap!(zfp_encode_block_int64_3(stream: *mut zfp_stream, block: *const i64) -> usize);
    wrap!(zfp_encode_block_float_3(stream: *mut zfp_stream, block: *const f32) -> usize);
    wrap!(zfp_encode_block_double_3(stream: *mut zfp_stream, block: *const f64) -> usize);
    wrap!(zfp_encode_block_int32_4(stream: *mut zfp_stream, block: *const i32) -> usize);
    wrap!(zfp_encode_block_int64_4(stream: *mut zfp_stream, block: *const i64) -> usize);
    wrap!(zfp_encode_block_float_4(stream: *mut zfp_stream, block: *const f32) -> usize);
    wrap!(zfp_encode_block_double_4(stream: *mut zfp_stream, block: *const f64) -> usize);
    wrap!(zfp_decode_partial_block_strided_int32_1(stream: *mut zfp_stream, p: *mut i32, nx: usize, sx: isize) -> usize);
    wrap!(zfp_decode_partial_block_strided_int64_1(stream: *mut zfp_stream, p: *mut i64, nx: usize, sx: isize) -> usize);
    wrap!(zfp_decode_partial_block_strided_float_1(stream: *mut zfp_stream, p: *mut f32, nx: usize, sx: isize) -> usize);
    wrap!(zfp_decode_partial_block_strided_double_1(stream: *mut zfp_stream, p: *mut f64, nx: usize, sx: isize) -> usize);
    wrap!(zfp_decode_partial_block_strided_int32_2(stream: *mut zfp_stream, p: *mut i32, nx: usize, ny: usize, sx: isize, sy: isize) -> usize);
    wrap!(zfp_decode_partial_block_strided_int64_2(stream: *mut zfp_stream, p: *mut i64, nx: usize, ny: usize, sx: isize, sy: isize) -> usize);
    wrap!(zfp_decode_partial_block_strided_float_2(stream: *mut zfp_stream, p: *mut f32, nx: usize, ny: usize, sx: isize, sy: isize) -> usize);
    wrap!(zfp_decode_partial_block_strided_double_2(stream: *mut zfp_stream, p: *mut f64, nx: usize, ny: usize, sx: isize, sy: isize) -> usize);
    wrap!(zfp_decode_partial_block_strided_int32_3(stream: *mut zfp_stream, p: *mut i32, nx: usize, ny: usize, nz: usize, sx: isize, sy: isize, sz: isize) -> usize);
    wrap!(zfp_decode_partial_block_strided_int64_3(stream: *mut zfp_stream, p: *mut i64, nx: usize, ny: usize, nz: usize, sx: isize, sy: isize, sz: isize) -> usize);
    wrap!(zfp_decode_partial_block_strided_float_3(stream: *mut zfp_stream, p: *mut f32, nx: usize, ny: usize, nz: usize, sx: isize, sy: isize, sz: isize) -> usize);
    wrap!(zfp_decode_partial_block_strided_double_3(stream: *mut zfp_stream, p: *mut f64, nx: usize, ny: usize, nz: usize, sx: isize, sy: isize, sz: isize) -> usize);
    wrap!(zfp_decode_partial_block_strided_int32_4(stream: *mut zfp_stream, p: *mut i32, nx: usize, ny: usize, nz: usize, nw: usize, sx: isize, sy: isize, sz: isize, sw: isize) -> usize);
    wrap!(zfp_decode_partial_block_strided_int64_4(stream: *mut zfp_stream, p: *mut i64, nx: usize, ny: usize, nz: usize, nw: usize, sx: isize, sy: isize, sz: isize, sw: isize) -> usize);
    wrap!(zfp_decode_partial_block_strided_float_4(stream: *mut zfp_stream, p: *mut f32, nx: usize, ny: usize, nz: usize, nw: usize, sx: isize, sy: isize, sz: isize, sw: isize) -> usize);
    wrap!(zfp_decode_partial_block_strided_double_4(stream: *mut zfp_stream, p: *mut f64, nx: usize, ny: usize, nz: usize, nw: usize, sx: isize, sy: isize, sz: isize, sw: isize) -> usize);
}

const CAPACITY: usize = 1 << 20;
const ZFP_MAX_PREC: u32 = 64;
const ZFP_MIN_EXP: i32 = -1074;
const ZFP_RATE_PARAM_BITS: u32 = 19;

#[derive(Clone, Debug)]
enum Mode {
    FixedRate(u32),
    FixedPrecision(u32),
    FixedAccuracy(i32),
    Reversible,
}

fn mode_strategy() -> impl Strategy<Value = Mode> {
    prop_oneof![
        (1u32..=2048).prop_map(Mode::FixedRate),
        (1u32..=64).prop_map(Mode::FixedPrecision),
        (-1074i32..=843).prop_map(Mode::FixedAccuracy),
        Just(Mode::Reversible),
    ]
}

fn normal_f32() -> impl Strategy<Value = f32> {
    any::<f32>().prop_filter("must be finite normal or zero", |v| {
        v.is_finite() && !v.is_subnormal()
    })
}

fn normal_f64() -> impl Strategy<Value = f64> {
    any::<f64>().prop_filter("must be finite normal or zero", |v| {
        v.is_finite() && !v.is_subnormal()
    })
}

struct CStream {
    zfp: *mut zfp_sys::zfp_stream,
    bs: *mut zfp_sys::bitstream,
    buf: Vec<u8>,
}

impl CStream {
    fn new() -> Self {
        let mut buf = vec![0u8; CAPACITY];
        let bs = unsafe { zfp_sys::stream_open(buf.as_mut_ptr().cast::<c_void>(), CAPACITY) };
        assert!(!bs.is_null());
        let zfp = unsafe { zfp_sys::zfp_stream_open(bs) };
        assert!(!zfp.is_null());
        Self { zfp, bs, buf }
    }

    fn with_bytes(bytes: &[u8]) -> Self {
        let mut stream = Self::new();
        stream.buf[..bytes.len()].copy_from_slice(bytes);
        unsafe {
            zfp_sys::stream_close(stream.bs);
            stream.bs = zfp_sys::stream_open(stream.buf.as_mut_ptr().cast::<c_void>(), CAPACITY);
            zfp_sys::zfp_stream_set_bit_stream(stream.zfp, stream.bs);
        }
        stream
    }

    fn flush(&mut self) {
        unsafe {
            zfp_sys::stream_flush(self.bs);
        }
    }

    fn rewind(&mut self) {
        unsafe {
            zfp_sys::zfp_stream_rewind(self.zfp);
        }
    }

    fn bytes(&self) -> Vec<u8> {
        let size = unsafe { zfp_sys::stream_size(self.bs) };
        self.buf[..size].to_vec()
    }
}

impl Drop for CStream {
    fn drop(&mut self) {
        unsafe {
            zfp_sys::zfp_stream_close(self.zfp);
            zfp_sys::stream_close(self.bs);
        }
    }
}

struct FfiStream {
    zfp: *mut ffi::zfp_stream,
    bs: *mut c_void,
    _buf: Option<Vec<u8>>,
}

impl FfiStream {
    fn new() -> Self {
        let bs = unsafe { ffi::stream_open(std::ptr::null_mut(), CAPACITY) };
        assert!(!bs.is_null());
        let zfp = unsafe { ffi::zfp_stream_open(bs) };
        assert!(!zfp.is_null());
        Self {
            zfp,
            bs,
            _buf: None,
        }
    }

    fn with_bytes(bytes: &[u8]) -> Self {
        let mut padded = bytes.to_vec();
        padded.extend_from_slice(&[0; 8]);
        let bs = unsafe { ffi::stream_open(padded.as_mut_ptr().cast::<c_void>(), padded.len()) };
        assert!(!bs.is_null());
        let zfp = unsafe { ffi::zfp_stream_open(bs) };
        assert!(!zfp.is_null());
        Self {
            zfp,
            bs,
            _buf: Some(padded),
        }
    }

    fn flush(&mut self) {
        unsafe {
            ffi::zfp_stream_flush(self.zfp);
        }
    }

    fn rewind(&mut self) {
        unsafe {
            ffi::zfp_stream_rewind(self.zfp);
        }
    }

    fn bytes(&self) -> Vec<u8> {
        unsafe {
            let size = ffi::stream_size(self.bs);
            let ptr = ffi::stream_data(self.bs);
            std::slice::from_raw_parts(ptr as *const u8, size).to_vec()
        }
    }
}

impl Drop for FfiStream {
    fn drop(&mut self) {
        unsafe {
            ffi::zfp_stream_close(self.zfp);
            ffi::stream_close(self.bs);
        }
    }
}

fn ffi_type(ty: zfp_sys::zfp_type) -> ffi::zfp_type {
    match ty {
        zfp_sys::zfp_type_zfp_type_int32 => ffi::zfp_type_zfp_type_int32,
        zfp_sys::zfp_type_zfp_type_int64 => ffi::zfp_type_zfp_type_int64,
        zfp_sys::zfp_type_zfp_type_float => ffi::zfp_type_zfp_type_float,
        zfp_sys::zfp_type_zfp_type_double => ffi::zfp_type_zfp_type_double,
        _ => ffi::zfp_type_zfp_type_none,
    }
}

unsafe fn apply_mode_c(
    zfp: *mut zfp_sys::zfp_stream,
    mode: &Mode,
    ty: zfp_sys::zfp_type,
    dims: u32,
) {
    match *mode {
        Mode::FixedRate(bits) => {
            zfp_sys::zfp_stream_set_rate(
                zfp,
                f64::from(bits) / f64::from(1u32 << (2 * dims)),
                ty,
                dims,
                0,
            );
        }
        Mode::FixedPrecision(precision) => {
            zfp_sys::zfp_stream_set_precision(zfp, precision);
        }
        Mode::FixedAccuracy(exp) => {
            zfp_sys::zfp_stream_set_accuracy(zfp, 2.0f64.powi(exp));
        }
        Mode::Reversible => {
            zfp_sys::zfp_stream_set_reversible(zfp);
        }
    }
}

unsafe fn apply_mode_ffi(zfp: *mut ffi::zfp_stream, mode: &Mode, ty: ffi::zfp_type, dims: u32) {
    match *mode {
        Mode::FixedRate(bits) => {
            ffi::zfp_stream_set_rate(
                zfp,
                f64::from(bits) / f64::from(1u32 << (2 * dims)),
                ty,
                dims,
                0,
            );
        }
        Mode::FixedPrecision(precision) => {
            ffi::zfp_stream_set_precision(zfp, precision);
        }
        Mode::FixedAccuracy(exp) => {
            ffi::zfp_stream_set_accuracy(zfp, 2.0f64.powi(exp));
        }
        Mode::Reversible => {
            ffi::zfp_stream_set_reversible(zfp);
        }
    }
}

unsafe fn c_field(
    data: *mut c_void,
    ty: zfp_sys::zfp_type,
    dims: &[usize],
) -> *mut zfp_sys::zfp_field {
    match dims {
        [nx] => zfp_sys::zfp_field_1d(data, ty, *nx),
        [nx, ny] => zfp_sys::zfp_field_2d(data, ty, *nx, *ny),
        [nx, ny, nz] => zfp_sys::zfp_field_3d(data, ty, *nx, *ny, *nz),
        [nx, ny, nz, nw] => zfp_sys::zfp_field_4d(data, ty, *nx, *ny, *nz, *nw),
        _ => std::ptr::null_mut(),
    }
}

fn ffi_field_from(data: *mut c_void, ty: ffi::zfp_type, dims: &[usize]) -> *mut ffi::zfp_field {
    match dims {
        [nx] => unsafe { ffi::zfp_field_1d(data, ty, *nx) },
        [nx, ny] => unsafe { ffi::zfp_field_2d(data, ty, *nx, *ny) },
        [nx, ny, nz] => unsafe { ffi::zfp_field_3d(data, ty, *nx, *ny, *nz) },
        [nx, ny, nz, nw] => unsafe { ffi::zfp_field_4d(data, ty, *nx, *ny, *nz, *nw) },
        _ => std::ptr::null_mut(),
    }
}

unsafe fn set_c_stride(field: *mut zfp_sys::zfp_field, strides: &[isize]) {
    match strides {
        [sx] => zfp_sys::zfp_field_set_stride_1d(field, *sx),
        [sx, sy] => zfp_sys::zfp_field_set_stride_2d(field, *sx, *sy),
        [sx, sy, sz] => zfp_sys::zfp_field_set_stride_3d(field, *sx, *sy, *sz),
        [sx, sy, sz, sw] => zfp_sys::zfp_field_set_stride_4d(field, *sx, *sy, *sz, *sw),
        _ => {}
    }
}

unsafe fn set_ffi_stride(field: *mut ffi::zfp_field, strides: &[isize]) {
    match strides {
        [sx] => ffi::zfp_field_set_stride_1d(field, *sx),
        [sx, sy] => ffi::zfp_field_set_stride_2d(field, *sx, *sy),
        [sx, sy, sz] => ffi::zfp_field_set_stride_3d(field, *sx, *sy, *sz),
        [sx, sy, sz, sw] => ffi::zfp_field_set_stride_4d(field, *sx, *sy, *sz, *sw),
        _ => {}
    }
}

fn dims_for(rank: u32, n: usize) -> Vec<usize> {
    match rank {
        1 => vec![n],
        2 => vec![n, 1],
        3 => vec![n, 1, 1],
        4 => vec![n, 1, 1, 1],
        _ => unreachable!(),
    }
}

#[test]
fn stream_open_with_buffer_writes_directly_to_caller_storage() {
    let mut words = vec![0u64; 4];
    let bs = unsafe {
        ffi::stream_open(
            words.as_mut_ptr().cast::<c_void>(),
            words.len() * size_of::<u64>(),
        )
    };
    assert!(!bs.is_null());

    unsafe {
        assert_eq!(ffi::stream_data(bs), words.as_mut_ptr().cast::<c_void>());
        ffi::stream_write_bits(bs, 0x0123_4567_89ab_cdef, 64);
        ffi::stream_close(bs);
    }

    assert_eq!(words[0], 0x0123_4567_89ab_cdef);
}

#[cfg(feature = "rayon")]
#[test]
fn borrowed_stream_omp_compress_writes_parallel_output_to_caller_storage() {
    let data: Vec<f32> = (0..17).map(|i| i as f32 * 0.25).collect();
    let owned = FfiStream::new();
    let mut borrowed_words = vec![0u64; 128];
    let borrowed_bs = unsafe {
        ffi::stream_open(
            borrowed_words.as_mut_ptr().cast::<c_void>(),
            borrowed_words.len() * size_of::<u64>(),
        )
    };
    assert!(!borrowed_bs.is_null());
    let borrowed_zfp = unsafe { ffi::zfp_stream_open(borrowed_bs) };
    assert!(!borrowed_zfp.is_null());

    let owned_field = ffi_field_from(
        data.as_ptr().cast_mut().cast::<c_void>(),
        ffi::zfp_type_zfp_type_float,
        &[data.len()],
    );
    let borrowed_field = ffi_field_from(
        data.as_ptr().cast_mut().cast::<c_void>(),
        ffi::zfp_type_zfp_type_float,
        &[data.len()],
    );
    assert!(!owned_field.is_null());
    assert!(!borrowed_field.is_null());

    unsafe {
        ffi::zfp_stream_set_rate(
            owned.zfp,
            5.0,
            ffi::zfp_type_zfp_type_float,
            1,
            ffi::zfp_false,
        );
        ffi::zfp_stream_set_rate(
            borrowed_zfp,
            5.0,
            ffi::zfp_type_zfp_type_float,
            1,
            ffi::zfp_false,
        );
        assert_eq!(ffi::zfp_stream_set_omp_threads(owned.zfp, 2), ffi::zfp_true);
        assert_eq!(
            ffi::zfp_stream_set_omp_threads(borrowed_zfp, 2),
            ffi::zfp_true
        );
        assert_eq!(
            ffi::zfp_stream_set_omp_chunk_size(owned.zfp, 2),
            ffi::zfp_true
        );
        assert_eq!(
            ffi::zfp_stream_set_omp_chunk_size(borrowed_zfp, 2),
            ffi::zfp_true
        );

        let owned_size = ffi::zfp_compress(owned.zfp, owned_field);
        let borrowed_size = ffi::zfp_compress(borrowed_zfp, borrowed_field);
        assert_eq!(borrowed_size, owned_size);
        assert_eq!(
            bytemuck::cast_slice::<u64, u8>(&borrowed_words)[..borrowed_size],
            owned.bytes()[..owned_size]
        );

        ffi::zfp_field_free(owned_field);
        ffi::zfp_field_free(borrowed_field);
        ffi::zfp_stream_close(borrowed_zfp);
        ffi::stream_close(borrowed_bs);
    }
}

#[test]
fn stream_copy_reads_from_live_source_without_reparsing_bytes() {
    let mut src_words = vec![0u64; 4];
    let mut dst_words = vec![0u64; 4];
    let src = unsafe {
        ffi::stream_open(
            src_words.as_mut_ptr().cast::<c_void>(),
            src_words.len() * size_of::<u64>(),
        )
    };
    let dst = unsafe {
        ffi::stream_open(
            dst_words.as_mut_ptr().cast::<c_void>(),
            dst_words.len() * size_of::<u64>(),
        )
    };
    assert!(!src.is_null());
    assert!(!dst.is_null());

    unsafe {
        ffi::stream_write_bits(src, 0xabcd, 16);
        ffi::stream_flush(src);
        ffi::stream_rewind(src);
        ffi::stream_copy(dst, src, 16);
        ffi::stream_flush(dst);
        ffi::stream_close(src);
        ffi::stream_close(dst);
    }

    assert_eq!(dst_words[0] & 0xffff, 0xabcd);
}

fn strided_dims_for(rank: u32) -> (Vec<usize>, Vec<isize>) {
    match rank {
        1 => (vec![6], vec![2]),
        2 => (vec![3, 4], vec![2, 8]),
        3 => (vec![2, 3, 3], vec![2, 5, 16]),
        4 => (vec![2, 2, 2, 2], vec![2, 5, 12, 28]),
        _ => unreachable!(),
    }
}

macro_rules! prop_header_and_full_field_compat {
    ($module:ident, $scalar:ty, $strategy:expr, $c_type:expr) => {
        mod $module {
            use super::*;

            proptest! {
                #[test]
                fn write_header_matches_zfp_sys(
                    rank in 1u32..=4,
                    len in 1usize..=32,
                    mask_bits in 0u32..=7,
                    mode in mode_strategy(),
                    data in prop::collection::vec($strategy, 1..=32),
                ) {
                    let len = len.min(data.len());
                    let dims = dims_for(rank, len);
                    let data = &data[..len];

                    let mut c = CStream::new();
                    let mut ffi = FfiStream::new();
                    unsafe {
                        apply_mode_c(c.zfp, &mode, $c_type, rank);
                        apply_mode_ffi(ffi.zfp, &mode, ffi_type($c_type), rank);

                        let c_field = c_field(data.as_ptr() as *mut c_void, $c_type, &dims);
                        let ffi_field = ffi_field_from(data.as_ptr() as *mut c_void, ffi_type($c_type), &dims);
                        let c_bits = zfp_sys::zfp_write_header(c.zfp, c_field, mask_bits);
                        let ffi_bits = ffi::zfp_write_header(ffi.zfp, ffi_field, mask_bits);
                        zfp_sys::zfp_field_free(c_field);
                        ffi::zfp_field_free(ffi_field);

                        c.flush();
                        ffi.flush();
                        prop_assert_eq!(ffi_bits, c_bits as usize);
                        prop_assert_eq!(ffi.bytes(), c.bytes());
                    }
                }

                #[test]
                fn read_header_accepts_zfp_sys_headers(
                    rank in 1u32..=4,
                    len in 1usize..=32,
                    mask_bits in 0u32..=7,
                    mode in mode_strategy(),
                    data in prop::collection::vec($strategy, 1..=32),
                ) {
                    let len = len.min(data.len());
                    let dims = dims_for(rank, len);
                    let data = &data[..len];

                    let mut c = CStream::new();
                    unsafe {
                        apply_mode_c(c.zfp, &mode, $c_type, rank);
                        let c_field = c_field(data.as_ptr() as *mut c_void, $c_type, &dims);
                        zfp_sys::zfp_write_header(c.zfp, c_field, mask_bits);
                        zfp_sys::zfp_field_free(c_field);
                        c.flush();
                    }

                    let ffi = FfiStream::with_bytes(&c.bytes());
                    unsafe {
                        let ffi_field = ffi::zfp_field_alloc();
                        let bits = ffi::zfp_read_header(ffi.zfp, ffi_field, mask_bits);
                        ffi::zfp_field_free(ffi_field);

                        let expected_bits = if mask_bits & ffi::ZFP_HEADER_MAGIC != 0 {
                            ffi::ZFP_MAGIC_BITS
                        } else {
                            0
                        } + if mask_bits & ffi::ZFP_HEADER_META != 0 {
                            ffi::ZFP_META_BITS
                        } else {
                            0
                        } + if mask_bits & ffi::ZFP_HEADER_MODE != 0 {
                            let mode_bits = zfp_sys::zfp_stream_mode(c.zfp);
                            if mode_bits > u64::from(ffi::ZFP_MODE_SHORT_MAX) {
                                ffi::ZFP_MODE_LONG_BITS
                            } else {
                                ffi::ZFP_MODE_SHORT_BITS
                            }
                        } else {
                            0
                        };
                        prop_assert_eq!(bits, expected_bits as usize);
                    }
                }

                #[test]
                fn compress_matches_zfp_sys_and_cross_decompresses(
                    rank in 1u32..=4,
                    len in 1usize..=32,
                    mode in mode_strategy(),
                    data in prop::collection::vec($strategy, 1..=32),
                ) {
                    let len = len.min(data.len());
                    let dims = dims_for(rank, len);
                    let data = &data[..len];

                    let mut c = CStream::new();
                    let mut ffi = FfiStream::new();
                    unsafe {
                        apply_mode_c(c.zfp, &mode, $c_type, rank);
                        apply_mode_ffi(ffi.zfp, &mode, ffi_type($c_type), rank);

                        let c_field = c_field(data.as_ptr() as *mut c_void, $c_type, &dims);
                        let ffi_field = ffi_field_from(data.as_ptr() as *mut c_void, ffi_type($c_type), &dims);
                        prop_assert!(zfp_sys::zfp_compress(c.zfp, c_field) > 0);
                        prop_assert!(ffi::zfp_compress(ffi.zfp, ffi_field) > 0);
                        zfp_sys::zfp_field_free(c_field);
                        ffi::zfp_field_free(ffi_field);
                        c.flush();
                        ffi.flush();
                    }

                    let c_bytes = c.bytes();
                    let ffi_bytes = ffi.bytes();
                    prop_assert_eq!(&ffi_bytes, &c_bytes);

                    let ffi_decode = FfiStream::with_bytes(&c_bytes);
                    let c_decode = CStream::with_bytes(&ffi_bytes);
                    let mut ffi_out = vec![<$scalar>::default(); len];
                    let mut c_out = vec![<$scalar>::default(); len];
                    unsafe {
                        apply_mode_ffi(ffi_decode.zfp, &mode, ffi_type($c_type), rank);
                        apply_mode_c(c_decode.zfp, &mode, $c_type, rank);
                        let ffi_field = ffi_field_from(ffi_out.as_mut_ptr().cast::<c_void>(), ffi_type($c_type), &dims);
                        let c_field = c_field(c_out.as_mut_ptr().cast::<c_void>(), $c_type, &dims);
                        prop_assert!(ffi::zfp_decompress(ffi_decode.zfp, ffi_field) > 0);
                        prop_assert!(zfp_sys::zfp_decompress(c_decode.zfp, c_field) > 0);
                        ffi::zfp_field_free(ffi_field);
                        zfp_sys::zfp_field_free(c_field);
                    }

                    prop_assert_eq!(
                        bytemuck::cast_slice::<$scalar, u8>(&ffi_out),
                        bytemuck::cast_slice::<$scalar, u8>(&c_out),
                    );
                }

                #[test]
                fn strided_compress_matches_zfp_sys(
                    rank in 1u32..=4,
                    mode in mode_strategy(),
                    data in prop::collection::vec($strategy, 64..=64),
                ) {
                    let (dims, strides) = strided_dims_for(rank);
                    let mut c = CStream::new();
                    let mut ffi = FfiStream::new();
                    unsafe {
                        apply_mode_c(c.zfp, &mode, $c_type, rank);
                        apply_mode_ffi(ffi.zfp, &mode, ffi_type($c_type), rank);

                        let c_field = c_field(data.as_ptr() as *mut c_void, $c_type, &dims);
                        let ffi_field = ffi_field_from(data.as_ptr() as *mut c_void, ffi_type($c_type), &dims);
                        set_c_stride(c_field, &strides);
                        set_ffi_stride(ffi_field, &strides);

                        prop_assert!(zfp_sys::zfp_compress(c.zfp, c_field) > 0);
                        prop_assert!(ffi::zfp_compress(ffi.zfp, ffi_field) > 0);
                        zfp_sys::zfp_field_free(c_field);
                        ffi::zfp_field_free(ffi_field);
                        c.flush();
                        ffi.flush();
                    }

                    prop_assert_eq!(ffi.bytes(), c.bytes());
                }
            }
        }
    };
}

prop_header_and_full_field_compat!(
    i32_compat,
    i32,
    any::<i32>(),
    zfp_sys::zfp_type_zfp_type_int32
);
prop_header_and_full_field_compat!(
    i64_compat,
    i64,
    any::<i64>(),
    zfp_sys::zfp_type_zfp_type_int64
);
prop_header_and_full_field_compat!(
    f32_compat,
    f32,
    normal_f32(),
    zfp_sys::zfp_type_zfp_type_float
);
prop_header_and_full_field_compat!(
    f64_compat,
    f64,
    normal_f64(),
    zfp_sys::zfp_type_zfp_type_double
);

#[test]
fn field_alloc_preserves_c_untyped_sentinel() {
    unsafe {
        let field = ffi::zfp_field_alloc();
        assert!(!field.is_null());
        assert_eq!(ffi::zfp_field_type(field), ffi::zfp_type_zfp_type_none);
        assert_eq!(ffi::zfp_field_precision(field), 0);
        assert_eq!(ffi::zfp_field_size_bytes(field), 0);
        assert_eq!(ffi::zfp_field_metadata(field), u64::MAX);
        ffi::zfp_field_free(field);
    }
}

#[test]
fn field_set_type_rejects_none_without_installing_it_as_usable_type() {
    unsafe {
        let field = ffi::zfp_field_1d(std::ptr::null_mut(), ffi::zfp_type_zfp_type_float, 4);
        assert!(!field.is_null());
        let actual = ffi::zfp_field_set_type(field, ffi::zfp_type_zfp_type_none);
        assert_eq!(actual, ffi::zfp_type_zfp_type_none);
        assert_eq!(ffi::zfp_field_type(field), ffi::zfp_type_zfp_type_float);
        assert_eq!(ffi::zfp_field_precision(field), 32);
        ffi::zfp_field_free(field);
    }
}

#[test]
fn untyped_allocated_field_rejects_header_write_and_compress() {
    let data = [1.0f32, 2.0, 3.0, 4.0];
    let ffi = FfiStream::new();
    unsafe {
        ffi::zfp_stream_set_precision(ffi.zfp, 8);
        let field = ffi::zfp_field_alloc();
        assert!(!field.is_null());
        ffi::zfp_field_set_pointer(field, data.as_ptr().cast::<c_void>().cast_mut());
        ffi::zfp_field_set_size_1d(field, data.len());

        assert_eq!(
            ffi::zfp_write_header(ffi.zfp, field, ffi::ZFP_HEADER_META),
            0
        );
        assert_eq!(ffi::zfp_compress(ffi.zfp, field), 0);
        ffi::zfp_field_free(field);
    }
}

#[test]
fn read_header_populates_allocated_field_metadata() {
    let data = [1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];
    let dims = [2usize, 3usize];
    let mut writer = FfiStream::new();
    unsafe {
        ffi::zfp_stream_set_precision(writer.zfp, 16);
        let src = ffi::zfp_field_2d(
            data.as_ptr().cast::<c_void>().cast_mut(),
            ffi::zfp_type_zfp_type_double,
            dims[0],
            dims[1],
        );
        assert_eq!(
            ffi::zfp_write_header(writer.zfp, src, ffi::ZFP_HEADER_META),
            ffi::ZFP_META_BITS as usize
        );
        ffi::zfp_field_free(src);
        writer.flush();
    }

    let reader = FfiStream::with_bytes(&writer.bytes());
    unsafe {
        let dst = ffi::zfp_field_alloc();
        assert_eq!(
            ffi::zfp_read_header(reader.zfp, dst, ffi::ZFP_HEADER_META),
            ffi::ZFP_META_BITS as usize
        );
        assert_eq!(ffi::zfp_field_type(dst), ffi::zfp_type_zfp_type_double);
        let mut decoded_dims = [0usize; 4];
        assert_eq!(
            ffi::zfp_field_size(dst, decoded_dims.as_mut_ptr()),
            data.len()
        );
        assert_eq!(decoded_dims, [dims[0], dims[1], 0, 0]);
        ffi::zfp_field_free(dst);
    }
}

#[test]
fn read_header_rejects_bad_magic_through_ffi_wrapper() {
    let mut ffi = FfiStream::new();
    unsafe {
        ffi::stream_write_bits(ffi.bs, u64::from(b'n'), 8);
        ffi::stream_write_bits(ffi.bs, u64::from(b'o'), 8);
        ffi::stream_write_bits(ffi.bs, u64::from(b'p'), 8);
        ffi::stream_write_bits(ffi.bs, u64::from(ffi::ZFP_CODEC_VERSION), 8);
        ffi.flush();
        ffi.rewind();

        let field = ffi::zfp_field_alloc();
        let bits = ffi::zfp_read_header(ffi.zfp, field, ffi::ZFP_HEADER_MAGIC);
        ffi::zfp_field_free(field);
        assert_eq!(bits, 0);
    }
}

fn invalid_long_mode_bits() -> u64 {
    let min_bits = 1u64;
    let max_bits = 0u64;
    let max_prec = 63u64;
    let min_exp = (ZFP_MIN_EXP + 16495) as u64;
    let mut mode = 0u64;
    mode = (mode << 15) | min_exp;
    mode = (mode << 7) | max_prec;
    mode = (mode << 15) | max_bits;
    mode = (mode << 15) | min_bits;
    (mode << 12) | 0xfff
}

#[test]
fn field_set_metadata_rejects_invalid_metadata_through_ffi_wrapper() {
    unsafe {
        let field = ffi::zfp_field_alloc();
        let ok = ffi::zfp_field_set_metadata(field, 1u64 << (ffi::ZFP_META_BITS + 1));
        ffi::zfp_field_free(field);
        assert_eq!(ok, ffi::zfp_false);
    }
}

#[test]
fn read_header_rejects_invalid_mode_through_ffi_wrapper() {
    let mut ffi = FfiStream::new();
    unsafe {
        ffi::stream_write_bits(
            ffi.bs,
            invalid_long_mode_bits(),
            ffi::ZFP_MODE_LONG_BITS as usize,
        );
        ffi.flush();
        ffi.rewind();

        let field = ffi::zfp_field_alloc();
        let bits = ffi::zfp_read_header(ffi.zfp, field, ffi::ZFP_HEADER_MODE);
        ffi::zfp_field_free(field);
        assert_eq!(bits, 0);
    }
}

fn partial_footprint(sizes: &[u32], strides: &[isize]) -> usize {
    1 + sizes
        .iter()
        .zip(strides)
        .map(|(size, stride)| (*size as usize - 1) * stride.cast_unsigned())
        .sum::<usize>()
}

fn ffi_stream_from_c_encoded<T>(
    data: &[T],
    block_len: usize,
    encode: unsafe extern "C" fn(*mut zfp_sys::zfp_stream, *const T) -> usize,
) -> (CStream, FfiStream) {
    #[allow(clippy::cast_possible_truncation)]
    // Test: block_len is 16/64/256 for 1D/3D/4D blocks, fits in u32.
    let maxbits = (block_len as u32) * ZFP_RATE_PARAM_BITS;
    let mut c = CStream::new();
    unsafe {
        zfp_sys::zfp_stream_set_params(c.zfp, maxbits, maxbits, ZFP_MAX_PREC, ZFP_MIN_EXP);
        encode(c.zfp, data.as_ptr());
        c.flush();
        let ffi = FfiStream::with_bytes(&c.bytes());
        ffi::zfp_stream_set_params(ffi.zfp, maxbits, maxbits, ZFP_MAX_PREC, ZFP_MIN_EXP);
        c.rewind();
        (c, ffi)
    }
}

macro_rules! partial_decode_1d {
    ($test_name:ident, $scalar:ty, $strategy:expr, $encode:path, $c_decode:path, $ffi_decode:path) => {
        proptest! {
            #[test]
            fn $test_name(data in prop::collection::vec($strategy, 4..5)) {
                let (c, ffi) = ffi_stream_from_c_encoded(&data, 4, $encode);
                let mut c_out = vec![<$scalar>::default(); partial_footprint(&[3], &[2])];
                let mut ffi_out = vec![<$scalar>::default(); c_out.len()];
                unsafe {
                    let c_bits = $c_decode(c.zfp, c_out.as_mut_ptr(), 3, 2);
                    let ffi_bits = $ffi_decode(ffi.zfp, ffi_out.as_mut_ptr(), 3, 2);
                    prop_assert_eq!(ffi_bits, c_bits as usize);
                }
                prop_assert_eq!(
                    bytemuck::cast_slice::<$scalar, u8>(&ffi_out),
                    bytemuck::cast_slice::<$scalar, u8>(&c_out),
                );
            }
        }
    };
}

macro_rules! partial_decode_2d {
    ($test_name:ident, $scalar:ty, $strategy:expr, $encode:path, $c_decode:path, $ffi_decode:path) => {
        proptest! {
            #[test]
            fn $test_name(data in prop::collection::vec($strategy, 16..17)) {
                let (c, ffi) = ffi_stream_from_c_encoded(&data, 16, $encode);
                let mut c_out = vec![<$scalar>::default(); partial_footprint(&[3, 2], &[1, 5])];
                let mut ffi_out = vec![<$scalar>::default(); c_out.len()];
                unsafe {
                    let c_bits = $c_decode(c.zfp, c_out.as_mut_ptr(), 3, 2, 1, 5);
                    let ffi_bits = $ffi_decode(ffi.zfp, ffi_out.as_mut_ptr(), 3, 2, 1, 5);
                    prop_assert_eq!(ffi_bits, c_bits as usize);
                }
                prop_assert_eq!(
                    bytemuck::cast_slice::<$scalar, u8>(&ffi_out),
                    bytemuck::cast_slice::<$scalar, u8>(&c_out),
                );
            }
        }
    };
}

macro_rules! partial_decode_3d {
    ($test_name:ident, $scalar:ty, $strategy:expr, $encode:path, $c_decode:path, $ffi_decode:path) => {
        proptest! {
            #[test]
            fn $test_name(data in prop::collection::vec($strategy, 64..65)) {
                let (c, ffi) = ffi_stream_from_c_encoded(&data, 64, $encode);
                let mut c_out = vec![<$scalar>::default(); partial_footprint(&[2, 3, 2], &[1, 3, 10])];
                let mut ffi_out = vec![<$scalar>::default(); c_out.len()];
                unsafe {
                    let c_bits = $c_decode(c.zfp, c_out.as_mut_ptr(), 2, 3, 2, 1, 3, 10);
                    let ffi_bits = $ffi_decode(ffi.zfp, ffi_out.as_mut_ptr(), 2, 3, 2, 1, 3, 10);
                    prop_assert_eq!(ffi_bits, c_bits as usize);
                }
                prop_assert_eq!(
                    bytemuck::cast_slice::<$scalar, u8>(&ffi_out),
                    bytemuck::cast_slice::<$scalar, u8>(&c_out),
                );
            }
        }
    };
}

macro_rules! partial_decode_4d {
    ($test_name:ident, $scalar:ty, $strategy:expr, $encode:path, $c_decode:path, $ffi_decode:path) => {
        proptest! {
            #[test]
            fn $test_name(data in prop::collection::vec($strategy, 256..257)) {
                let (c, ffi) = ffi_stream_from_c_encoded(&data, 256, $encode);
                let mut c_out = vec![<$scalar>::default(); partial_footprint(&[2, 2, 2, 2], &[1, 3, 8, 20])];
                let mut ffi_out = vec![<$scalar>::default(); c_out.len()];
                unsafe {
                    let c_bits = $c_decode(c.zfp, c_out.as_mut_ptr(), 2, 2, 2, 2, 1, 3, 8, 20);
                    let ffi_bits =
                        $ffi_decode(ffi.zfp, ffi_out.as_mut_ptr(), 2, 2, 2, 2, 1, 3, 8, 20);
                    prop_assert_eq!(ffi_bits, c_bits as usize);
                }
                prop_assert_eq!(
                    bytemuck::cast_slice::<$scalar, u8>(&ffi_out),
                    bytemuck::cast_slice::<$scalar, u8>(&c_out),
                );
            }
        }
    };
}

partial_decode_1d!(
    partial_decode_strided_i32_1d,
    i32,
    any::<i32>(),
    zfp_sys::zfp_encode_block_int32_1,
    zfp_sys::zfp_decode_partial_block_strided_int32_1,
    ffi::zfp_decode_partial_block_strided_int32_1
);
partial_decode_1d!(
    partial_decode_strided_i64_1d,
    i64,
    any::<i64>(),
    zfp_sys::zfp_encode_block_int64_1,
    zfp_sys::zfp_decode_partial_block_strided_int64_1,
    ffi::zfp_decode_partial_block_strided_int64_1
);
partial_decode_1d!(
    partial_decode_strided_f32_1d,
    f32,
    normal_f32(),
    zfp_sys::zfp_encode_block_float_1,
    zfp_sys::zfp_decode_partial_block_strided_float_1,
    ffi::zfp_decode_partial_block_strided_float_1
);
partial_decode_1d!(
    partial_decode_strided_f64_1d,
    f64,
    normal_f64(),
    zfp_sys::zfp_encode_block_double_1,
    zfp_sys::zfp_decode_partial_block_strided_double_1,
    ffi::zfp_decode_partial_block_strided_double_1
);

partial_decode_2d!(
    partial_decode_strided_i32_2d,
    i32,
    any::<i32>(),
    zfp_sys::zfp_encode_block_int32_2,
    zfp_sys::zfp_decode_partial_block_strided_int32_2,
    ffi::zfp_decode_partial_block_strided_int32_2
);
partial_decode_2d!(
    partial_decode_strided_i64_2d,
    i64,
    any::<i64>(),
    zfp_sys::zfp_encode_block_int64_2,
    zfp_sys::zfp_decode_partial_block_strided_int64_2,
    ffi::zfp_decode_partial_block_strided_int64_2
);
partial_decode_2d!(
    partial_decode_strided_f32_2d,
    f32,
    normal_f32(),
    zfp_sys::zfp_encode_block_float_2,
    zfp_sys::zfp_decode_partial_block_strided_float_2,
    ffi::zfp_decode_partial_block_strided_float_2
);
partial_decode_2d!(
    partial_decode_strided_f64_2d,
    f64,
    normal_f64(),
    zfp_sys::zfp_encode_block_double_2,
    zfp_sys::zfp_decode_partial_block_strided_double_2,
    ffi::zfp_decode_partial_block_strided_double_2
);

partial_decode_3d!(
    partial_decode_strided_i32_3d,
    i32,
    any::<i32>(),
    zfp_sys::zfp_encode_block_int32_3,
    zfp_sys::zfp_decode_partial_block_strided_int32_3,
    ffi::zfp_decode_partial_block_strided_int32_3
);
partial_decode_3d!(
    partial_decode_strided_i64_3d,
    i64,
    any::<i64>(),
    zfp_sys::zfp_encode_block_int64_3,
    zfp_sys::zfp_decode_partial_block_strided_int64_3,
    ffi::zfp_decode_partial_block_strided_int64_3
);
partial_decode_3d!(
    partial_decode_strided_f32_3d,
    f32,
    normal_f32(),
    zfp_sys::zfp_encode_block_float_3,
    zfp_sys::zfp_decode_partial_block_strided_float_3,
    ffi::zfp_decode_partial_block_strided_float_3
);
partial_decode_3d!(
    partial_decode_strided_f64_3d,
    f64,
    normal_f64(),
    zfp_sys::zfp_encode_block_double_3,
    zfp_sys::zfp_decode_partial_block_strided_double_3,
    ffi::zfp_decode_partial_block_strided_double_3
);

partial_decode_4d!(
    partial_decode_strided_i32_4d,
    i32,
    any::<i32>(),
    zfp_sys::zfp_encode_block_int32_4,
    zfp_sys::zfp_decode_partial_block_strided_int32_4,
    ffi::zfp_decode_partial_block_strided_int32_4
);
partial_decode_4d!(
    partial_decode_strided_i64_4d,
    i64,
    any::<i64>(),
    zfp_sys::zfp_encode_block_int64_4,
    zfp_sys::zfp_decode_partial_block_strided_int64_4,
    ffi::zfp_decode_partial_block_strided_int64_4
);
partial_decode_4d!(
    partial_decode_strided_f32_4d,
    f32,
    normal_f32(),
    zfp_sys::zfp_encode_block_float_4,
    zfp_sys::zfp_decode_partial_block_strided_float_4,
    ffi::zfp_decode_partial_block_strided_float_4
);
partial_decode_4d!(
    partial_decode_strided_f64_4d,
    f64,
    normal_f64(),
    zfp_sys::zfp_encode_block_double_4,
    zfp_sys::zfp_decode_partial_block_strided_double_4,
    ffi::zfp_decode_partial_block_strided_double_4
);
