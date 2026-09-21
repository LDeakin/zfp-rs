//! C-compatible type definitions and export helper pattern.
//!
//! Defines all ABI types that mirror the `zfp-sys` generated bindings:
//! scalar aliases, enums, structs/unions, and the `zfp_field` representation.

#![allow(non_camel_case_types)]

// ---------------------------------------------------------------------------
// Scalar type aliases (mirrors zfp-sys / C types)
// ---------------------------------------------------------------------------

/// C `unsigned int`
pub type uint = u32;
/// C `unsigned long long`
pub type uint64 = u64;
/// C `signed char`
pub type int8 = i8;
/// C `unsigned char`
pub type uint8 = u8;
/// C `signed short`
pub type int16 = i16;
/// C `unsigned short`
pub type uint16 = u16;
/// C `signed int`
pub type int32 = i32;
/// C `long long`
pub type int64 = i64;
/// C `int` used as boolean
pub type zfp_bool = i32;

// ---------------------------------------------------------------------------
// Enum constants
// ---------------------------------------------------------------------------

/// C `zfp_exec_policy` values
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum zfp_exec_policy {
    zfp_exec_serial = 0,
    zfp_exec_omp = 1,
    zfp_exec_cuda = 2,
}
pub const zfp_exec_policy_zfp_exec_serial: zfp_exec_policy = zfp_exec_policy::zfp_exec_serial;
pub const zfp_exec_policy_zfp_exec_omp: zfp_exec_policy = zfp_exec_policy::zfp_exec_omp;
pub const zfp_exec_policy_zfp_exec_cuda: zfp_exec_policy = zfp_exec_policy::zfp_exec_cuda;

/// C `zfp_mode` values
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum zfp_mode {
    zfp_mode_null = 0,
    zfp_mode_expert = 1,
    zfp_mode_fixed_rate = 2,
    zfp_mode_fixed_precision = 3,
    zfp_mode_fixed_accuracy = 4,
    zfp_mode_reversible = 5,
}
pub const zfp_mode_zfp_mode_null: zfp_mode = zfp_mode::zfp_mode_null;
pub const zfp_mode_zfp_mode_expert: zfp_mode = zfp_mode::zfp_mode_expert;
pub const zfp_mode_zfp_mode_fixed_rate: zfp_mode = zfp_mode::zfp_mode_fixed_rate;
pub const zfp_mode_zfp_mode_fixed_precision: zfp_mode = zfp_mode::zfp_mode_fixed_precision;
pub const zfp_mode_zfp_mode_fixed_accuracy: zfp_mode = zfp_mode::zfp_mode_fixed_accuracy;
pub const zfp_mode_zfp_mode_reversible: zfp_mode = zfp_mode::zfp_mode_reversible;

/// C `zfp_type` values
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum zfp_type {
    zfp_type_none = 0,
    zfp_type_int32 = 1,
    zfp_type_int64 = 2,
    zfp_type_float = 3,
    zfp_type_double = 4,
}
pub const zfp_type_zfp_type_none: zfp_type = zfp_type::zfp_type_none;
pub const zfp_type_zfp_type_int32: zfp_type = zfp_type::zfp_type_int32;
pub const zfp_type_zfp_type_int64: zfp_type = zfp_type::zfp_type_int64;
pub const zfp_type_zfp_type_float: zfp_type = zfp_type::zfp_type_float;
pub const zfp_type_zfp_type_double: zfp_type = zfp_type::zfp_type_double;

// ---------------------------------------------------------------------------
// Bitstream type aliases
// ---------------------------------------------------------------------------

/// C `bitstream_offset` type
pub type bitstream_offset = uint64;
/// C `bitstream_size` type
pub type bitstream_size = bitstream_offset;
/// C `bitstream_count` type
pub type bitstream_count = usize;

// ---------------------------------------------------------------------------
// Boolean constants
// ---------------------------------------------------------------------------

/// C `zfp_false`
pub const zfp_false: zfp_bool = 0;
/// C `zfp_true`
pub const zfp_true: zfp_bool = 1;

// ---------------------------------------------------------------------------
// Struct: zfp_exec_params_omp
// ---------------------------------------------------------------------------

/// C `zfp_exec_params_omp`
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct zfp_exec_params_omp {
    pub threads: uint,
    pub chunk_size: uint,
}

// ---------------------------------------------------------------------------
// Struct: zfp_execution
// ---------------------------------------------------------------------------

/// C `zfp_execution`
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct zfp_execution {
    pub policy: zfp_exec_policy,
    pub params: *mut std::os::raw::c_void,
}

impl Default for zfp_execution {
    fn default() -> Self {
        Self {
            policy: crate::abi::zfp_exec_policy::zfp_exec_serial,
            params: std::ptr::null_mut(),
        }
    }
}

// ---------------------------------------------------------------------------
// Struct: zfp_stream
// ---------------------------------------------------------------------------

/// Opaque C `bitstream` pointer type (opaque to Rust).
pub type bitstream = std::ffi::c_void;

/// C `zfp_stream`
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct zfp_stream {
    pub minbits: uint,
    pub maxbits: uint,
    pub maxprec: uint,
    pub minexp: i32,
    pub stream: *mut bitstream,
    pub exec: zfp_execution,
}

// ---------------------------------------------------------------------------
// Union: zfp_config__bindgen_ty_1
// ---------------------------------------------------------------------------

/// Payload of the `zfp_config` union: mirrors the C `zfp_config__bindgen_ty_1`.
#[repr(C)]
#[derive(Copy, Clone)]
pub union zfp_config__bindgen_ty_1 {
    pub rate: f64,
    pub precision: uint,
    pub tolerance: f64,
    pub expert: zfp_config__bindgen_ty_1__bindgen_ty_1,
}

impl Default for zfp_config__bindgen_ty_1 {
    fn default() -> Self {
        Self { rate: 0.0 }
    }
}

impl std::fmt::Debug for zfp_config__bindgen_ty_1 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "zfp_config__bindgen_ty_1 {{ ... }}")
    }
}

/// Expert-mode payload inside `zfp_config__bindgen_ty_1`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct zfp_config__bindgen_ty_1__bindgen_ty_1 {
    pub minbits: uint,
    pub maxbits: uint,
    pub maxprec: uint,
    pub minexp: i32,
}

// ---------------------------------------------------------------------------
// Struct: zfp_config
// ---------------------------------------------------------------------------

/// C `zfp_config`
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct zfp_config {
    pub mode: zfp_mode,
    pub arg: zfp_config__bindgen_ty_1,
}

impl Default for zfp_config {
    fn default() -> Self {
        Self {
            mode: crate::abi::zfp_mode::zfp_mode_null,
            arg: zfp_config__bindgen_ty_1::default(),
        }
    }
}

// ---------------------------------------------------------------------------
// Struct: zfp_field
// ---------------------------------------------------------------------------

/// C `zfp_field`: field descriptor for compression/decompression.
/// cbindgen:field-names=[type, nx, ny, nz, nw, sx, sy, sz, sw, data]
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct zfp_field {
    pub r#type: zfp_type,
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub nw: usize,
    pub sx: isize,
    pub sy: isize,
    pub sz: isize,
    pub sw: isize,
    pub data: *mut std::os::raw::c_void,
}

impl Default for zfp_field {
    fn default() -> Self {
        Self {
            r#type: crate::abi::zfp_type::zfp_type_none,
            nx: 0,
            ny: 0,
            nz: 0,
            nw: 0,
            sx: 0,
            sy: 0,
            sz: 0,
            sw: 0,
            data: std::ptr::null_mut(),
        }
    }
}

// ---------------------------------------------------------------------------
// Compression limit constants (mirror zfp.h macros)
// ---------------------------------------------------------------------------

/// Minimum number of bits per block (`ZFP_MIN_BITS`).
pub const ZFP_MIN_BITS: uint = 1;
/// Maximum number of bits per block (`ZFP_MAX_BITS`).
pub const ZFP_MAX_BITS: uint = 16658;
/// Maximum precision in bits (`ZFP_MAX_PREC`).
pub const ZFP_MAX_PREC: uint = 64;
/// Minimum exponent (`ZFP_MIN_EXP`).
pub const ZFP_MIN_EXP: i32 = -1074;

// ---------------------------------------------------------------------------
// Header mask constants (mirror ZFP_HEADER_* macros)
// ---------------------------------------------------------------------------

/// No header sections (`ZFP_HEADER_NONE`).
pub const ZFP_HEADER_NONE: uint = 0;
/// 32-bit magic word (`ZFP_HEADER_MAGIC`).
pub const ZFP_HEADER_MAGIC: uint = 1;
/// 52-bit field metadata (`ZFP_HEADER_META`).
pub const ZFP_HEADER_META: uint = 2;
/// 12- or 64-bit compression mode (`ZFP_HEADER_MODE`).
pub const ZFP_HEADER_MODE: uint = 4;
/// All header sections (`ZFP_HEADER_FULL`).
pub const ZFP_HEADER_FULL: uint = 7;

// ---------------------------------------------------------------------------
// Data position constants (mirror ZFP_DATA_* macros)
// ---------------------------------------------------------------------------

/// Unused data position (`ZFP_DATA_UNUSED`).
pub const ZFP_DATA_UNUSED: uint = 1;
/// Padding data position (`ZFP_DATA_PADDING`).
pub const ZFP_DATA_PADDING: uint = 2;
/// Metadata data position (`ZFP_DATA_META`).
pub const ZFP_DATA_META: uint = 4;
/// Miscellaneous data position (`ZFP_DATA_MISC`).
pub const ZFP_DATA_MISC: uint = 8;
/// Compressed payload data position (`ZFP_DATA_PAYLOAD`).
pub const ZFP_DATA_PAYLOAD: uint = 16;
/// Index data position (`ZFP_DATA_INDEX`).
pub const ZFP_DATA_INDEX: uint = 32;
/// Cache data position (`ZFP_DATA_CACHE`).
pub const ZFP_DATA_CACHE: uint = 64;
/// Header data position (`ZFP_DATA_HEADER`).
pub const ZFP_DATA_HEADER: uint = 128;
/// All data positions (`ZFP_DATA_ALL`).
pub const ZFP_DATA_ALL: uint = 255;

// ---------------------------------------------------------------------------
// Metadata / magic constants
// ---------------------------------------------------------------------------

/// Null metadata value (`u64::MAX`).
pub const ZFP_META_NULL: uint64 = u64::MAX;
/// Number of bits in the magic word (`ZFP_MAGIC_BITS`).
pub const ZFP_MAGIC_BITS: uint = 32;
/// Number of bits in the metadata word (`ZFP_META_BITS`).
pub const ZFP_META_BITS: uint = 52;
/// Number of bits in the short mode word (`ZFP_MODE_SHORT_BITS`).
pub const ZFP_MODE_SHORT_BITS: uint = 12;
/// Number of bits in the long mode word (`ZFP_MODE_LONG_BITS`).
pub const ZFP_MODE_LONG_BITS: uint = 64;
/// Maximum header size in bits (`ZFP_HEADER_MAX_BITS`).
pub const ZFP_HEADER_MAX_BITS: uint = 148;
/// Maximum value representable in a 12-bit mode word (`ZFP_MODE_SHORT_MAX`).
pub const ZFP_MODE_SHORT_MAX: uint = 4094;

// ---------------------------------------------------------------------------
// Rounding policy constants
// ---------------------------------------------------------------------------

/// Round during compression (`ZFP_ROUND_FIRST`).
pub const ZFP_ROUND_FIRST: i32 = -1;
/// Truncate; never round (`ZFP_ROUND_NEVER`). The zfp default.
pub const ZFP_ROUND_NEVER: uint = 0;
/// Round during decompression (`ZFP_ROUND_LAST`).
pub const ZFP_ROUND_LAST: uint = 1;

// ---------------------------------------------------------------------------
// Export attribute note:
//
// All public extern "C" functions use `#[unsafe(no_mangle)]` so the static
// library exports upstream-compatible C symbols.
//
