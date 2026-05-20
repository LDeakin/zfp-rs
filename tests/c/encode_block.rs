#![allow(clippy::cast_possible_wrap)] // usize↔isize for stride computation
//! Port of `zfp/tests/src/encode/testZfpEncodeBlock{1-4}{d,f,i,l}.c` (16 files).
//!
//! Each C file sets DIMS, ZFP_TYPE, Scalar, BLOCK_SIZE and includes
//! `zfpEncodeBlockBase.c` + `testcases/block.c`.
//!
//! Constants (from universalConsts.h / block[1-4].h):
//!   BLOCK_SIDE_LEN = 4
//!   ZFP_RATE_PARAM_BITS = 19
//!
//! For each (dims, type) combination the block size is 4^dims elements.
#![allow(dead_code)] // Ported from upstream; not all modes/variants are exercised.
#![cfg(feature = "ffi")]

use zfp_rs::ZfpDimensionality;
use zfp_rs::bitstream::ZfpBitStream;
use zfp_rs::codec::block::{encode_block_reversible_f32, encode_block_reversible_f64};
use zfp_rs::codec::encode::{float, integer};

use super::checksums::{
    Subject, TestType, ZfpMode, ZfpScalarType, compute_key, compute_key_original_input,
    get_checksum, hash_array32, hash_array64, hash_bitstream,
};

// ZFP fixed-rate stream parameters (from universalConsts.h)
const ZFP_RATE_PARAM_BITS: u32 = 19;
const ZFP_MAX_PREC: u32 = 64;
const ZFP_MIN_EXP: i32 = -1074;

// ---------------------------------------------------------------------------
// LCG random generators — exact copies from encode_block_strided.rs
// ---------------------------------------------------------------------------

struct Rand32 {
    x: u64,
}

impl Rand32 {
    fn new() -> Self {
        Self { x: 5 }
    }

    fn next_unsigned(&mut self) -> u32 {
        const MULTIPLIER: u64 = 0x5deece66d;
        const INCREMENT: u64 = 0xb;
        const MODULO: u64 = 1 << 48;
        const MASK_31: u32 = 0x7fff_ffff;
        self.x = (MULTIPLIER.wrapping_mul(self.x).wrapping_add(INCREMENT)) % MODULO;
        ((self.x >> 16) as u32) & MASK_31
    }

    fn next_signed_int(&mut self) -> i32 {
        self.next_unsigned() as i32 - 0x4000_0000
    }

    fn next_signed_f32(&mut self) -> f32 {
        let u_val = (self.next_unsigned() >> 7) & 0x00ff_ffff;
        let s_val = u_val as i32 - 0x0080_0000;
        libm::ldexpf(s_val as f32, -12)
    }
}

struct Rand64 {
    x: u64,
}

impl Rand64 {
    fn new() -> Self {
        Self { x: 5 }
    }

    fn next_unsigned(&mut self) -> u64 {
        const MULTIPLIER: u64 = 2862933555777941757;
        const INCREMENT: u64 = 3037000493;
        const MAX_RAND_63: u64 = 0x7fff_ffff_ffff_ffff;
        self.x = MULTIPLIER.wrapping_mul(self.x).wrapping_add(INCREMENT);
        self.x & MAX_RAND_63
    }

    fn next_signed_int(&mut self) -> i64 {
        let u_displace: u64 = 1 << 62;
        self.next_unsigned() as i64 - u_displace as i64
    }

    fn next_signed_f64(&mut self) -> f64 {
        let u_val = (self.next_unsigned() >> 11) & 0x001f_ffff_ffff_ffff;
        let s_val = u_val as i64 - 0x0010_0000_0000_0000_i64;
        libm::ldexp(s_val as f64, -26)
    }
}

// ---------------------------------------------------------------------------
// IEEE-754 special value tables (same as decode_block.rs)
// ---------------------------------------------------------------------------

const SPECIAL_FLOAT_BITS: [u32; 10] = [
    0x0000_0000, // +0
    0x8000_0000, // -0
    0x0000_0001, // +FLT_TRUE_MIN
    0x8000_0001, // -FLT_TRUE_MIN
    0x7f7f_ffff, // +FLT_MAX
    0xff7f_ffff, // -FLT_MAX
    0x7f80_0000, // +infinity
    0xff80_0000, // -infinity
    0x7fc0_0000, // qNaN
    0x7fa0_0000, // sNaN
];

const SPECIAL_DOUBLE_BITS: [u64; 10] = [
    0x0000_0000_0000_0000, // +0
    0x8000_0000_0000_0000, // -0
    0x0000_0000_0000_0001, // +DBL_TRUE_MIN
    0x8000_0000_0000_0001, // -DBL_TRUE_MIN
    0x7fef_ffff_ffff_ffff, // +DBL_MAX
    0xffef_ffff_ffff_ffff, // -DBL_MAX
    0x7ff0_0000_0000_0000, // +infinity
    0xfff0_0000_0000_0000, // -infinity
    0x7ff8_0000_0000_0000, // qNaN
    0x7ff4_0000_0000_0000, // sNaN
];

fn make_special_block_f32(block_size: usize, special_bits: u32) -> Vec<f32> {
    (0..block_size)
        .map(|i| {
            if i % 4 == 0 {
                f32::from_bits(special_bits)
            } else {
                0.0f32
            }
        })
        .collect()
}

fn make_special_block_f64(block_size: usize, special_bits: u64) -> Vec<f64> {
    (0..block_size)
        .map(|i| {
            if i % 4 == 0 {
                f64::from_bits(special_bits)
            } else {
                0.0f64
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Macro: generate tests for one (dims, integer scalar) combination
// ---------------------------------------------------------------------------

macro_rules! encode_block_tests_int {
    (
        $mod_name:ident,
        $scalar:ty,
        $dims:expr,
        $block_size:expr,
        $block_side:expr,
        $rng_new:expr,
        $rng_method:ident,
        $zfp_type:expr,
        $hash_fn:ident,
        $cast:ty,
        $enc_fn:path
    ) => {
        mod $mod_name {
            use super::*;

            // maxbits for fixed-rate mode: block_size * ZFP_RATE_PARAM_BITS
            const MAXBITS: u32 = ($block_size as u32) * ZFP_RATE_PARAM_BITS;

            fn make_block() -> Vec<$scalar> {
                let mut rng = $rng_new();
                (0..$block_size).map(|_| rng.$rng_method()).collect()
            }

            fn dim_lens() -> [usize; 4] {
                let mut n = [0usize; 4];
                for i in 0..$dims {
                    n[i] = $block_side;
                }
                n
            }

            #[test]
            fn when_seeded_random_data_generated_expect_checksum_matches() {
                let data = make_block();
                // reinterpret as u32/u64 via raw pointer cast (matches C's (const UInt*)dataArr)
                let words: Vec<$cast> = unsafe {
                    std::slice::from_raw_parts(data.as_ptr().cast::<$cast>(), $block_size).to_vec()
                };
                let computed = $hash_fn(&words, 1);
                let (key1, key2) = compute_key_original_input(TestType::BlockFull, dim_lens());
                let expected =
                    get_checksum($dims, $zfp_type, key1, key2).expect("checksum not found");
                assert_eq!(u64::from(computed), expected);
            }

            #[test]
            fn given_block_when_encode_block_expect_return_val_reflects_num_bits_written_to_bitstream(
            ) {
                let data = make_block();
                let mut bs = ZfpBitStream::new(65536);
                let block: &[$scalar; $block_size] = data.as_slice().try_into().unwrap();
                // minbits = maxbits for fixed-rate mode (matches C: zfp_stream_set_rate)
                let bits_written = $enc_fn(&mut bs, block, MAXBITS, MAXBITS, ZFP_MAX_PREC);
                // do not flush — extra zero padding would inflate wtell
                assert_eq!(bits_written, bs.bits_written());
            }

            #[test]
            fn given_block_when_encode_block_expect_bitstream_checksum_matches() {
                let data = make_block();
                let mut bs = ZfpBitStream::new(65536);
                let block: &[$scalar; $block_size] = data.as_slice().try_into().unwrap();
                $enc_fn(&mut bs, block, MAXBITS, MAXBITS, ZFP_MAX_PREC);
                bs.flush();
                let computed = hash_bitstream(&bs.as_bytes());
                let (key1, key2) = compute_key(
                    TestType::BlockFull,
                    Subject::CompressedBitstream,
                    dim_lens(),
                    ZfpMode::FixedRate,
                    0,
                );
                let expected =
                    get_checksum($dims, $zfp_type, key1, key2).expect("checksum not found");
                assert_eq!(computed, expected);
            }

            // Integer types do not have a special-values test.
        }
    };
}

// ---------------------------------------------------------------------------
// Macro: generate tests for one (dims, float scalar) combination
// ---------------------------------------------------------------------------

macro_rules! encode_block_tests_float {
    (
        $mod_name:ident,
        $scalar:ty,
        $dims:expr,
        $block_size:expr,
        $block_side:expr,
        $rng_new:expr,
        $rng_method:ident,
        $zfp_type:expr,
        $hash_fn:ident,
        $cast:ty,
        $enc_fn:path,
        $special_bits:expr,
        $make_special:ident,
        $encode_rev:ident
    ) => {
        mod $mod_name {
            use super::*;

            const MAXBITS: u32 = ($block_size as u32) * ZFP_RATE_PARAM_BITS;

            fn make_block() -> Vec<$scalar> {
                let mut rng = $rng_new();
                (0..$block_size).map(|_| rng.$rng_method()).collect()
            }

            fn dim_lens() -> [usize; 4] {
                let mut n = [0usize; 4];
                for i in 0..$dims {
                    n[i] = $block_side;
                }
                n
            }

            #[test]
            fn when_seeded_random_data_generated_expect_checksum_matches() {
                let data = make_block();
                let words: Vec<$cast> = unsafe {
                    std::slice::from_raw_parts(data.as_ptr().cast::<$cast>(), $block_size).to_vec()
                };
                let computed = $hash_fn(&words, 1);
                let (key1, key2) = compute_key_original_input(TestType::BlockFull, dim_lens());
                let expected =
                    get_checksum($dims, $zfp_type, key1, key2).expect("checksum not found");
                assert_eq!(u64::from(computed), expected);
            }

            #[test]
            fn given_block_when_encode_block_expect_return_val_reflects_num_bits_written_to_bitstream(
            ) {
                let data = make_block();
                let mut bs = ZfpBitStream::new(65536);
                let block: &[$scalar; $block_size] = data.as_slice().try_into().unwrap();
                // minbits = maxbits for fixed-rate mode (matches C: zfp_stream_set_rate)
                let bits_written =
                    $enc_fn(&mut bs, block, MAXBITS, MAXBITS, ZFP_MAX_PREC, ZFP_MIN_EXP);
                assert_eq!(bits_written, bs.bits_written());
            }

            #[test]
            fn given_block_when_encode_block_expect_bitstream_checksum_matches() {
                let data = make_block();
                let mut bs = ZfpBitStream::new(65536);
                let block: &[$scalar; $block_size] = data.as_slice().try_into().unwrap();
                $enc_fn(&mut bs, block, MAXBITS, MAXBITS, ZFP_MAX_PREC, ZFP_MIN_EXP);
                bs.flush();
                let computed = hash_bitstream(&bs.as_bytes());
                let (key1, key2) = compute_key(
                    TestType::BlockFull,
                    Subject::CompressedBitstream,
                    dim_lens(),
                    ZfpMode::FixedRate,
                    0,
                );
                let expected =
                    get_checksum($dims, $zfp_type, key1, key2).expect("checksum not found");
                assert_eq!(computed, expected);
            }

            /// Encodes 10 IEEE-754 special value patterns in reversible mode and checks
            /// each compressed bitstream against the pre-computed checksum table.
            /// `misc_param = i + 1` matches `computeKey(…, zfp_mode_reversible, i+1, …)`.
            #[test]
            fn given_block_when_encode_special_blocks_expect_bitstream_checksum_matches() {
                let mut failures = 0;
                for (i, &special_bits) in $special_bits.iter().enumerate() {
                    let data = $make_special($block_size, special_bits);
                    let mut bs = ZfpBitStream::new(65536);
                    $encode_rev(&mut bs, &data, ZfpDimensionality::try_from($dims as u32).unwrap()).unwrap();
                    bs.flush();
                    let computed = hash_bitstream(&bs.as_bytes());
                    let (key1, key2) = compute_key(
                        TestType::BlockFull,
                        Subject::CompressedBitstream,
                        dim_lens(),
                        ZfpMode::Reversible,
                        (i + 1) as u64,
                    );
                    let expected =
                        get_checksum($dims, $zfp_type, key1, key2).expect("checksum not found");
                    if computed != expected {
                        eprintln!("Special Block testcase {i} failed: got {computed:#x}, expected {expected:#x}");
                        failures += 1;
                    }
                }
                assert_eq!(failures, 0, "At least 1 special block testcase failed");
            }
        }
    };
}

// ---------------------------------------------------------------------------
// Test instantiations: 4 dims × 4 types = 16 modules
// ---------------------------------------------------------------------------

// --- 1-D (block_size = 4, block_side = 4) ---
encode_block_tests_float!(
    dim1_double,
    f64,
    1,
    4,
    4,
    Rand64::new,
    next_signed_f64,
    ZfpScalarType::Double,
    hash_array64,
    u64,
    float::encode_block_1d_f64,
    &SPECIAL_DOUBLE_BITS,
    make_special_block_f64,
    encode_block_reversible_f64
);
encode_block_tests_float!(
    dim1_float,
    f32,
    1,
    4,
    4,
    Rand32::new,
    next_signed_f32,
    ZfpScalarType::Float,
    hash_array32,
    u32,
    float::encode_block_1d_f32,
    &SPECIAL_FLOAT_BITS,
    make_special_block_f32,
    encode_block_reversible_f32
);
encode_block_tests_int!(
    dim1_int32,
    i32,
    1,
    4,
    4,
    Rand32::new,
    next_signed_int,
    ZfpScalarType::Int32,
    hash_array32,
    u32,
    integer::encode_block_1d_i32
);
encode_block_tests_int!(
    dim1_int64,
    i64,
    1,
    4,
    4,
    Rand64::new,
    next_signed_int,
    ZfpScalarType::Int64,
    hash_array64,
    u64,
    integer::encode_block_1d_i64
);

// --- 2-D (block_size = 16, block_side = 4) ---
encode_block_tests_float!(
    dim2_double,
    f64,
    2,
    16,
    4,
    Rand64::new,
    next_signed_f64,
    ZfpScalarType::Double,
    hash_array64,
    u64,
    float::encode_block_2d_f64,
    &SPECIAL_DOUBLE_BITS,
    make_special_block_f64,
    encode_block_reversible_f64
);
encode_block_tests_float!(
    dim2_float,
    f32,
    2,
    16,
    4,
    Rand32::new,
    next_signed_f32,
    ZfpScalarType::Float,
    hash_array32,
    u32,
    float::encode_block_2d_f32,
    &SPECIAL_FLOAT_BITS,
    make_special_block_f32,
    encode_block_reversible_f32
);
encode_block_tests_int!(
    dim2_int32,
    i32,
    2,
    16,
    4,
    Rand32::new,
    next_signed_int,
    ZfpScalarType::Int32,
    hash_array32,
    u32,
    integer::encode_block_2d_i32
);
encode_block_tests_int!(
    dim2_int64,
    i64,
    2,
    16,
    4,
    Rand64::new,
    next_signed_int,
    ZfpScalarType::Int64,
    hash_array64,
    u64,
    integer::encode_block_2d_i64
);

// --- 3-D (block_size = 64, block_side = 4) ---
encode_block_tests_float!(
    dim3_double,
    f64,
    3,
    64,
    4,
    Rand64::new,
    next_signed_f64,
    ZfpScalarType::Double,
    hash_array64,
    u64,
    float::encode_block_3d_f64,
    &SPECIAL_DOUBLE_BITS,
    make_special_block_f64,
    encode_block_reversible_f64
);
encode_block_tests_float!(
    dim3_float,
    f32,
    3,
    64,
    4,
    Rand32::new,
    next_signed_f32,
    ZfpScalarType::Float,
    hash_array32,
    u32,
    float::encode_block_3d_f32,
    &SPECIAL_FLOAT_BITS,
    make_special_block_f32,
    encode_block_reversible_f32
);
encode_block_tests_int!(
    dim3_int32,
    i32,
    3,
    64,
    4,
    Rand32::new,
    next_signed_int,
    ZfpScalarType::Int32,
    hash_array32,
    u32,
    integer::encode_block_3d_i32
);
encode_block_tests_int!(
    dim3_int64,
    i64,
    3,
    64,
    4,
    Rand64::new,
    next_signed_int,
    ZfpScalarType::Int64,
    hash_array64,
    u64,
    integer::encode_block_3d_i64
);

// --- 4-D (block_size = 256, block_side = 4) ---
encode_block_tests_float!(
    dim4_double,
    f64,
    4,
    256,
    4,
    Rand64::new,
    next_signed_f64,
    ZfpScalarType::Double,
    hash_array64,
    u64,
    float::encode_block_4d_f64,
    &SPECIAL_DOUBLE_BITS,
    make_special_block_f64,
    encode_block_reversible_f64
);
encode_block_tests_float!(
    dim4_float,
    f32,
    4,
    256,
    4,
    Rand32::new,
    next_signed_f32,
    ZfpScalarType::Float,
    hash_array32,
    u32,
    float::encode_block_4d_f32,
    &SPECIAL_FLOAT_BITS,
    make_special_block_f32,
    encode_block_reversible_f32
);
encode_block_tests_int!(
    dim4_int32,
    i32,
    4,
    256,
    4,
    Rand32::new,
    next_signed_int,
    ZfpScalarType::Int32,
    hash_array32,
    u32,
    integer::encode_block_4d_i32
);
encode_block_tests_int!(
    dim4_int64,
    i64,
    4,
    256,
    4,
    Rand64::new,
    next_signed_int,
    ZfpScalarType::Int64,
    hash_array64,
    u64,
    integer::encode_block_4d_i64
);
