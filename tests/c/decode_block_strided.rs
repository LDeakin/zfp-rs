#![allow(clippy::cast_possible_wrap)] // usize↔isize for stride computation
//! Port of `zfp/tests/src/decode/testZfpDecodeBlockStrided{1-4}{d,f,i,l}.c` (16 files).
//!
//! Each C file sets DIMS, ZFP_TYPE, Scalar and includes
//! `zfpDecodeBlockStridedBase.c`.
//!
//! Strides (BLOCK_SIDE_LEN = 4):
//!   SX = 2
//!   SY = 3 * BLOCK_SIDE_LEN * SX  = 24
//!   SZ = 2 * BLOCK_SIDE_LEN * SY  = 192
//!   SW = 3 * BLOCK_SIDE_LEN * SZ  = 2304
//!
//! Partial block sizes: PX=1, PY=2, PZ=3, PW=4
//! Dummy value written to non-strided positions: DUMMY_VAL = 99
#![allow(dead_code)] // Ported from upstream; not all modes/variants are exercised.
#![cfg(feature = "ffi")]

use zfp_rs::bitstream::ZfpBitStream;

use super::checksums::{
    Subject, TestType, ZfpMode, ZfpScalarType, compute_key, compute_key_original_input,
    get_checksum, hash_strided_array32, hash_strided_array64,
};

// ZFP fixed-rate stream parameters
const ZFP_RATE_PARAM_BITS: u32 = 19;
const ZFP_MAX_PREC: u32 = 64;
const ZFP_MIN_EXP: i32 = -1074;

// ---------------------------------------------------------------------------
// LCG random generators matching the C test utilities
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
// 1-D decode strided tests
// ---------------------------------------------------------------------------

macro_rules! decode_block_strided_tests_1d {
    (
        $mod_name:ident,
        $scalar:ty,
        $rng_new:expr,
        $rng_val:ident,
        $enc_strided:path,
        $enc_partial:path,
        $dec_strided:path,
        $dec_partial:path,
        $zfp_type:expr,
        $cast:ty,
        $hash_strided_fn:ident,
        $enc_strided_rate:path,
        $enc_partial_rate:path,
        $dec_strided_rate:path,
        $dec_partial_rate:path
        $(, $dec_minexp:expr)?
    ) => {
        mod $mod_name {
            use super::*;

            const BLOCK_SIDE_LEN: usize = 4;
            const SX: isize = 2;
            const PX: usize = 1;
            const DUMMY_VAL: $scalar = 99 as $scalar;
            const MAXBITS: u32 = (BLOCK_SIDE_LEN as u32) * ZFP_RATE_PARAM_BITS;

            fn make_strided_array(dummy: $scalar) -> Vec<$scalar> {
                let count_x = BLOCK_SIDE_LEN * SX as usize;
                let mut data = vec![dummy; count_x];
                let mut rng = $rng_new();
                for i in 0..count_x {
                    if i % SX as usize == 0 {
                        data[i] = rng.$rng_val();
                    }
                }
                data
            }

            fn dim_lens() -> [usize; 4] {
                [BLOCK_SIDE_LEN, 0, 0, 0]
            }

            fn encode_and_rewind_strided(data: &[$scalar]) -> ZfpBitStream {
                let mut bs = ZfpBitStream::new(4096);
                $enc_strided(&mut bs, data, SX);
                bs.flush();
                bs.rewind();
                bs
            }

            fn encode_and_rewind_partial(data: &[$scalar]) -> ZfpBitStream {
                let mut bs = ZfpBitStream::new(4096);
                $enc_partial(&mut bs, data, PX, SX);
                bs.flush();
                bs.rewind();
                bs
            }

            fn encode_rate_and_rewind_strided(data: &[$scalar]) -> ZfpBitStream {
                let mut bs = ZfpBitStream::new(4096);
                $enc_strided_rate(&mut bs, data, SX, MAXBITS, MAXBITS, ZFP_MAX_PREC $(, $dec_minexp)?);
                bs.flush();
                bs.rewind();
                bs
            }

            fn encode_rate_and_rewind_partial(data: &[$scalar]) -> ZfpBitStream {
                let mut bs = ZfpBitStream::new(4096);
                $enc_partial_rate(
                    &mut bs, data, PX, SX, MAXBITS, MAXBITS, ZFP_MAX_PREC $(, $dec_minexp)?,
                );
                bs.flush();
                bs.rewind();
                bs
            }

            #[test]
            fn when_seeded_random_data_generated_expect_checksum_matches() {
                let data = make_strided_array(DUMMY_VAL);
                let words: &[$cast] = unsafe {
                    std::slice::from_raw_parts(data.as_ptr().cast::<$cast>(), data.len())
                };
                let computed =
                    $hash_strided_fn(words, [BLOCK_SIDE_LEN, 0, 0, 0], [SX, 0, 0, 0]);
                let (key1, key2) = compute_key_original_input(TestType::BlockFull, dim_lens());
                let expected =
                    get_checksum(1, $zfp_type, key1, key2).expect("checksum not found");
                assert_eq!(u64::from(computed), expected);
            }

            #[test]
            fn given_block_when_decode_block_strided_expect_return_val_reflects_num_bits_read_from_bitstream(
            ) {
                let data = make_strided_array(DUMMY_VAL);
                let mut bs = encode_and_rewind_strided(&data);
                let arr_len = BLOCK_SIDE_LEN * SX as usize;
                let mut out = vec![0 as $scalar; arr_len];
                let bits_read = $dec_strided(&mut bs, &mut out, SX);
                assert_eq!(bits_read, bs.read_pos() as usize);
            }

            #[test]
            fn given_block_when_decode_block_strided_expect_only_strided_entries_written() {
                let data = make_strided_array(DUMMY_VAL);
                let mut bs = encode_and_rewind_strided(&data);
                let arr_len = BLOCK_SIDE_LEN * SX as usize;
                let mut out = vec![0 as $scalar; arr_len];
                $dec_strided(&mut bs, &mut out, SX);
                // Non-strided entries must remain zero (unchanged)
                for i in 0..arr_len {
                    if i % SX as usize != 0 {
                        assert_eq!(out[i], 0 as $scalar, "non-strided entry [{}] should be 0", i);
                    }
                }
            }

            #[test]
            fn given_block_when_decode_block_strided_expect_array_checksum_matches() {
                let data = make_strided_array(DUMMY_VAL);
                let mut bs = encode_rate_and_rewind_strided(&data);
                let arr_len = BLOCK_SIDE_LEN * SX as usize;
                let mut out = vec![0 as $scalar; arr_len];
                $dec_strided_rate(&mut bs, &mut out, SX, MAXBITS, MAXBITS, ZFP_MAX_PREC $(, $dec_minexp)?);
                let words: &[$cast] = unsafe {
                    std::slice::from_raw_parts(out.as_ptr().cast::<$cast>(), out.len())
                };
                let computed =
                    $hash_strided_fn(words, [BLOCK_SIDE_LEN, 0, 0, 0], [SX, 0, 0, 0]);
                let (key1, key2) = compute_key(
                    TestType::BlockFull,
                    Subject::DecompressedArray,
                    dim_lens(),
                    ZfpMode::FixedRate,
                    0,
                );
                let expected =
                    get_checksum(1, $zfp_type, key1, key2).expect("checksum not found");
                assert_eq!(u64::from(computed), expected);
            }

            #[test]
            fn given_block_when_decode_partial_block_strided_expect_return_val_reflects_num_bits_read_from_bitstream(
            ) {
                let data = make_strided_array(DUMMY_VAL);
                let mut bs = encode_and_rewind_partial(&data);
                let arr_len = BLOCK_SIDE_LEN * SX as usize;
                let mut out = vec![0 as $scalar; arr_len];
                let bits_read = $dec_partial(&mut bs, &mut out, PX, SX);
                assert_eq!(bits_read, bs.read_pos() as usize);
            }

            #[test]
            fn given_block_when_decode_partial_block_strided_expect_only_strided_entries_written() {
                let data = make_strided_array(DUMMY_VAL);
                let mut bs = encode_and_rewind_partial(&data);
                let arr_len = BLOCK_SIDE_LEN * SX as usize;
                let mut out = vec![0 as $scalar; arr_len];
                $dec_partial(&mut bs, &mut out, PX, SX);
                // Non-strided entries must remain zero
                for i in 0..arr_len {
                    if i % SX as usize != 0 {
                        assert_eq!(out[i], 0 as $scalar);
                    }
                }
            }

            #[test]
            fn given_block_when_decode_partial_block_strided_expect_only_entries_within_partial_block_bounds_written(
            ) {
                let data = make_strided_array(DUMMY_VAL);
                let mut bs = encode_and_rewind_partial(&data);
                let arr_len = BLOCK_SIDE_LEN * SX as usize;
                let mut out = vec![0 as $scalar; arr_len];
                $dec_partial(&mut bs, &mut out, PX, SX);
                // Entries at or beyond PX in block coords must remain zero
                for i in 0..arr_len {
                    if i / SX as usize >= PX {
                        assert_eq!(out[i], 0 as $scalar);
                    }
                }
            }

            #[test]
            fn given_block_when_decode_partial_block_strided_expect_array_checksum_matches() {
                let data = make_strided_array(DUMMY_VAL);
                let mut bs = encode_rate_and_rewind_partial(&data);
                let arr_len = BLOCK_SIDE_LEN * SX as usize;
                let mut out = vec![0 as $scalar; arr_len];
                $dec_partial_rate(&mut bs, &mut out, PX, SX, MAXBITS, MAXBITS, ZFP_MAX_PREC $(, $dec_minexp)?);
                let words: &[$cast] = unsafe {
                    std::slice::from_raw_parts(out.as_ptr().cast::<$cast>(), out.len())
                };
                let computed =
                    $hash_strided_fn(words, [BLOCK_SIDE_LEN, 0, 0, 0], [SX, 0, 0, 0]);
                let (key1, key2) = compute_key(
                    TestType::BlockPartial,
                    Subject::DecompressedArray,
                    dim_lens(),
                    ZfpMode::FixedRate,
                    0,
                );
                let expected =
                    get_checksum(1, $zfp_type, key1, key2).expect("checksum not found");
                assert_eq!(u64::from(computed), expected);
            }
        }
    };
}

// ---------------------------------------------------------------------------
// 2-D decode strided tests
// ---------------------------------------------------------------------------

macro_rules! decode_block_strided_tests_2d {
    (
        $mod_name:ident,
        $scalar:ty,
        $rng_new:expr,
        $rng_val:ident,
        $enc_strided:path,
        $enc_partial:path,
        $dec_strided:path,
        $dec_partial:path,
        $zfp_type:expr,
        $cast:ty,
        $hash_strided_fn:ident,
        $enc_strided_rate:path,
        $enc_partial_rate:path,
        $dec_strided_rate:path,
        $dec_partial_rate:path
        $(, $dec_minexp:expr)?
    ) => {
        mod $mod_name {
            use super::*;

            const BLOCK_SIDE_LEN: usize = 4;
            const SX: isize = 2;
            const SY: isize = 3 * BLOCK_SIDE_LEN as isize * SX;
            const PX: usize = 1;
            const PY: usize = 2;
            const DUMMY_VAL: $scalar = 99 as $scalar;
            const MAXBITS: u32 =
                (BLOCK_SIDE_LEN as u32 * BLOCK_SIDE_LEN as u32) * ZFP_RATE_PARAM_BITS;

            fn make_strided_array(dummy: $scalar) -> Vec<$scalar> {
                let count_x = BLOCK_SIDE_LEN * SX as usize;
                let count_y = SY as usize / SX as usize;
                let mut data = vec![dummy; count_x * count_y];
                let mut rng = $rng_new();
                for j in 0..count_y {
                    for i in 0..count_x {
                        let idx = count_x * j + i;
                        if i % (count_x / BLOCK_SIDE_LEN) == 0
                            && j % (count_y / BLOCK_SIDE_LEN) == 0
                        {
                            data[idx] = rng.$rng_val();
                        }
                    }
                }
                data
            }

            fn dim_lens() -> [usize; 4] {
                [BLOCK_SIDE_LEN, BLOCK_SIDE_LEN, 0, 0]
            }

            fn encode_and_rewind_strided(data: &[$scalar]) -> ZfpBitStream {
                let mut bs = ZfpBitStream::new(4096);
                $enc_strided(&mut bs, data, SX, SY);
                bs.flush();
                bs.rewind();
                bs
            }

            fn encode_and_rewind_partial(data: &[$scalar]) -> ZfpBitStream {
                let mut bs = ZfpBitStream::new(4096);
                $enc_partial(&mut bs, data, PX, PY, SX, SY);
                bs.flush();
                bs.rewind();
                bs
            }

            fn encode_rate_and_rewind_strided(data: &[$scalar]) -> ZfpBitStream {
                let mut bs = ZfpBitStream::new(4096);
                $enc_strided_rate(
                    &mut bs, data, SX, SY, MAXBITS, MAXBITS, ZFP_MAX_PREC $(, $dec_minexp)?,
                );
                bs.flush();
                bs.rewind();
                bs
            }

            fn encode_rate_and_rewind_partial(data: &[$scalar]) -> ZfpBitStream {
                let mut bs = ZfpBitStream::new(4096);
                $enc_partial_rate(
                    &mut bs, data, PX, PY, SX, SY, MAXBITS, MAXBITS, ZFP_MAX_PREC $(, $dec_minexp)?,
                );
                bs.flush();
                bs.rewind();
                bs
            }

            #[test]
            fn when_seeded_random_data_generated_expect_checksum_matches() {
                let data = make_strided_array(DUMMY_VAL);
                let words: &[$cast] = unsafe {
                    std::slice::from_raw_parts(data.as_ptr().cast::<$cast>(), data.len())
                };
                let computed = $hash_strided_fn(
                    words,
                    [BLOCK_SIDE_LEN, BLOCK_SIDE_LEN, 0, 0],
                    [SX, SY, 0, 0],
                );
                let (key1, key2) = compute_key_original_input(TestType::BlockFull, dim_lens());
                let expected =
                    get_checksum(2, $zfp_type, key1, key2).expect("checksum not found");
                assert_eq!(u64::from(computed), expected);
            }

            #[test]
            fn given_block_when_decode_block_strided_expect_return_val_reflects_num_bits_read_from_bitstream(
            ) {
                let data = make_strided_array(DUMMY_VAL);
                let mut bs = encode_and_rewind_strided(&data);
                let count_x = BLOCK_SIDE_LEN * SX as usize;
                let count_y = SY as usize / SX as usize;
                let mut out = vec![0 as $scalar; count_x * count_y];
                let bits_read = $dec_strided(&mut bs, &mut out, SX, SY);
                assert_eq!(bits_read, bs.read_pos() as usize);
            }

            #[test]
            fn given_block_when_decode_block_strided_expect_only_strided_entries_written() {
                let data = make_strided_array(DUMMY_VAL);
                let mut bs = encode_and_rewind_strided(&data);
                let count_x = BLOCK_SIDE_LEN * SX as usize;
                let count_y = SY as usize / SX as usize;
                let mut out = vec![0 as $scalar; count_x * count_y];
                $dec_strided(&mut bs, &mut out, SX, SY);
                for j in 0..count_y {
                    for i in 0..count_x {
                        if i % (count_x / BLOCK_SIDE_LEN) != 0
                            || j % (count_y / BLOCK_SIDE_LEN) != 0
                        {
                            assert_eq!(
                                out[count_x * j + i],
                                0 as $scalar,
                                "non-strided [{j},{i}] should be 0"
                            );
                        }
                    }
                }
            }

            #[test]
            fn given_block_when_decode_block_strided_expect_array_checksum_matches() {
                let data = make_strided_array(DUMMY_VAL);
                let mut bs = encode_rate_and_rewind_strided(&data);
                let count_x = BLOCK_SIDE_LEN * SX as usize;
                let count_y = SY as usize / SX as usize;
                let mut out = vec![0 as $scalar; count_x * count_y];
                $dec_strided_rate(&mut bs, &mut out, SX, SY, MAXBITS, MAXBITS, ZFP_MAX_PREC $(, $dec_minexp)?);
                let words: &[$cast] = unsafe {
                    std::slice::from_raw_parts(out.as_ptr().cast::<$cast>(), out.len())
                };
                let computed = $hash_strided_fn(
                    words,
                    [BLOCK_SIDE_LEN, BLOCK_SIDE_LEN, 0, 0],
                    [SX, SY, 0, 0],
                );
                let (key1, key2) = compute_key(
                    TestType::BlockFull,
                    Subject::DecompressedArray,
                    dim_lens(),
                    ZfpMode::FixedRate,
                    0,
                );
                let expected =
                    get_checksum(2, $zfp_type, key1, key2).expect("checksum not found");
                assert_eq!(u64::from(computed), expected);
            }

            #[test]
            fn given_block_when_decode_partial_block_strided_expect_return_val_reflects_num_bits_read_from_bitstream(
            ) {
                let data = make_strided_array(DUMMY_VAL);
                let mut bs = encode_and_rewind_partial(&data);
                let count_x = BLOCK_SIDE_LEN * SX as usize;
                let count_y = SY as usize / SX as usize;
                let mut out = vec![0 as $scalar; count_x * count_y];
                let bits_read = $dec_partial(&mut bs, &mut out, PX, PY, SX, SY);
                assert_eq!(bits_read, bs.read_pos() as usize);
            }

            #[test]
            fn given_block_when_decode_partial_block_strided_expect_only_strided_entries_written() {
                let data = make_strided_array(DUMMY_VAL);
                let mut bs = encode_and_rewind_partial(&data);
                let count_x = BLOCK_SIDE_LEN * SX as usize;
                let count_y = SY as usize / SX as usize;
                let mut out = vec![0 as $scalar; count_x * count_y];
                $dec_partial(&mut bs, &mut out, PX, PY, SX, SY);
                for j in 0..count_y {
                    for i in 0..count_x {
                        if i % (count_x / BLOCK_SIDE_LEN) != 0
                            || j % (count_y / BLOCK_SIDE_LEN) != 0
                        {
                            assert_eq!(out[count_x * j + i], 0 as $scalar);
                        }
                    }
                }
            }

            #[test]
            fn given_block_when_decode_partial_block_strided_expect_only_entries_within_partial_block_bounds_written(
            ) {
                let data = make_strided_array(DUMMY_VAL);
                let mut bs = encode_and_rewind_partial(&data);
                let count_x = BLOCK_SIDE_LEN * SX as usize;
                let count_y = SY as usize / SX as usize;
                let mut out = vec![0 as $scalar; count_x * count_y];
                $dec_partial(&mut bs, &mut out, PX, PY, SX, SY);
                for j in 0..count_y {
                    for i in 0..count_x {
                        if i / (count_x / BLOCK_SIDE_LEN) >= PX
                            || j / (count_y / BLOCK_SIDE_LEN) >= PY
                        {
                            assert_eq!(out[count_x * j + i], 0 as $scalar);
                        }
                    }
                }
            }

            #[test]
            fn given_block_when_decode_partial_block_strided_expect_array_checksum_matches() {
                let data = make_strided_array(DUMMY_VAL);
                let mut bs = encode_rate_and_rewind_partial(&data);
                let count_x = BLOCK_SIDE_LEN * SX as usize;
                let count_y = SY as usize / SX as usize;
                let mut out = vec![0 as $scalar; count_x * count_y];
                $dec_partial_rate(
                    &mut bs, &mut out, PX, PY, SX, SY, MAXBITS, MAXBITS, ZFP_MAX_PREC $(, $dec_minexp)?,
                );
                let words: &[$cast] = unsafe {
                    std::slice::from_raw_parts(out.as_ptr().cast::<$cast>(), out.len())
                };
                let computed = $hash_strided_fn(
                    words,
                    [BLOCK_SIDE_LEN, BLOCK_SIDE_LEN, 0, 0],
                    [SX, SY, 0, 0],
                );
                let (key1, key2) = compute_key(
                    TestType::BlockPartial,
                    Subject::DecompressedArray,
                    dim_lens(),
                    ZfpMode::FixedRate,
                    0,
                );
                let expected =
                    get_checksum(2, $zfp_type, key1, key2).expect("checksum not found");
                assert_eq!(u64::from(computed), expected);
            }
        }
    };
}

// ---------------------------------------------------------------------------
// 3-D decode strided tests
// ---------------------------------------------------------------------------

macro_rules! decode_block_strided_tests_3d {
    (
        $mod_name:ident,
        $scalar:ty,
        $rng_new:expr,
        $rng_val:ident,
        $enc_strided:path,
        $enc_partial:path,
        $dec_strided:path,
        $dec_partial:path,
        $zfp_type:expr,
        $cast:ty,
        $hash_strided_fn:ident,
        $enc_strided_rate:path,
        $enc_partial_rate:path,
        $dec_strided_rate:path,
        $dec_partial_rate:path
        $(, $dec_minexp:expr)?
    ) => {
        mod $mod_name {
            use super::*;

            const BLOCK_SIDE_LEN: usize = 4;
            const SX: isize = 2;
            const SY: isize = 3 * BLOCK_SIDE_LEN as isize * SX;
            const SZ: isize = 2 * BLOCK_SIDE_LEN as isize * SY;
            const PX: usize = 1;
            const PY: usize = 2;
            const PZ: usize = 3;
            const DUMMY_VAL: $scalar = 99 as $scalar;
            const MAXBITS: u32 = (BLOCK_SIDE_LEN as u32
                * BLOCK_SIDE_LEN as u32
                * BLOCK_SIDE_LEN as u32)
                * ZFP_RATE_PARAM_BITS;

            fn make_strided_array(dummy: $scalar) -> Vec<$scalar> {
                let count_x = BLOCK_SIDE_LEN * SX as usize;
                let count_y = SY as usize / SX as usize;
                let count_z = SZ as usize / SY as usize;
                let mut data = vec![dummy; count_x * count_y * count_z];
                let mut rng = $rng_new();
                for k in 0..count_z {
                    for j in 0..count_y {
                        for i in 0..count_x {
                            let idx = count_x * count_y * k + count_x * j + i;
                            if i % (count_x / BLOCK_SIDE_LEN) == 0
                                && j % (count_y / BLOCK_SIDE_LEN) == 0
                                && k % (count_z / BLOCK_SIDE_LEN) == 0
                            {
                                data[idx] = rng.$rng_val();
                            }
                        }
                    }
                }
                data
            }

            fn dim_lens() -> [usize; 4] {
                [BLOCK_SIDE_LEN, BLOCK_SIDE_LEN, BLOCK_SIDE_LEN, 0]
            }

            fn encode_and_rewind_strided(data: &[$scalar]) -> ZfpBitStream {
                let mut bs = ZfpBitStream::new(65536);
                $enc_strided(&mut bs, data, SX, SY, SZ);
                bs.flush();
                bs.rewind();
                bs
            }

            fn encode_and_rewind_partial(data: &[$scalar]) -> ZfpBitStream {
                let mut bs = ZfpBitStream::new(65536);
                $enc_partial(&mut bs, data, PX, PY, PZ, SX, SY, SZ);
                bs.flush();
                bs.rewind();
                bs
            }

            fn encode_rate_and_rewind_strided(data: &[$scalar]) -> ZfpBitStream {
                let mut bs = ZfpBitStream::new(65536);
                $enc_strided_rate(
                    &mut bs, data, SX, SY, SZ, MAXBITS, MAXBITS, ZFP_MAX_PREC $(, $dec_minexp)?,
                );
                bs.flush();
                bs.rewind();
                bs
            }

            fn encode_rate_and_rewind_partial(data: &[$scalar]) -> ZfpBitStream {
                let mut bs = ZfpBitStream::new(65536);
                $enc_partial_rate(
                    &mut bs, data, PX, PY, PZ, SX, SY, SZ, MAXBITS, MAXBITS, ZFP_MAX_PREC
                    $(, $dec_minexp)?,
                );
                bs.flush();
                bs.rewind();
                bs
            }

            #[test]
            fn when_seeded_random_data_generated_expect_checksum_matches() {
                let data = make_strided_array(DUMMY_VAL);
                let words: &[$cast] = unsafe {
                    std::slice::from_raw_parts(data.as_ptr().cast::<$cast>(), data.len())
                };
                let computed = $hash_strided_fn(
                    words,
                    [BLOCK_SIDE_LEN, BLOCK_SIDE_LEN, BLOCK_SIDE_LEN, 0],
                    [SX, SY, SZ, 0],
                );
                let (key1, key2) = compute_key_original_input(TestType::BlockFull, dim_lens());
                let expected =
                    get_checksum(3, $zfp_type, key1, key2).expect("checksum not found");
                assert_eq!(u64::from(computed), expected);
            }

            #[test]
            fn given_block_when_decode_block_strided_expect_return_val_reflects_num_bits_read_from_bitstream(
            ) {
                let data = make_strided_array(DUMMY_VAL);
                let mut bs = encode_and_rewind_strided(&data);
                let count_x = BLOCK_SIDE_LEN * SX as usize;
                let count_y = SY as usize / SX as usize;
                let count_z = SZ as usize / SY as usize;
                let mut out = vec![0 as $scalar; count_x * count_y * count_z];
                let bits_read = $dec_strided(&mut bs, &mut out, SX, SY, SZ);
                assert_eq!(bits_read, bs.read_pos() as usize);
            }

            #[test]
            fn given_block_when_decode_block_strided_expect_only_strided_entries_written() {
                let data = make_strided_array(DUMMY_VAL);
                let mut bs = encode_and_rewind_strided(&data);
                let count_x = BLOCK_SIDE_LEN * SX as usize;
                let count_y = SY as usize / SX as usize;
                let count_z = SZ as usize / SY as usize;
                let mut out = vec![0 as $scalar; count_x * count_y * count_z];
                $dec_strided(&mut bs, &mut out, SX, SY, SZ);
                for k in 0..count_z {
                    for j in 0..count_y {
                        for i in 0..count_x {
                            if i % (count_x / BLOCK_SIDE_LEN) != 0
                                || j % (count_y / BLOCK_SIDE_LEN) != 0
                                || k % (count_z / BLOCK_SIDE_LEN) != 0
                            {
                                assert_eq!(
                                    out[count_x * count_y * k + count_x * j + i],
                                    0 as $scalar
                                );
                            }
                        }
                    }
                }
            }

            #[test]
            fn given_block_when_decode_block_strided_expect_array_checksum_matches() {
                let data = make_strided_array(DUMMY_VAL);
                let mut bs = encode_rate_and_rewind_strided(&data);
                let count_x = BLOCK_SIDE_LEN * SX as usize;
                let count_y = SY as usize / SX as usize;
                let count_z = SZ as usize / SY as usize;
                let mut out = vec![0 as $scalar; count_x * count_y * count_z];
                $dec_strided_rate(
                    &mut bs, &mut out, SX, SY, SZ, MAXBITS, MAXBITS, ZFP_MAX_PREC $(, $dec_minexp)?,
                );
                let words: &[$cast] = unsafe {
                    std::slice::from_raw_parts(out.as_ptr().cast::<$cast>(), out.len())
                };
                let computed = $hash_strided_fn(
                    words,
                    [BLOCK_SIDE_LEN, BLOCK_SIDE_LEN, BLOCK_SIDE_LEN, 0],
                    [SX, SY, SZ, 0],
                );
                let (key1, key2) = compute_key(
                    TestType::BlockFull,
                    Subject::DecompressedArray,
                    dim_lens(),
                    ZfpMode::FixedRate,
                    0,
                );
                let expected =
                    get_checksum(3, $zfp_type, key1, key2).expect("checksum not found");
                assert_eq!(u64::from(computed), expected);
            }

            #[test]
            fn given_block_when_decode_partial_block_strided_expect_return_val_reflects_num_bits_read_from_bitstream(
            ) {
                let data = make_strided_array(DUMMY_VAL);
                let mut bs = encode_and_rewind_partial(&data);
                let count_x = BLOCK_SIDE_LEN * SX as usize;
                let count_y = SY as usize / SX as usize;
                let count_z = SZ as usize / SY as usize;
                let mut out = vec![0 as $scalar; count_x * count_y * count_z];
                let bits_read = $dec_partial(&mut bs, &mut out, PX, PY, PZ, SX, SY, SZ);
                assert_eq!(bits_read, bs.read_pos() as usize);
            }

            #[test]
            fn given_block_when_decode_partial_block_strided_expect_only_strided_entries_written() {
                let data = make_strided_array(DUMMY_VAL);
                let mut bs = encode_and_rewind_partial(&data);
                let count_x = BLOCK_SIDE_LEN * SX as usize;
                let count_y = SY as usize / SX as usize;
                let count_z = SZ as usize / SY as usize;
                let mut out = vec![0 as $scalar; count_x * count_y * count_z];
                $dec_partial(&mut bs, &mut out, PX, PY, PZ, SX, SY, SZ);
                for k in 0..count_z {
                    for j in 0..count_y {
                        for i in 0..count_x {
                            if i % (count_x / BLOCK_SIDE_LEN) != 0
                                || j % (count_y / BLOCK_SIDE_LEN) != 0
                                || k % (count_z / BLOCK_SIDE_LEN) != 0
                            {
                                assert_eq!(
                                    out[count_x * count_y * k + count_x * j + i],
                                    0 as $scalar
                                );
                            }
                        }
                    }
                }
            }

            #[test]
            fn given_block_when_decode_partial_block_strided_expect_only_entries_within_partial_block_bounds_written(
            ) {
                let data = make_strided_array(DUMMY_VAL);
                let mut bs = encode_and_rewind_partial(&data);
                let count_x = BLOCK_SIDE_LEN * SX as usize;
                let count_y = SY as usize / SX as usize;
                let count_z = SZ as usize / SY as usize;
                let mut out = vec![0 as $scalar; count_x * count_y * count_z];
                $dec_partial(&mut bs, &mut out, PX, PY, PZ, SX, SY, SZ);
                for k in 0..count_z {
                    for j in 0..count_y {
                        for i in 0..count_x {
                            if i / (count_x / BLOCK_SIDE_LEN) >= PX
                                || j / (count_y / BLOCK_SIDE_LEN) >= PY
                                || k / (count_z / BLOCK_SIDE_LEN) >= PZ
                            {
                                assert_eq!(
                                    out[count_x * count_y * k + count_x * j + i],
                                    0 as $scalar
                                );
                            }
                        }
                    }
                }
            }

            #[test]
            fn given_block_when_decode_partial_block_strided_expect_array_checksum_matches() {
                let data = make_strided_array(DUMMY_VAL);
                let mut bs = encode_rate_and_rewind_partial(&data);
                let count_x = BLOCK_SIDE_LEN * SX as usize;
                let count_y = SY as usize / SX as usize;
                let count_z = SZ as usize / SY as usize;
                let mut out = vec![0 as $scalar; count_x * count_y * count_z];
                $dec_partial_rate(
                    &mut bs, &mut out, PX, PY, PZ, SX, SY, SZ, MAXBITS, MAXBITS, ZFP_MAX_PREC $(, $dec_minexp)?,
                );
                let words: &[$cast] = unsafe {
                    std::slice::from_raw_parts(out.as_ptr().cast::<$cast>(), out.len())
                };
                let computed = $hash_strided_fn(
                    words,
                    [BLOCK_SIDE_LEN, BLOCK_SIDE_LEN, BLOCK_SIDE_LEN, 0],
                    [SX, SY, SZ, 0],
                );
                let (key1, key2) = compute_key(
                    TestType::BlockPartial,
                    Subject::DecompressedArray,
                    dim_lens(),
                    ZfpMode::FixedRate,
                    0,
                );
                let expected =
                    get_checksum(3, $zfp_type, key1, key2).expect("checksum not found");
                assert_eq!(u64::from(computed), expected);
            }
        }
    };
}

// ---------------------------------------------------------------------------
// 4-D decode strided tests
// ---------------------------------------------------------------------------

macro_rules! decode_block_strided_tests_4d {
    (
        $mod_name:ident,
        $scalar:ty,
        $rng_new:expr,
        $rng_val:ident,
        $enc_strided:path,
        $enc_partial:path,
        $dec_strided:path,
        $dec_partial:path,
        $zfp_type:expr,
        $cast:ty,
        $hash_strided_fn:ident,
        $enc_strided_rate:path,
        $enc_partial_rate:path,
        $dec_strided_rate:path,
        $dec_partial_rate:path
        $(, $dec_minexp:expr)?
    ) => {
        mod $mod_name {
            use super::*;

            const BLOCK_SIDE_LEN: usize = 4;
            const SX: isize = 2;
            const SY: isize = 3 * BLOCK_SIDE_LEN as isize * SX;
            const SZ: isize = 2 * BLOCK_SIDE_LEN as isize * SY;
            const SW: isize = 3 * BLOCK_SIDE_LEN as isize * SZ;
            const PX: usize = 1;
            const PY: usize = 2;
            const PZ: usize = 3;
            const PW: usize = 4;
            const DUMMY_VAL: $scalar = 99 as $scalar;
            const MAXBITS: u32 = (BLOCK_SIDE_LEN as u32
                * BLOCK_SIDE_LEN as u32
                * BLOCK_SIDE_LEN as u32
                * BLOCK_SIDE_LEN as u32)
                * ZFP_RATE_PARAM_BITS;

            fn make_strided_array(dummy: $scalar) -> Vec<$scalar> {
                let count_x = BLOCK_SIDE_LEN * SX as usize;
                let count_y = SY as usize / SX as usize;
                let count_z = SZ as usize / SY as usize;
                let count_w = SW as usize / SZ as usize;
                let mut data = vec![dummy; count_x * count_y * count_z * count_w];
                let mut rng = $rng_new();
                for l in 0..count_w {
                    for k in 0..count_z {
                        for j in 0..count_y {
                            for i in 0..count_x {
                                let idx = count_x * count_y * count_z * l
                                    + count_x * count_y * k
                                    + count_x * j
                                    + i;
                                if i % (count_x / BLOCK_SIDE_LEN) == 0
                                    && j % (count_y / BLOCK_SIDE_LEN) == 0
                                    && k % (count_z / BLOCK_SIDE_LEN) == 0
                                    && l % (count_w / BLOCK_SIDE_LEN) == 0
                                {
                                    data[idx] = rng.$rng_val();
                                }
                            }
                        }
                    }
                }
                data
            }

            fn dim_lens() -> [usize; 4] {
                [BLOCK_SIDE_LEN, BLOCK_SIDE_LEN, BLOCK_SIDE_LEN, BLOCK_SIDE_LEN]
            }

            fn encode_and_rewind_strided(data: &[$scalar]) -> ZfpBitStream {
                let mut bs = ZfpBitStream::new(1 << 20);
                $enc_strided(&mut bs, data, SX, SY, SZ, SW);
                bs.flush();
                bs.rewind();
                bs
            }

            fn encode_and_rewind_partial(data: &[$scalar]) -> ZfpBitStream {
                let mut bs = ZfpBitStream::new(1 << 20);
                $enc_partial(&mut bs, data, PX, PY, PZ, PW, SX, SY, SZ, SW);
                bs.flush();
                bs.rewind();
                bs
            }

            fn encode_rate_and_rewind_strided(data: &[$scalar]) -> ZfpBitStream {
                let mut bs = ZfpBitStream::new(1 << 20);
                $enc_strided_rate(
                    &mut bs, data, SX, SY, SZ, SW, MAXBITS, MAXBITS, ZFP_MAX_PREC $(, $dec_minexp)?,
                );
                bs.flush();
                bs.rewind();
                bs
            }

            fn encode_rate_and_rewind_partial(data: &[$scalar]) -> ZfpBitStream {
                let mut bs = ZfpBitStream::new(1 << 20);
                $enc_partial_rate(
                    &mut bs, data, PX, PY, PZ, PW, SX, SY, SZ, SW, MAXBITS, MAXBITS,
                    ZFP_MAX_PREC $(, $dec_minexp)?,
                );
                bs.flush();
                bs.rewind();
                bs
            }

            #[test]
            fn when_seeded_random_data_generated_expect_checksum_matches() {
                let data = make_strided_array(DUMMY_VAL);
                let words: &[$cast] = unsafe {
                    std::slice::from_raw_parts(data.as_ptr().cast::<$cast>(), data.len())
                };
                let computed = $hash_strided_fn(
                    words,
                    [BLOCK_SIDE_LEN, BLOCK_SIDE_LEN, BLOCK_SIDE_LEN, BLOCK_SIDE_LEN],
                    [SX, SY, SZ, SW],
                );
                let (key1, key2) = compute_key_original_input(TestType::BlockFull, dim_lens());
                let expected =
                    get_checksum(4, $zfp_type, key1, key2).expect("checksum not found");
                assert_eq!(u64::from(computed), expected);
            }

            #[test]
            fn given_block_when_decode_block_strided_expect_return_val_reflects_num_bits_read_from_bitstream(
            ) {
                let data = make_strided_array(DUMMY_VAL);
                let mut bs = encode_and_rewind_strided(&data);
                let count_x = BLOCK_SIDE_LEN * SX as usize;
                let count_y = SY as usize / SX as usize;
                let count_z = SZ as usize / SY as usize;
                let count_w = SW as usize / SZ as usize;
                let mut out = vec![0 as $scalar; count_x * count_y * count_z * count_w];
                let bits_read = $dec_strided(&mut bs, &mut out, SX, SY, SZ, SW);
                assert_eq!(bits_read, bs.read_pos() as usize);
            }

            #[test]
            fn given_block_when_decode_block_strided_expect_only_strided_entries_written() {
                let data = make_strided_array(DUMMY_VAL);
                let mut bs = encode_and_rewind_strided(&data);
                let count_x = BLOCK_SIDE_LEN * SX as usize;
                let count_y = SY as usize / SX as usize;
                let count_z = SZ as usize / SY as usize;
                let count_w = SW as usize / SZ as usize;
                let mut out = vec![0 as $scalar; count_x * count_y * count_z * count_w];
                $dec_strided(&mut bs, &mut out, SX, SY, SZ, SW);
                for l in 0..count_w {
                    for k in 0..count_z {
                        for j in 0..count_y {
                            for i in 0..count_x {
                                if i % (count_x / BLOCK_SIDE_LEN) != 0
                                    || j % (count_y / BLOCK_SIDE_LEN) != 0
                                    || k % (count_z / BLOCK_SIDE_LEN) != 0
                                    || l % (count_w / BLOCK_SIDE_LEN) != 0
                                {
                                    let idx = count_x * count_y * count_z * l
                                        + count_x * count_y * k
                                        + count_x * j
                                        + i;
                                    assert_eq!(out[idx], 0 as $scalar);
                                }
                            }
                        }
                    }
                }
            }

            #[test]
            fn given_block_when_decode_block_strided_expect_array_checksum_matches() {
                let data = make_strided_array(DUMMY_VAL);
                let mut bs = encode_rate_and_rewind_strided(&data);
                let count_x = BLOCK_SIDE_LEN * SX as usize;
                let count_y = SY as usize / SX as usize;
                let count_z = SZ as usize / SY as usize;
                let count_w = SW as usize / SZ as usize;
                let mut out = vec![0 as $scalar; count_x * count_y * count_z * count_w];
                $dec_strided_rate(
                    &mut bs, &mut out, SX, SY, SZ, SW, MAXBITS, MAXBITS, ZFP_MAX_PREC $(, $dec_minexp)?,
                );
                let words: &[$cast] = unsafe {
                    std::slice::from_raw_parts(out.as_ptr().cast::<$cast>(), out.len())
                };
                let computed = $hash_strided_fn(
                    words,
                    [BLOCK_SIDE_LEN, BLOCK_SIDE_LEN, BLOCK_SIDE_LEN, BLOCK_SIDE_LEN],
                    [SX, SY, SZ, SW],
                );
                let (key1, key2) = compute_key(
                    TestType::BlockFull,
                    Subject::DecompressedArray,
                    dim_lens(),
                    ZfpMode::FixedRate,
                    0,
                );
                let expected =
                    get_checksum(4, $zfp_type, key1, key2).expect("checksum not found");
                assert_eq!(u64::from(computed), expected);
            }

            #[test]
            fn given_block_when_decode_partial_block_strided_expect_return_val_reflects_num_bits_read_from_bitstream(
            ) {
                let data = make_strided_array(DUMMY_VAL);
                let mut bs = encode_and_rewind_partial(&data);
                let count_x = BLOCK_SIDE_LEN * SX as usize;
                let count_y = SY as usize / SX as usize;
                let count_z = SZ as usize / SY as usize;
                let count_w = SW as usize / SZ as usize;
                let mut out = vec![0 as $scalar; count_x * count_y * count_z * count_w];
                let bits_read = $dec_partial(&mut bs, &mut out, PX, PY, PZ, PW, SX, SY, SZ, SW);
                assert_eq!(bits_read, bs.read_pos() as usize);
            }

            #[test]
            fn given_block_when_decode_partial_block_strided_expect_only_strided_entries_written() {
                let data = make_strided_array(DUMMY_VAL);
                let mut bs = encode_and_rewind_partial(&data);
                let count_x = BLOCK_SIDE_LEN * SX as usize;
                let count_y = SY as usize / SX as usize;
                let count_z = SZ as usize / SY as usize;
                let count_w = SW as usize / SZ as usize;
                let mut out = vec![0 as $scalar; count_x * count_y * count_z * count_w];
                $dec_partial(&mut bs, &mut out, PX, PY, PZ, PW, SX, SY, SZ, SW);
                for l in 0..count_w {
                    for k in 0..count_z {
                        for j in 0..count_y {
                            for i in 0..count_x {
                                if i % (count_x / BLOCK_SIDE_LEN) != 0
                                    || j % (count_y / BLOCK_SIDE_LEN) != 0
                                    || k % (count_z / BLOCK_SIDE_LEN) != 0
                                    || l % (count_w / BLOCK_SIDE_LEN) != 0
                                {
                                    let idx = count_x * count_y * count_z * l
                                        + count_x * count_y * k
                                        + count_x * j
                                        + i;
                                    assert_eq!(out[idx], 0 as $scalar);
                                }
                            }
                        }
                    }
                }
            }

            #[test]
            fn given_block_when_decode_partial_block_strided_expect_only_entries_within_partial_block_bounds_written(
            ) {
                let data = make_strided_array(DUMMY_VAL);
                let mut bs = encode_and_rewind_partial(&data);
                let count_x = BLOCK_SIDE_LEN * SX as usize;
                let count_y = SY as usize / SX as usize;
                let count_z = SZ as usize / SY as usize;
                let count_w = SW as usize / SZ as usize;
                let mut out = vec![0 as $scalar; count_x * count_y * count_z * count_w];
                $dec_partial(&mut bs, &mut out, PX, PY, PZ, PW, SX, SY, SZ, SW);
                for l in 0..count_w {
                    for k in 0..count_z {
                        for j in 0..count_y {
                            for i in 0..count_x {
                                if i / (count_x / BLOCK_SIDE_LEN) >= PX
                                    || j / (count_y / BLOCK_SIDE_LEN) >= PY
                                    || k / (count_z / BLOCK_SIDE_LEN) >= PZ
                                    || l / (count_w / BLOCK_SIDE_LEN) >= PW
                                {
                                    let idx = count_x * count_y * count_z * l
                                        + count_x * count_y * k
                                        + count_x * j
                                        + i;
                                    assert_eq!(out[idx], 0 as $scalar);
                                }
                            }
                        }
                    }
                }
            }

            #[test]
            fn given_block_when_decode_partial_block_strided_expect_array_checksum_matches() {
                let data = make_strided_array(DUMMY_VAL);
                let mut bs = encode_rate_and_rewind_partial(&data);
                let count_x = BLOCK_SIDE_LEN * SX as usize;
                let count_y = SY as usize / SX as usize;
                let count_z = SZ as usize / SY as usize;
                let count_w = SW as usize / SZ as usize;
                let mut out = vec![0 as $scalar; count_x * count_y * count_z * count_w];
                $dec_partial_rate(
                    &mut bs, &mut out, PX, PY, PZ, PW, SX, SY, SZ, SW, MAXBITS, MAXBITS,
                    ZFP_MAX_PREC $(, $dec_minexp)?,
                );
                let words: &[$cast] = unsafe {
                    std::slice::from_raw_parts(out.as_ptr().cast::<$cast>(), out.len())
                };
                let computed = $hash_strided_fn(
                    words,
                    [BLOCK_SIDE_LEN, BLOCK_SIDE_LEN, BLOCK_SIDE_LEN, BLOCK_SIDE_LEN],
                    [SX, SY, SZ, SW],
                );
                let (key1, key2) = compute_key(
                    TestType::BlockPartial,
                    Subject::DecompressedArray,
                    dim_lens(),
                    ZfpMode::FixedRate,
                    0,
                );
                let expected =
                    get_checksum(4, $zfp_type, key1, key2).expect("checksum not found");
                assert_eq!(u64::from(computed), expected);
            }
        }
    };
}

// ---------------------------------------------------------------------------
// Instantiate tests for all 16 (dims × type) combinations
// ---------------------------------------------------------------------------

use zfp_rs::codec::decode::{dim1, dim2, dim3, dim4};
use zfp_rs::codec::encode::{dim1 as enc1, dim2 as enc2, dim3 as enc3, dim4 as enc4};

decode_block_strided_tests_1d!(
    dim1_double,
    f64,
    Rand64::new,
    next_signed_f64,
    enc1::encode_block_strided_1d_f64,
    enc1::encode_partial_block_strided_1d_f64,
    dim1::decode_block_strided_1d_f64,
    dim1::decode_partial_block_strided_1d_f64,
    ZfpScalarType::Double,
    u64,
    hash_strided_array64,
    enc1::encode_block_strided_1d_f64_rate,
    enc1::encode_partial_block_strided_1d_f64_rate,
    dim1::decode_block_strided_1d_f64_rate,
    dim1::decode_partial_block_strided_1d_f64_rate,
    ZFP_MIN_EXP
);
decode_block_strided_tests_1d!(
    dim1_float,
    f32,
    Rand32::new,
    next_signed_f32,
    enc1::encode_block_strided_1d_f32,
    enc1::encode_partial_block_strided_1d_f32,
    dim1::decode_block_strided_1d_f32,
    dim1::decode_partial_block_strided_1d_f32,
    ZfpScalarType::Float,
    u32,
    hash_strided_array32,
    enc1::encode_block_strided_1d_f32_rate,
    enc1::encode_partial_block_strided_1d_f32_rate,
    dim1::decode_block_strided_1d_f32_rate,
    dim1::decode_partial_block_strided_1d_f32_rate,
    ZFP_MIN_EXP
);
decode_block_strided_tests_1d!(
    dim1_int32,
    i32,
    Rand32::new,
    next_signed_int,
    enc1::encode_block_strided_1d_i32,
    enc1::encode_partial_block_strided_1d_i32,
    dim1::decode_block_strided_1d_i32,
    dim1::decode_partial_block_strided_1d_i32,
    ZfpScalarType::Int32,
    u32,
    hash_strided_array32,
    enc1::encode_block_strided_1d_i32_rate,
    enc1::encode_partial_block_strided_1d_i32_rate,
    dim1::decode_block_strided_1d_i32_rate,
    dim1::decode_partial_block_strided_1d_i32_rate
);
decode_block_strided_tests_1d!(
    dim1_int64,
    i64,
    Rand64::new,
    next_signed_int,
    enc1::encode_block_strided_1d_i64,
    enc1::encode_partial_block_strided_1d_i64,
    dim1::decode_block_strided_1d_i64,
    dim1::decode_partial_block_strided_1d_i64,
    ZfpScalarType::Int64,
    u64,
    hash_strided_array64,
    enc1::encode_block_strided_1d_i64_rate,
    enc1::encode_partial_block_strided_1d_i64_rate,
    dim1::decode_block_strided_1d_i64_rate,
    dim1::decode_partial_block_strided_1d_i64_rate
);

decode_block_strided_tests_2d!(
    dim2_double,
    f64,
    Rand64::new,
    next_signed_f64,
    enc2::encode_block_strided_2d_f64,
    enc2::encode_partial_block_strided_2d_f64,
    dim2::decode_block_strided_2d_f64,
    dim2::decode_partial_block_strided_2d_f64,
    ZfpScalarType::Double,
    u64,
    hash_strided_array64,
    enc2::encode_block_strided_2d_f64_rate,
    enc2::encode_partial_block_strided_2d_f64_rate,
    dim2::decode_block_strided_2d_f64_rate,
    dim2::decode_partial_block_strided_2d_f64_rate,
    ZFP_MIN_EXP
);
decode_block_strided_tests_2d!(
    dim2_float,
    f32,
    Rand32::new,
    next_signed_f32,
    enc2::encode_block_strided_2d_f32,
    enc2::encode_partial_block_strided_2d_f32,
    dim2::decode_block_strided_2d_f32,
    dim2::decode_partial_block_strided_2d_f32,
    ZfpScalarType::Float,
    u32,
    hash_strided_array32,
    enc2::encode_block_strided_2d_f32_rate,
    enc2::encode_partial_block_strided_2d_f32_rate,
    dim2::decode_block_strided_2d_f32_rate,
    dim2::decode_partial_block_strided_2d_f32_rate,
    ZFP_MIN_EXP
);
decode_block_strided_tests_2d!(
    dim2_int32,
    i32,
    Rand32::new,
    next_signed_int,
    enc2::encode_block_strided_2d_i32,
    enc2::encode_partial_block_strided_2d_i32,
    dim2::decode_block_strided_2d_i32,
    dim2::decode_partial_block_strided_2d_i32,
    ZfpScalarType::Int32,
    u32,
    hash_strided_array32,
    enc2::encode_block_strided_2d_i32_rate,
    enc2::encode_partial_block_strided_2d_i32_rate,
    dim2::decode_block_strided_2d_i32_rate,
    dim2::decode_partial_block_strided_2d_i32_rate
);
decode_block_strided_tests_2d!(
    dim2_int64,
    i64,
    Rand64::new,
    next_signed_int,
    enc2::encode_block_strided_2d_i64,
    enc2::encode_partial_block_strided_2d_i64,
    dim2::decode_block_strided_2d_i64,
    dim2::decode_partial_block_strided_2d_i64,
    ZfpScalarType::Int64,
    u64,
    hash_strided_array64,
    enc2::encode_block_strided_2d_i64_rate,
    enc2::encode_partial_block_strided_2d_i64_rate,
    dim2::decode_block_strided_2d_i64_rate,
    dim2::decode_partial_block_strided_2d_i64_rate
);

decode_block_strided_tests_3d!(
    dim3_double,
    f64,
    Rand64::new,
    next_signed_f64,
    enc3::encode_block_strided_3d_f64,
    enc3::encode_partial_block_strided_3d_f64,
    dim3::decode_block_strided_3d_f64,
    dim3::decode_partial_block_strided_3d_f64,
    ZfpScalarType::Double,
    u64,
    hash_strided_array64,
    enc3::encode_block_strided_3d_f64_rate,
    enc3::encode_partial_block_strided_3d_f64_rate,
    dim3::decode_block_strided_3d_f64_rate,
    dim3::decode_partial_block_strided_3d_f64_rate,
    ZFP_MIN_EXP
);
decode_block_strided_tests_3d!(
    dim3_float,
    f32,
    Rand32::new,
    next_signed_f32,
    enc3::encode_block_strided_3d_f32,
    enc3::encode_partial_block_strided_3d_f32,
    dim3::decode_block_strided_3d_f32,
    dim3::decode_partial_block_strided_3d_f32,
    ZfpScalarType::Float,
    u32,
    hash_strided_array32,
    enc3::encode_block_strided_3d_f32_rate,
    enc3::encode_partial_block_strided_3d_f32_rate,
    dim3::decode_block_strided_3d_f32_rate,
    dim3::decode_partial_block_strided_3d_f32_rate,
    ZFP_MIN_EXP
);
decode_block_strided_tests_3d!(
    dim3_int32,
    i32,
    Rand32::new,
    next_signed_int,
    enc3::encode_block_strided_3d_i32,
    enc3::encode_partial_block_strided_3d_i32,
    dim3::decode_block_strided_3d_i32,
    dim3::decode_partial_block_strided_3d_i32,
    ZfpScalarType::Int32,
    u32,
    hash_strided_array32,
    enc3::encode_block_strided_3d_i32_rate,
    enc3::encode_partial_block_strided_3d_i32_rate,
    dim3::decode_block_strided_3d_i32_rate,
    dim3::decode_partial_block_strided_3d_i32_rate
);
decode_block_strided_tests_3d!(
    dim3_int64,
    i64,
    Rand64::new,
    next_signed_int,
    enc3::encode_block_strided_3d_i64,
    enc3::encode_partial_block_strided_3d_i64,
    dim3::decode_block_strided_3d_i64,
    dim3::decode_partial_block_strided_3d_i64,
    ZfpScalarType::Int64,
    u64,
    hash_strided_array64,
    enc3::encode_block_strided_3d_i64_rate,
    enc3::encode_partial_block_strided_3d_i64_rate,
    dim3::decode_block_strided_3d_i64_rate,
    dim3::decode_partial_block_strided_3d_i64_rate
);

decode_block_strided_tests_4d!(
    dim4_double,
    f64,
    Rand64::new,
    next_signed_f64,
    enc4::encode_block_strided_4d_f64,
    enc4::encode_partial_block_strided_4d_f64,
    dim4::decode_block_strided_4d_f64,
    dim4::decode_partial_block_strided_4d_f64,
    ZfpScalarType::Double,
    u64,
    hash_strided_array64,
    enc4::encode_block_strided_4d_f64_rate,
    enc4::encode_partial_block_strided_4d_f64_rate,
    dim4::decode_block_strided_4d_f64_rate,
    dim4::decode_partial_block_strided_4d_f64_rate,
    ZFP_MIN_EXP
);
decode_block_strided_tests_4d!(
    dim4_float,
    f32,
    Rand32::new,
    next_signed_f32,
    enc4::encode_block_strided_4d_f32,
    enc4::encode_partial_block_strided_4d_f32,
    dim4::decode_block_strided_4d_f32,
    dim4::decode_partial_block_strided_4d_f32,
    ZfpScalarType::Float,
    u32,
    hash_strided_array32,
    enc4::encode_block_strided_4d_f32_rate,
    enc4::encode_partial_block_strided_4d_f32_rate,
    dim4::decode_block_strided_4d_f32_rate,
    dim4::decode_partial_block_strided_4d_f32_rate,
    ZFP_MIN_EXP
);
decode_block_strided_tests_4d!(
    dim4_int32,
    i32,
    Rand32::new,
    next_signed_int,
    enc4::encode_block_strided_4d_i32,
    enc4::encode_partial_block_strided_4d_i32,
    dim4::decode_block_strided_4d_i32,
    dim4::decode_partial_block_strided_4d_i32,
    ZfpScalarType::Int32,
    u32,
    hash_strided_array32,
    enc4::encode_block_strided_4d_i32_rate,
    enc4::encode_partial_block_strided_4d_i32_rate,
    dim4::decode_block_strided_4d_i32_rate,
    dim4::decode_partial_block_strided_4d_i32_rate
);
decode_block_strided_tests_4d!(
    dim4_int64,
    i64,
    Rand64::new,
    next_signed_int,
    enc4::encode_block_strided_4d_i64,
    enc4::encode_partial_block_strided_4d_i64,
    dim4::decode_block_strided_4d_i64,
    dim4::decode_partial_block_strided_4d_i64,
    ZfpScalarType::Int64,
    u64,
    hash_strided_array64,
    enc4::encode_block_strided_4d_i64_rate,
    enc4::encode_partial_block_strided_4d_i64_rate,
    dim4::decode_block_strided_4d_i64_rate,
    dim4::decode_partial_block_strided_4d_i64_rate
);
