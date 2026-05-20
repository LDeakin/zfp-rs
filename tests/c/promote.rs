#![allow(clippy::cast_possible_truncation)] // usize→i8/u8/i16/u16 for test constants
#![allow(clippy::cast_possible_wrap)] // usize→i16 for test constants
//! Port of `zfp/tests/src/misc/testZfpPromote.c`.

use zfp_rs::{ZfpDimensionality, codec::promote};

/// Block element count for dims=3: 1 << (2*3) = 64.
const DIMS: ZfpDimensionality = ZfpDimensionality::D3;
const SZ: usize = 1 << (2 * (DIMS as u32));

#[test]
fn given_int8_when_promote_to_int32_expect_demote_to_int8_matches() {
    let iblock: Vec<i8> = (0..SZ as i8).collect();
    let mut block32 = vec![0i32; SZ];
    let mut oblock: Vec<i8> = vec![0i8; SZ];

    promote::promote_i8_to_i32(&mut block32, &iblock, DIMS);
    promote::demote_i32_to_i8(&mut oblock, &block32, DIMS);

    for i in 0..SZ {
        assert_eq!(iblock[i], oblock[i], "mismatch at index {i}");
    }
}

#[test]
fn given_uint8_when_promote_to_int32_expect_demote_to_uint8_matches() {
    let iblock: Vec<u8> = (0..SZ as u8).collect();
    let mut block32 = vec![0i32; SZ];
    let mut oblock: Vec<u8> = vec![0u8; SZ];

    promote::promote_u8_to_i32(&mut block32, &iblock, DIMS);
    promote::demote_i32_to_u8(&mut oblock, &block32, DIMS);

    for i in 0..SZ {
        assert_eq!(iblock[i], oblock[i], "mismatch at index {i}");
    }
}

#[test]
fn given_int16_when_promote_to_int32_expect_demote_to_int16_matches() {
    let iblock: Vec<i16> = (0..SZ as i16).collect();
    let mut block32 = vec![0i32; SZ];
    let mut oblock: Vec<i16> = vec![0i16; SZ];

    promote::promote_i16_to_i32(&mut block32, &iblock, DIMS);
    promote::demote_i32_to_i16(&mut oblock, &block32, DIMS);

    for i in 0..SZ {
        assert_eq!(iblock[i], oblock[i], "mismatch at index {i}");
    }
}

#[test]
fn given_uint16_when_promote_to_int32_expect_demote_to_uint16_matches() {
    let iblock: Vec<u16> = (0..SZ as u16).collect();
    let mut block32 = vec![0i32; SZ];
    let mut oblock: Vec<u16> = vec![0u16; SZ];

    promote::promote_u16_to_i32(&mut block32, &iblock, DIMS);
    promote::demote_i32_to_u16(&mut oblock, &block32, DIMS);

    for i in 0..SZ {
        assert_eq!(iblock[i], oblock[i], "mismatch at index {i}");
    }
}
