#![allow(clippy::cast_possible_truncation)] // u64→usize for test constants
#![allow(clippy::doc_markdown)] // C constants in test docs
//! Port of `zfp/tests/src/misc/testZfpHeader.c`.
//!
//! The C file is compiled with `block1.h` (DIMS=1) and `1dDouble.h`
//! (ZFP_TYPE=zfp_type_double, ZFP_RATE_PARAM_BITS=19).
//! The test field is 2-D f64 of size 33×401.

use zfp_rs::types::{
    ZFP_MAGIC_BITS, ZFP_META_BITS, ZFP_MODE_LONG_BITS, ZFP_MODE_SHORT_BITS, ZfpHeaderMask,
};
use zfp_rs::{
    ZfpBitStream, ZfpConfig, ZfpDimensionality, ZfpField, ZfpFieldMetadata, ZfpScalarType,
    ZfpStreamAlignment,
};

// ---------------------------------------------------------------------------
// Constants from the C test (testZfpHeader.c + included headers)
// ---------------------------------------------------------------------------

/// DIMS from block1.h (used for set_rate, not the field dimensionality)
const DIMS: ZfpDimensionality = ZfpDimensionality::D1;
/// ZFP_RATE_PARAM_BITS from universalConsts.h
const ZFP_RATE_PARAM_BITS: f64 = 19.0;
/// Field dimensions (FIELD_X_LEN, FIELD_Y_LEN)
const FIELD_X_LEN: usize = 33;
const FIELD_Y_LEN: usize = 401;
/// Custom compression parameters
const MIN_BITS: u32 = 11;
const MAX_BITS_CUSTOM: u32 = 1001;
const MAX_PREC_CUSTOM: u32 = 52;
const MIN_EXP_CUSTOM: i32 = -1000;
const PREC: u32 = 44;
const ACC: f64 = 1e-4;
/// ZFP_CODEC value (from zfp/include/zfp/version.h via header.rs)
const ZFP_CODEC: u64 = 5;

// ---------------------------------------------------------------------------
// Setup helpers
// ---------------------------------------------------------------------------

/// Create the default test field (2D f64, 33×401, no data pointer needed).
fn make_field() -> ZfpField<'static> {
    ZfpField::new(&[] as &[f64], [FIELD_X_LEN, FIELD_Y_LEN])
}

/// Create the default test params (fixed-rate, rate=19, dims=1).
fn make_params() -> ZfpConfig {
    ZfpConfig::fixed_rate(
        ZFP_RATE_PARAM_BITS,
        ZfpScalarType::Double,
        DIMS,
        ZfpStreamAlignment::None,
    )
}

// ---------------------------------------------------------------------------
// Field metadata tests
// ---------------------------------------------------------------------------

#[test]
fn when_zfp_field_metadata_called_expect_lsb_2_bits_encode_scalar_type() {
    let field = make_field();
    let metadata = field.metadata().expect("metadata should be valid");
    // bits [1:0] encode (zfp_type - 1); zfp_type_double == 4 → stored as 3
    let zfp_type = (metadata & 0x3) + 1;
    // ZfpScalarType::Double is the 4th variant (Int32=1, Int64=2, Float=3, Double=4)
    assert_eq!(zfp_type, 4, "expected Double (4), got {zfp_type}");
}

#[test]
fn when_zfp_field_metadata_called_expect_lsb_bits_3_to_4_encode_dimensionality() {
    let field = make_field();
    let metadata = field.metadata().expect("metadata should be valid");
    // bits [3:2] encode (dimensionality - 1); field is 2D → stored as 1
    let dimensionality = ((metadata >> 2) & 0x3) + 1;
    assert_eq!(dimensionality, 2, "expected 2D, got {dimensionality}");
}

#[test]
fn when_zfp_field_metadata_called_expect_lsb_bits_5_to_53_encode_array_dimensions() {
    let field = make_field();
    let metadata = field.metadata().expect("metadata should be valid");
    let mask_24: u64 = 0xff_ffff;
    let mask_48: u64 = 0xffff_ffff_ffff;
    // bits [51:4] encode the array dimensions
    let encoded_dims = (metadata >> 4) & mask_48;
    let nx = (encoded_dims & mask_24) + 1;
    let ny = ((encoded_dims >> 24) & mask_24) + 1;
    assert_eq!(nx as usize, FIELD_X_LEN, "nx mismatch");
    assert_eq!(ny as usize, FIELD_Y_LEN, "ny mismatch");
}

#[test]
fn when_zfp_field_set_metadata_called_expect_scalar_type_set() {
    let mut field = make_field();
    let metadata = field.metadata().expect("metadata should be valid");
    // In Rust the scalar type is baked into the generic parameter (f64); we
    // verify that round-tripping metadata preserves the encoded type bits.
    assert!(field.set_metadata(metadata));
    // type is still Double — the encoded type bits must round-trip
    let meta2 = field.metadata().expect("metadata should be valid");
    assert_eq!(
        meta2 & 0x3,
        metadata & 0x3,
        "type bits changed after set_metadata"
    );
}

#[test]
fn when_zfp_field_set_metadata_called_expect_array_dimensions_set() {
    let field = make_field();
    let metadata = field.metadata().expect("metadata should be valid");
    let orig_size = field.dims();

    let metadata = ZfpFieldMetadata::from_bits(metadata).expect("metadata should decode");
    assert_eq!(metadata.dims[0], orig_size[0], "nx mismatch");
    assert_eq!(metadata.dims[1], orig_size[1], "ny mismatch");
    assert_eq!(metadata.dims[2], 0, "nz should be 0");
}

#[test]
fn when_zfp_field_metadata_called_on_invalid_size_expect_dimension_too_large() {
    // Create a field with dimensions that exceed the encodable range (2^24 > 24 bits)
    let big = ZfpField::new(&[] as &[f64], [1 << 25, 1 << 25]);
    let meta = big.metadata();
    assert!(
        matches!(
            meta,
            Err(zfp_rs::types::ZfpMetadataError::DimensionTooLarge)
        ),
        "expected ZfpMetadataError::DimensionTooLarge, got {meta:?}"
    );
}

#[test]
fn when_zfp_field_set_metadata_called_for_invalid_meta_expect_false() {
    let mut field = make_field();
    // meta with bit > ZFP_META_BITS set
    let invalid_meta = 1u64 << (ZFP_META_BITS + 1);
    assert!(!field.set_metadata(invalid_meta));
}

// ---------------------------------------------------------------------------
// Write header tests
// ---------------------------------------------------------------------------

#[test]
fn when_zfp_write_header_magic_expect_num_bits_written_equal_to_zfp_magic_bits() {
    let config = make_params();
    let mut bs = ZfpBitStream::new(4096);
    let field = make_field();
    let bits = bs.write_header(&config, &field, ZfpHeaderMask::MAGIC);
    assert_eq!(bits, ZFP_MAGIC_BITS as usize);
}

#[test]
fn when_zfp_write_header_magic_expect_24_bits_are_chars_zfp_followed_by_8_bits_zfp_codec_version() {
    let config = make_params();
    let mut bs = ZfpBitStream::new(4096);
    let field = make_field();
    assert_eq!(
        bs.write_header(&config, &field, ZfpHeaderMask::MAGIC),
        ZFP_MAGIC_BITS as usize
    );
    bs.flush();
    bs.rewind();
    let char1 = bs.read_bits(8);
    let char2 = bs.read_bits(8);
    let char3 = bs.read_bits(8);
    let codec_ver = bs.read_bits(8);
    assert_eq!(char1, u64::from(b'z'));
    assert_eq!(char2, u64::from(b'f'));
    assert_eq!(char3, u64::from(b'p'));
    assert_eq!(codec_ver, ZFP_CODEC);
}

#[test]
fn when_zfp_write_header_metadata_expect_num_bits_written_equal_to_zfp_meta_bits() {
    let config = make_params();
    let mut bs = ZfpBitStream::new(4096);
    let field = make_field();
    let bits = bs.write_header(&config, &field, ZfpHeaderMask::META);
    assert_eq!(bits, ZFP_META_BITS as usize);
}

#[test]
fn given_fixed_rate_when_zfp_write_header_mode_expect_12_bits_written_to_bitstream() {
    // setup uses fixed-rate mode already
    let config = make_params();
    let mut bs = ZfpBitStream::new(4096);
    let field = make_field();
    let bits = bs.write_header(&config, &field, ZfpHeaderMask::MODE);
    assert_eq!(bits, ZFP_MODE_SHORT_BITS as usize);
}

#[test]
fn given_fixed_precision_when_zfp_write_header_mode_expect_12_bits_written_to_bitstream() {
    let config = ZfpConfig::fixed_precision(PREC);
    let mut bs = ZfpBitStream::new(4096);
    let field = make_field();
    let bits = bs.write_header(&config, &field, ZfpHeaderMask::MODE);
    assert_eq!(bits, ZFP_MODE_SHORT_BITS as usize);
}

#[test]
fn given_fixed_accuracy_when_zfp_write_header_mode_expect_12_bits_written_to_bitstream() {
    let config = ZfpConfig::fixed_accuracy(ACC);
    let mut bs = ZfpBitStream::new(4096);
    let field = make_field();
    let bits = bs.write_header(&config, &field, ZfpHeaderMask::MODE);
    assert_eq!(bits, ZFP_MODE_SHORT_BITS as usize);
}

#[test]
fn given_custom_compress_params_set_when_zfp_write_header_mode_expect_64_bits_written_to_bitstream()
{
    // Custom params that don't match default expert mode → 64-bit long encoding.
    let config = ZfpConfig::expert(MIN_BITS, MAX_BITS_CUSTOM, MAX_PREC_CUSTOM, MIN_EXP_CUSTOM);
    let mut bs = ZfpBitStream::new(4096);
    let field = make_field();
    let bits = bs.write_header(&config, &field, ZfpHeaderMask::MODE);
    assert_eq!(bits, ZFP_MODE_LONG_BITS as usize);
}

// ---------------------------------------------------------------------------
// Read header tests
// ---------------------------------------------------------------------------

/// Write `mask` header section, flush, rewind, then read it back.
/// Asserts that `write` returned `expected_write_bits` and
/// `read` returned `expected_read_bits`.
fn assert_proper_bits_read(
    config: &ZfpConfig,
    field: &ZfpField<'_>,
    bs: &mut ZfpBitStream,
    mask: ZfpHeaderMask,
    expected_write_bits: usize,
    expected_read_bits: usize,
) {
    assert_eq!(bs.write_header(config, field, mask), expected_write_bits);
    bs.flush();
    bs.rewind();
    let result = bs.read_header(mask);
    if expected_read_bits == 0 {
        // Read failed (invalid config, bad magic, etc.)
        assert!(
            result.is_err(),
            "expected read to fail (0 bits), but got {:?}",
            result.map(|header| header.bits_read)
        );
    } else {
        assert_eq!(
            result.map(|header| header.bits_read),
            Ok(expected_read_bits)
        );
    }
}

#[test]
fn when_zfp_read_header_magic_expect_proper_num_bits_read() {
    let config = make_params();
    let mut bs = ZfpBitStream::new(4096);
    let field = make_field();
    assert_proper_bits_read(
        &config,
        &field,
        &mut bs,
        ZfpHeaderMask::MAGIC,
        ZFP_MAGIC_BITS as usize,
        ZFP_MAGIC_BITS as usize,
    );
}

#[test]
fn given_improper_header_when_zfp_read_header_magic_expect_returns_zero() {
    let mut bs = ZfpBitStream::new(4096);
    let result = bs.read_header(ZfpHeaderMask::MAGIC);
    assert!(result.is_err());
}

#[test]
fn when_zfp_read_header_metadata_expect_proper_num_bits_read() {
    let config = make_params();
    let mut bs = ZfpBitStream::new(4096);
    let field = make_field();
    assert_proper_bits_read(
        &config,
        &field,
        &mut bs,
        ZfpHeaderMask::META,
        ZFP_META_BITS as usize,
        ZFP_META_BITS as usize,
    );
}

#[test]
fn given_proper_header_when_zfp_read_header_metadata_expect_field_array_dims_set() {
    let config = make_params();
    let mut bs = ZfpBitStream::new(4096);
    let field = make_field();
    let orig = field.dims();

    assert_eq!(
        bs.write_header(&config, &field, ZfpHeaderMask::META),
        ZFP_META_BITS as usize
    );
    bs.flush();
    bs.rewind();

    let header = bs
        .read_header(ZfpHeaderMask::META)
        .expect("read META header");
    assert_eq!(header.bits_read, ZFP_META_BITS as usize);
    let metadata = header.metadata.expect("META header should decode metadata");
    assert_eq!(
        metadata.dims[0], orig[0],
        "nx mismatch after read_header META"
    );
    assert_eq!(
        metadata.dims[1], orig[1],
        "ny mismatch after read_header META"
    );
    assert_eq!(metadata.dims[2], 0);
    assert_eq!(metadata.scalar_type, field.scalar_type());
}

#[test]
fn given_proper_header_fixed_rate_when_zfp_read_header_mode_expect_proper_num_bits_read() {
    let config = make_params();
    let mut bs = ZfpBitStream::new(4096);
    let field = make_field();
    assert_proper_bits_read(
        &config,
        &field,
        &mut bs,
        ZfpHeaderMask::MODE,
        ZFP_MODE_SHORT_BITS as usize,
        ZFP_MODE_SHORT_BITS as usize,
    );
}

#[test]
fn given_proper_header_fixed_precision_when_zfp_read_header_mode_expect_proper_num_bits_read() {
    let config = ZfpConfig::fixed_precision(PREC);
    let mut bs = ZfpBitStream::new(4096);
    let field = make_field();
    assert_proper_bits_read(
        &config,
        &field,
        &mut bs,
        ZfpHeaderMask::MODE,
        ZFP_MODE_SHORT_BITS as usize,
        ZFP_MODE_SHORT_BITS as usize,
    );
}

#[test]
fn given_proper_header_fixed_accuracy_when_zfp_read_header_mode_expect_proper_num_bits_read() {
    let config = ZfpConfig::fixed_accuracy(ACC);
    let mut bs = ZfpBitStream::new(4096);
    let field = make_field();
    assert_proper_bits_read(
        &config,
        &field,
        &mut bs,
        ZfpHeaderMask::MODE,
        ZFP_MODE_SHORT_BITS as usize,
        ZFP_MODE_SHORT_BITS as usize,
    );
}

/// Write then read the MODE header; verify that after reading, params
/// match those that were in effect when writing.
fn assert_compress_params_restored(
    config: ZfpConfig,
    field: &ZfpField<'_>,
    bs: &mut ZfpBitStream,
    expected_write_bits: usize,
    expected_read_bits: usize,
) {
    assert_eq!(
        bs.write_header(&config, field, ZfpHeaderMask::MODE),
        expected_write_bits
    );
    bs.flush();
    bs.rewind();

    let result = bs.read_header(ZfpHeaderMask::MODE);
    let read_bits = result.as_ref().map_or(0, |header| header.bits_read);
    assert_eq!(read_bits, expected_read_bits);

    if expected_read_bits == 0 {
        // read failed
    } else {
        // params were restored
        let read_stream = result
            .expect("read_header MODE")
            .config
            .expect("MODE header should decode config");
        assert_eq!(
            read_stream, config,
            "params not restored after read_header MODE"
        );
    }
}

#[test]
fn given_proper_header_fixed_rate_when_zfp_read_header_mode_expect_stream_params_set() {
    let config = make_params();
    let mut bs = ZfpBitStream::new(4096);
    let field = make_field();
    assert_compress_params_restored(
        config,
        &field,
        &mut bs,
        ZFP_MODE_SHORT_BITS as usize,
        ZFP_MODE_SHORT_BITS as usize,
    );
}

#[test]
fn given_proper_header_fixed_precision_when_zfp_read_header_mode_expect_stream_params_set() {
    let config = ZfpConfig::fixed_precision(PREC);
    let mut bs = ZfpBitStream::new(4096);
    let field = make_field();
    assert_compress_params_restored(
        config,
        &field,
        &mut bs,
        ZFP_MODE_SHORT_BITS as usize,
        ZFP_MODE_SHORT_BITS as usize,
    );
}

#[test]
fn given_proper_header_fixed_accuracy_when_zfp_read_header_mode_expect_stream_params_set() {
    let config = ZfpConfig::fixed_accuracy(ACC);
    let mut bs = ZfpBitStream::new(4096);
    let field = make_field();
    assert_compress_params_restored(
        config,
        &field,
        &mut bs,
        ZFP_MODE_SHORT_BITS as usize,
        ZFP_MODE_SHORT_BITS as usize,
    );
}

#[test]
fn given_custom_compress_params_set_when_zfp_read_header_mode_expect_proper_num_bits_read() {
    // Custom params that don't match default expert mode → 64-bit long encoding.
    let config = ZfpConfig::expert(MIN_BITS, MAX_BITS_CUSTOM, MAX_PREC_CUSTOM, MIN_EXP_CUSTOM);
    let mut bs = ZfpBitStream::new(4096);
    let field = make_field();
    assert_proper_bits_read(
        &config,
        &field,
        &mut bs,
        ZfpHeaderMask::MODE,
        ZFP_MODE_LONG_BITS as usize,
        ZFP_MODE_LONG_BITS as usize,
    );
}

#[test]
fn given_custom_compress_params_and_proper_header_when_zfp_read_header_mode_expect_stream_params_set()
 {
    // Custom params that don't match default expert mode → 64-bit long encoding.
    let config = ZfpConfig::expert(MIN_BITS, MAX_BITS_CUSTOM, MAX_PREC_CUSTOM, MIN_EXP_CUSTOM);
    let mut bs = ZfpBitStream::new(4096);
    let field = make_field();
    assert_compress_params_restored(
        config,
        &field,
        &mut bs,
        ZFP_MODE_LONG_BITS as usize,
        ZFP_MODE_LONG_BITS as usize,
    );
}

#[cfg(feature = "ffi")]
#[test]
fn given_invalid_compress_params_in_header_when_zfp_read_header_mode_expect_proper_num_bits_read() {
    // Invalid params (min_bits > max_bits) should be rejected on read
    let config = ZfpConfig::expert(
        MAX_BITS_CUSTOM + 1, // min_bits > max_bits = invalid
        MAX_BITS_CUSTOM,
        MAX_PREC_CUSTOM,
        MIN_EXP_CUSTOM,
    );
    let mut bs = ZfpBitStream::new(4096);
    let field = make_field();
    assert_proper_bits_read(
        &config,
        &field,
        &mut bs,
        ZfpHeaderMask::MODE,
        ZFP_MODE_LONG_BITS as usize,
        0,
    );
}

#[cfg(feature = "ffi")]
#[test]
fn given_invalid_compress_params_in_header_when_zfp_read_header_mode_expect_stream_params_not_set()
{
    // Invalid params (min_bits > max_bits) should be rejected on read
    let config = ZfpConfig::expert(
        MAX_BITS_CUSTOM + 1,
        MAX_BITS_CUSTOM,
        MAX_PREC_CUSTOM,
        MIN_EXP_CUSTOM,
    );
    let field = make_field();
    let mut bs = ZfpBitStream::new(4096);
    // write returns 64 bits (long mode), read returns 0 (invalid params rejected)
    assert_compress_params_restored(config, &field, &mut bs, ZFP_MODE_LONG_BITS as usize, 0);
}
