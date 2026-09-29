use zfp_rs::{
    ZfpBitStream, ZfpBitStreamMutOps, ZfpBitStreamRefMut, ZfpCompressionError, ZfpConfig,
    ZfpConfigError, ZfpField, ZfpFieldMut, ZfpHeaderMask,
};

const INPUT: [i32; 8] = [1, 2, 3, 4, 5, 6, 7, 8];

fn roundtrip_two_blocks(bs: &mut impl ZfpBitStreamMutOps, budget: u32) {
    let config = ZfpConfig::try_expert(budget, budget, 64, -1075).unwrap();
    let field = ZfpField::new(&INPUT, [8usize]).unwrap();
    assert!(config.checked_mode_bits().is_ok());
    bs.write_header(&config, &field.metadata(), ZfpHeaderMask::FULL)
        .unwrap();
    let written = bs.compress(&config, &field).unwrap();

    bs.rewind();
    let header = bs.read_header(ZfpHeaderMask::FULL).unwrap();
    let decoded_config = header.config.unwrap();
    assert_eq!(decoded_config, config);
    let mut from_header = [0; 8];
    let mut output = ZfpFieldMut::new(&mut from_header, [8usize]).unwrap();
    let consumed = bs.decompress(&decoded_config, &mut output).unwrap();
    assert_eq!(consumed, written);
    assert_eq!(bs.read_pos(), (written * 8) as u64);

    bs.rewind();
    bs.read_header(ZfpHeaderMask::FULL).unwrap();
    let mut from_original = [0; 8];
    let mut output = ZfpFieldMut::new(&mut from_original, [8usize]).unwrap();
    assert_eq!(bs.decompress(&config, &mut output).unwrap(), written);
    assert_eq!(from_header, from_original);
    if budget == 32768 {
        assert_eq!(from_header, INPUT);
    }
}

fn reject_without_writing(bs: &mut impl ZfpBitStreamMutOps, config: ZfpConfig) {
    let field = ZfpField::new(&INPUT, [8usize]).unwrap();
    bs.write_bits(0x5a, 8);
    let pos = bs.write_pos();
    let words = bs.backing_words().to_vec();
    assert_eq!(
        bs.write_header(&config, &field.metadata(), ZfpHeaderMask::FULL),
        Err(ZfpCompressionError::Config(
            ZfpConfigError::UnrepresentableMode
        ))
    );
    assert_eq!(bs.write_pos(), pos);
    assert_eq!(bs.backing_words(), words);
    bs.write_bits(0xc3, 8);
    bs.flush();
    bs.rewind();
    assert_eq!(bs.read_bits(16), 0xc35a);
}

fn roundtrip_mode_header(bs: &mut impl ZfpBitStreamMutOps, config: ZfpConfig, bits: usize) {
    let field = ZfpField::new(&INPUT, [8usize]).unwrap();
    assert_eq!(
        bs.write_header(&config, &field.metadata(), ZfpHeaderMask::MODE),
        Ok(bits)
    );
    bs.flush();
    bs.rewind();
    let header = bs.read_header(ZfpHeaderMask::MODE).unwrap();
    assert_eq!(header.bits_read, bits);
    assert_eq!(header.config, Some(config));
    assert_eq!(bs.read_pos(), bits as u64);
}

#[test]
fn mode_budget_boundaries_on_owned_and_borrowed_streams() {
    for budget in [1, 32768] {
        let mut owned = ZfpBitStream::new(12000);
        roundtrip_two_blocks(&mut owned, budget);
        let mut words = vec![0u64; 1500];
        let mut borrowed = ZfpBitStreamRefMut::from_words(&mut words);
        roundtrip_two_blocks(&mut borrowed, budget);
    }

    for budget in [0, 32769, 40000] {
        let config = ZfpConfig::try_expert(budget, budget, 64, -1075).unwrap();
        assert_eq!(
            config.checked_mode_bits(),
            Err(ZfpConfigError::UnrepresentableMode)
        );
        let mut owned = ZfpBitStream::new(64);
        reject_without_writing(&mut owned, config);
        let mut words = [0u64; 8];
        let mut borrowed = ZfpBitStreamRefMut::from_words(&mut words);
        reject_without_writing(&mut borrowed, config);
    }
}

#[test]
fn mode_exponent_boundaries_and_codec_validation() {
    for exponent in [-16495, -1075, -1074, 843, 844, 16272] {
        let config = ZfpConfig::try_expert(2, 100, 30, exponent).unwrap();
        assert!(config.checked_mode_bits().is_ok());
        let mut owned = ZfpBitStream::new(16);
        roundtrip_mode_header(&mut owned, config, 64);
        let mut words = [0u64; 2];
        let mut borrowed = ZfpBitStreamRefMut::from_words(&mut words);
        roundtrip_mode_header(&mut borrowed, config, 64);
    }
    for exponent in [-16496, 16273, i32::MIN, i32::MAX] {
        let config = ZfpConfig::try_expert(2, 100, 30, exponent).unwrap();
        assert_eq!(
            config.checked_mode_bits(),
            Err(ZfpConfigError::UnrepresentableMode)
        );
        let mut owned = ZfpBitStream::new(16);
        reject_without_writing(&mut owned, config);
        let mut words = [0u64; 2];
        let mut borrowed = ZfpBitStreamRefMut::from_words(&mut words);
        reject_without_writing(&mut borrowed, config);
    }
    for (exponent, bits) in [(-1075, 12), (-1074, 64), (843, 12), (844, 64)] {
        let config = ZfpConfig::try_expert(1, 16658, 64, exponent).unwrap();
        let mut owned = ZfpBitStream::new(16);
        roundtrip_mode_header(&mut owned, config, bits);
    }
    assert_eq!(
        ZfpConfig::expert(4, 3, 64, -1075).checked_mode_bits(),
        Err(ZfpConfigError::InvalidParameters)
    );
}

#[test]
fn headers_without_mode_do_not_require_representable_config() {
    let config = ZfpConfig::expert(40000, 40000, 64, -1075);
    let field = ZfpField::new(&INPUT, [8usize]).unwrap();
    let mut bs = ZfpBitStream::new(32);
    assert_eq!(
        bs.write_header(&config, &field.metadata(), ZfpHeaderMask::MAGIC),
        Ok(32)
    );
}
