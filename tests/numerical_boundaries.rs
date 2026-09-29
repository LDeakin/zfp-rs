use zfp_rs::{ZfpBitStream, ZfpConfig, ZfpField, ZfpFieldMut, ZfpScalarType};

fn encode_and_decode(min_exp: i32) -> (Vec<u8>, [f64; 4]) {
    let input = [1e-100f64; 4];
    let config = ZfpConfig::try_expert(1, 16658, 64, min_exp).unwrap();
    let field = ZfpField::new(&input, [4usize]).unwrap();
    let capacity = config.maximum_size(ZfpScalarType::F64, [4usize]).unwrap();
    let mut stream = ZfpBitStream::new(capacity);
    stream.compress(&config, &field).unwrap();
    let encoded = stream.as_bytes().to_vec();
    stream.rewind();
    let mut output = [0.0f64; 4];
    let mut output_field = ZfpFieldMut::new(&mut output, [4usize]).unwrap();
    stream.decompress(&config, &mut output_field).unwrap();
    (encoded, output)
}

#[test]
fn extreme_expert_exponents_do_not_wrap_precision() {
    let (default_encoded, default_output) = encode_and_decode(-1075);
    let (low_encoded, low_output) = encode_and_decode(i32::MIN);
    assert_eq!(low_encoded, default_encoded);
    assert_eq!(low_output, default_output);

    let (high_encoded, high_output) = encode_and_decode(i32::MAX);
    assert_eq!(high_encoded.len(), 8);
    assert_eq!(high_output, [0.0; 4]);
}
