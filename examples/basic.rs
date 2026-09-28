//! Basic compress/decompress example.
//!
//! ```text
//! $ cargo run --example basic --features ffi
//! Compressed: 64 bytes
//! Original data: [0.0, 1.0, 2.0, 3.0, 4.0]
//! Decompressed: [0.0, 1.0, 2.0, 3.0, 4.0]
//! Match: true
//! ```

use zfp_rs::{ZfpBitStream, ZfpConfig, ZfpField, ZfpFieldMut, ZfpStreamAlignment};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let data: Vec<f64> = (0..5).map(f64::from).collect();
    println!("Original data: {data:?}");

    let field = ZfpField::new(&data, [data.len()])?;

    // Configure compression parameters.
    let config = ZfpConfig::fixed_rate(
        8.0,
        zfp_rs::ZfpScalarType::F64,
        zfp_rs::ZfpDimensionality::D1,
        ZfpStreamAlignment::Unaligned,
    );

    // Allocate a bitstream for the compressed output.
    let mut bs = ZfpBitStream::new(1024);

    // Compress.
    let bytes = bs.compress(&config, &field)?;
    println!("Compressed: {bytes} bytes");

    // Decompress into a new field.
    let mut out_data = vec![0.0_f64; data.len()];
    let mut output = ZfpFieldMut::new(&mut out_data, [data.len()])?;

    bs.rewind();
    let decompressed = bs.decompress(&config, &mut output)?;
    println!("Decompressed: {decompressed} bytes");

    let decompressed_data: &[f64] = bytemuck::cast_slice(output.data());
    println!("Decompressed: {decompressed_data:?}");

    // Compare.
    let match_ = data
        .iter()
        .zip(decompressed_data.iter())
        .all(|(a, b)| (*a - *b).abs() < 1e-6);
    println!("Match: {match_}");
    Ok(())
}
