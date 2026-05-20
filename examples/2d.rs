//! 2-D compress/decompress example.
//!
//! ```text
//! $ cargo run --example 2d --features ffi
//! ```

use zfp_rs::{ZfpBitStream, ZfpConfig, ZfpField, ZfpFieldMut, ZfpStreamAlignment};

fn main() {
    // Create a 10x10 grid.
    let width = 10;
    let height = 10;
    let mut data: Vec<f64> = Vec::with_capacity(width * height);
    for y in 0..height {
        for x in 0..width {
            data.push((x + y) as f64);
        }
    }
    println!("Original data: {} elements", data.len());

    let field = ZfpField::new(&data, [width, height]);

    // Configure compression.
    let config = ZfpConfig::fixed_rate(
        4.0,
        zfp_rs::ZfpScalarType::Double,
        zfp_rs::ZfpDimensionality::D2,
        ZfpStreamAlignment::None,
    );

    let mut bs = ZfpBitStream::new(4096);

    let bytes = bs.compress(&config, &field).expect("compress failed");
    println!("Compressed: {bytes} bytes");

    // Decompress.
    let mut out_data = vec![0.0_f64; width * height];
    let mut output = ZfpFieldMut::new(&mut out_data, [width, height]);

    bs.rewind();
    let decompressed = bs
        .decompress(&config, &mut output)
        .expect("decompress failed");
    println!("Decompressed: {decompressed} bytes");

    let result: &[f64] = bytemuck::cast_slice(output.data());
    let match_ = data
        .iter()
        .zip(result.iter())
        .all(|(a, b)| (*a - *b).abs() < 1e-3);
    println!("Match: {match_}");
}
