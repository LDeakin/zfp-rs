//! 3-D compress/decompress example.
//!
//! ```text
//! $ cargo run --example 3d --features ffi
//! ```

use zfp_rs::{ZfpBitStream, ZfpConfig, ZfpField, ZfpFieldMut, ZfpStreamAlignment};

fn main() {
    // Create a 4x4x4 3D grid with smooth data.
    let size = 4;
    let mut data: Vec<f64> = Vec::with_capacity(size * size * size);
    for z in 0..size {
        for y in 0..size {
            for x in 0..size {
                data.push((x + y + z) as f64 / (size as f64));
            }
        }
    }
    println!("Original data: {} elements", data.len());

    let field = ZfpField::new(&data, [size, size, size]);

    // Configure compression.
    let config = ZfpConfig::fixed_rate(
        8.0,
        zfp_rs::ZfpScalarType::Double,
        zfp_rs::ZfpDimensionality::D3,
        ZfpStreamAlignment::None,
    );

    let mut bs = ZfpBitStream::new(4096);

    let bytes = bs.compress(&config, &field).expect("compress failed");
    println!("Compressed: {bytes} bytes");

    // Decompress.
    let mut out_data = vec![0.0_f64; size * size * size];
    let mut output = ZfpFieldMut::new(&mut out_data, [size, size, size]);

    bs.rewind();
    let decompressed = bs
        .decompress(&config, &mut output)
        .expect("decompress failed");
    println!("Decompressed: {decompressed} bytes");

    let result: &[f64] = bytemuck::cast_slice(output.data());
    let match_ = data
        .iter()
        .zip(result.iter())
        .all(|(a, b)| (*a - *b).abs() < 1e-6);
    println!("Match: {match_}");
}
