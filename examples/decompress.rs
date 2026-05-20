//! Decompress example: reads a compressed bitstream and decompresses it.
//!
//! The example demonstrates the borrow-based API: the `ZfpBitStream` is
//! allocated separately and passed to `decompress()` by mutable reference.

use zfp_rs::{ZfpBitStream, ZfpConfig, ZfpField, ZfpFieldMut, ZfpStreamAlignment};

fn main() {
    // Create some data.
    let data: Vec<f32> = (0..64).map(|i| i as f32 * 0.5).collect();
    println!("Original data: {} elements", data.len());

    // Compress first (simulating reading from a file).
    let compress_stream = ZfpConfig::fixed_rate(
        8.0,
        zfp_rs::ZfpScalarType::Float,
        zfp_rs::ZfpDimensionality::D1,
        ZfpStreamAlignment::None,
    );

    let field = ZfpField::new(&data, [data.len()]);
    let mut bs = ZfpBitStream::new(1024);
    let _compressed_bytes = bs
        .compress(&compress_stream, &field)
        .expect("compress failed");
    println!("Compressed: {} bytes", bs.size());

    // Now "read" the compressed data back.
    let config = ZfpConfig::new();

    // Decompress into a new field.
    let mut out_data = vec![0.0_f32; data.len()];
    let mut output = ZfpFieldMut::new(&mut out_data, [data.len()]);

    bs.rewind();
    let _decompressed = bs
        .decompress(&config, &mut output)
        .expect("decompress failed");

    let result: &[f32] = bytemuck::cast_slice(output.data());
    println!("Decompressed: {} elements", result.len());

    let match_ = data
        .iter()
        .zip(result.iter())
        .all(|(a, b)| (*a - *b).abs() < 1e-5);
    println!("Match: {match_}");
}
