//! Regenerate the committed seed corpora under `fuzz/seeds/`.
//!
//! Run with `just fuzz_seeds`. Seeds are small, deterministic and committed, so
//! the CI smoke job gets a good starting point without depending on a cache.
//!
//! The `.proptest-regressions` files cannot be reused here: they store 32-byte
//! RNG seeds rather than values. The canned payloads below cover the same
//! ground (and then some, since subnormals are in scope for fuzzing).

use std::fs;
use std::path::{Path, PathBuf};

use zfp_fuzz_common::targets::decompress_stream::HEADER_LEN;
use zfp_rs::{ZfpBitStream, ZfpConfig, ZfpField, types::ZfpScalarType};

/// Payload shapes worth starting from, as raw bytes that `decode_scalars`
/// cycles into scalars.
const PAYLOADS: &[(&str, &[u8])] = &[
    ("zero", &[0x00]),
    ("ones", &[0xFF]),
    ("alternating", &[0xAA, 0x55]),
    ("ramp", &[0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07]),
    // f32/f64 exponent-only patterns: subnormal, NaN and infinity bit patterns
    // that the differential proptests must filter out but fuzzing wants.
    ("subnormal", &[0x01, 0x00, 0x00, 0x00]),
    ("nan", &[0xFF, 0xFF, 0xFF, 0x7F]),
    ("inf", &[0x00, 0x00, 0x80, 0x7F]),
];

fn main() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("fuzz")
        .join("seeds");

    let mut written = 0usize;
    written += gen_structured(&root.join("roundtrip"), PAYLOADS);
    // `config_mode` never looks at the payload, so one is enough: the useful
    // variation is entirely in the 12-byte prefix.
    written += gen_structured(&root.join("config_mode"), &PAYLOADS[..1]);
    written += gen_block_codec(&root.join("block_codec"));
    written += gen_decompress_stream(&root.join("decompress_stream"));
    written += gen_header_decode(&root.join("header_decode"));
    written += gen_bitstream_ops(&root.join("bitstream_ops"));

    println!("wrote {written} seed files under {}", root.display());
}

/// Seeds for the two targets sharing the 12-byte prefix
/// `kind, rank, 4 sides, mode family, 3 mode-param bytes, mode flags, exec`.
///
/// `roundtrip` derives its input with `#[derive(Arbitrary)]`, which consumes
/// bytes in field-declaration order from the front, producing exactly the same
/// layout that `config_mode` parses by hand.
fn gen_structured(dir: &Path, payloads: &[(&str, &[u8])]) -> usize {
    const PREFIX_LEN: usize = 12;
    let mut count = 0;
    for kind in 0u8..4 {
        for rank in 0u8..4 {
            for family in 0u8..5 {
                for (name, payload) in payloads {
                    let mut buf = vec![0u8; PREFIX_LEN];
                    buf[0] = kind;
                    buf[1] = rank;
                    // 6 elements per axis: `Shape::from_bytes` maps byte `b`
                    // to `1 + b % max_side`. Not a multiple of 4, so every
                    // axis ends in a partial block.
                    for side in &mut buf[2..6] {
                        *side = 5;
                    }
                    buf[6] = family;
                    // A mid-range mode parameter: rate 512 b/block, precision
                    // 32, min_exp near zero.
                    buf[7] = 0x20;
                    buf[8] = 0x02;
                    buf.extend_from_slice(payload);
                    write_seed(dir, &format!("k{kind}-r{rank}-m{family}-{name}"), &buf);
                    count += 1;
                }
            }
        }
    }
    count
}

/// `block_codec` framing: kind, rank, 4 stride bytes, 4 length bytes, mode.
///
/// The mode family deliberately excludes reversible (`family % 5 == 3`): the
/// target drives the `_with_params` entry points, which do not implement it,
/// and skips any input whose config has `min_exp < ZFP_MIN_EXP`. A seed in
/// that family would exercise nothing.
fn gen_block_codec(dir: &Path) -> usize {
    let mut count = 0;
    for kind in 0u8..4 {
        for rank in 0u8..4 {
            // Contiguous, gapped, and sign-flipped stride layouts.
            for (layout, stride_byte) in [("contig", 0u8), ("gap", 2), ("negative", 0x80)] {
                for full in [true, false] {
                    for (family, mode) in [
                        (0u8, "rate"),
                        (1, "precision"),
                        (2, "accuracy"),
                        (4, "expert"),
                    ] {
                        let mut buf = vec![0u8; 15];
                        buf[0] = kind;
                        buf[1] = rank;
                        for s in &mut buf[2..6] {
                            *s = stride_byte;
                        }
                        for l in &mut buf[6..10] {
                            *l = if full { 3 } else { 1 };
                        }
                        buf[10] = family;
                        // A mid-range mode parameter, matching `gen_structured`:
                        // rate 545 b/block, precision 33, min_exp near zero.
                        buf[11] = 0x20;
                        buf[12] = 0x02;
                        buf.extend_from_slice(&[0xAA, 0x55, 0x12, 0x34]);
                        let shape = if full { "full" } else { "partial" };
                        write_seed(
                            dir,
                            &format!("k{kind}-r{rank}-{layout}-{shape}-{mode}"),
                            &buf,
                        );
                        count += 1;
                    }
                }
            }
        }
    }
    count
}

/// The highest-value seeding: real compressed streams behind the target's
/// framing prefix, so the fuzzer starts from valid data it can then corrupt.
fn gen_decompress_stream(dir: &Path) -> usize {
    let mut count = 0;
    for (kind, ty) in [
        (0u8, ZfpScalarType::Int32),
        (1, ZfpScalarType::Int64),
        (2, ZfpScalarType::Float),
        (3, ZfpScalarType::Double),
    ] {
        for rank in 0u8..4 {
            for (family, config) in [
                (
                    0u8,
                    ZfpConfig::fixed_rate(8.0, ty, dimensionality(rank), Align),
                ),
                (1, ZfpConfig::fixed_precision(32)),
                (3, ZfpConfig::reversible()),
            ] {
                let dims = seed_dims(rank);
                let n: usize = dims[..=usize::from(rank)].iter().product();
                let cap = config.maximum_size(ty, &dims[..=usize::from(rank)]);
                if cap == 0 {
                    continue;
                }

                // Compress a simple ramp so the stream is structurally valid.
                let src: Vec<f64> = (0..n).map(|i| i as f64).collect();
                let bytes = compress_as(ty, &src, dims, &config, cap);

                let mut buf = vec![0u8; HEADER_LEN];
                buf[0] = kind;
                buf[1] = rank;
                for (axis, side) in buf[2..6].iter_mut().enumerate() {
                    *side = u8::try_from(dims[axis].saturating_sub(1)).unwrap_or(7);
                }
                buf[6] = family;
                buf[7] = 0x20;
                buf[8] = 0x02;
                buf.extend_from_slice(&bytes);

                write_seed(dir, &format!("k{kind}-r{rank}-m{family}-stream"), &buf);
                count += 1;
            }
        }
    }
    count
}

/// Real headers for every mask combination, so the fuzzer gets past
/// `InvalidMagic` immediately.
fn gen_header_decode(dir: &Path) -> usize {
    let mut count = 0;
    for selector in 0u8..8 {
        let mask = zfp_rs::types::ZfpHeaderMask::from_bits_truncate(u32::from(selector));
        let data = [1.0f64; 64];
        let field = ZfpField::new(&data, [4usize, 4, 4]);
        let config = ZfpConfig::reversible();

        let mut bs = ZfpBitStream::new(64);
        bs.write_header(&config, &field, mask);
        bs.flush();

        let mut buf = vec![selector];
        buf.extend_from_slice(bs.as_bytes());
        write_seed(dir, &format!("mask{selector}"), &buf);
        count += 1;
    }
    count
}

/// Short op sequences that reach the interesting cursor states quickly.
fn gen_bitstream_ops(dir: &Path) -> usize {
    let cases: &[(&str, &[u8])] = &[
        // write_bits(_, 3) then read_pos: the underflow shape.
        ("write-then-readpos", &[0, 3, 0, 7, 0, 0]),
        // Fill a word, flush, rewind, read it back.
        (
            "write-flush-rewind-read",
            &[2, 0xFF, 0xFF, 10, 0, 0, 11, 0, 0, 5, 0, 0],
        ),
        // Seek past the first word, write, seek back, read.
        (
            "seek-write-seek-read",
            &[6, 0x40, 0, 0, 8, 0x12, 7, 0x40, 0, 3, 8, 0],
        ),
        // Pad and align interactions.
        ("pad-align", &[9, 0x11, 0, 10, 0, 0, 8, 0x07, 0]),
    ];
    for (name, bytes) in cases {
        write_seed(dir, name, bytes);
    }
    cases.len()
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

use zfp_rs::ZfpStreamAlignment::None as Align;

fn dimensionality(rank: u8) -> zfp_rs::types::ZfpDimensionality {
    match rank {
        0 => zfp_rs::types::ZfpDimensionality::D1,
        1 => zfp_rs::types::ZfpDimensionality::D2,
        2 => zfp_rs::types::ZfpDimensionality::D3,
        _ => zfp_rs::types::ZfpDimensionality::D4,
    }
}

/// Small dims that stay partial-block in every rank.
///
/// 6 rather than 8: a multiple of 4 would tile exactly into 4^d blocks and
/// never reach the partial-block paths.
fn seed_dims(rank: u8) -> [usize; 4] {
    let mut dims = [0usize; 4];
    for dim in dims.iter_mut().take(usize::from(rank) + 1) {
        *dim = 6;
    }
    dims
}

fn compress_as(
    ty: ZfpScalarType,
    src: &[f64],
    dims: [usize; 4],
    config: &ZfpConfig,
    cap: usize,
) -> Vec<u8> {
    let mut bs = ZfpBitStream::new(cap);
    match ty {
        ZfpScalarType::Int32 => {
            let v: Vec<i32> = src.iter().map(|&x| x as i32).collect();
            let _ = bs.compress(config, &ZfpField::new(&v, dims));
        }
        ZfpScalarType::Int64 => {
            let v: Vec<i64> = src.iter().map(|&x| x as i64).collect();
            let _ = bs.compress(config, &ZfpField::new(&v, dims));
        }
        ZfpScalarType::Float => {
            let v: Vec<f32> = src.iter().map(|&x| x as f32).collect();
            let _ = bs.compress(config, &ZfpField::new(&v, dims));
        }
        ZfpScalarType::Double => {
            let _ = bs.compress(config, &ZfpField::new(src, dims));
        }
    }
    bs.flush();
    bs.as_bytes().to_vec()
}

fn write_seed(dir: &Path, name: &str, bytes: &[u8]) {
    fs::create_dir_all(dir).unwrap_or_else(|e| panic!("create {}: {e}", dir.display()));
    let path = dir.join(format!("{name}.bin"));
    fs::write(&path, bytes).unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
}
