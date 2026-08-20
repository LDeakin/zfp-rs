//! Fuzz target: compress → decompress round-trip across the full
//! (type × rank × mode × execution) matrix.
//!
//! This target has **no C oracle**, which is the point: it is free to feed
//! subnormals, NaN and infinities, the exact values the differential proptests
//! in `tests/proptest/` must filter out. Its strength comes from reversible
//! mode being genuinely lossless for every bit pattern — when the block
//! floating-point path fails its reversibility check the encoder falls back to
//! a raw two's-complement path (`src/codec/encode/reversible.rs`).

use arbitrary::{Arbitrary, Unstructured};
use zfp_rs::{ZfpBitStream, ZfpField, ZfpFieldMut, types::ZfpMode};

use crate::input::{ExecSpec, ModeSpec, ScalarKind, Shape};
use crate::limits::MAX_STREAM_BYTES;
use crate::scalar::{FuzzScalar, decode_scalars};

#[derive(Debug, Arbitrary)]
struct RoundtripInput<'a> {
    kind: ScalarKind,
    shape: Shape,
    mode: ModeSpec,
    exec: ExecSpec,
    /// Remaining bytes, cycled to fill the field.
    payload: &'a [u8],
}

/// Entry point shared by the libFuzzer harness and the stable regression test.
pub fn run(data: &[u8]) {
    let u = Unstructured::new(data);
    let Ok(input) = RoundtripInput::arbitrary_take_rest(u) else {
        return;
    };
    match input.kind {
        ScalarKind::I32 => typed::<i32>(&input),
        ScalarKind::I64 => typed::<i64>(&input),
        ScalarKind::F32 => typed::<f32>(&input),
        ScalarKind::F64 => typed::<f64>(&input),
    }
}

fn typed<T: FuzzScalar>(input: &RoundtripInput<'_>) {
    let shape = input.shape;
    let dims = shape.dims();
    let rank = shape.rank();
    let n = shape.elements();
    let ty = T::scalar_type();

    let Some(config) = input.mode.to_config(ty, shape.dimensionality()) else {
        return;
    };

    // `maximum_size` is num_blocks * max_bits, so this one check bounds both
    // the allocation and the runtime. Sizing the stream to exactly this value
    // also turns any violation of the documented upper bound into an immediate
    // crash rather than a silent over-allocation.
    let cap = config.maximum_size(ty, &dims[..rank]);
    if cap == 0 || cap > MAX_STREAM_BYTES {
        return;
    }

    let src: Vec<T> = decode_scalars::<T>(input.payload, n);
    let exec = input.exec.to_execution();

    let mut bs = ZfpBitStream::new(cap);
    let written = {
        let field = ZfpField::new(&src, dims);
        match bs.compress_with_execution(&config, &field, exec) {
            Ok(written) => written,
            // The field is exactly sized and non-empty, so neither `NoData`
            // nor `InvalidField` is reachable here.
            Err(e) => panic!("compress failed for an exactly-sized field: {e} (dims={dims:?})"),
        }
    };
    assert!(
        written <= cap,
        "compress wrote {written} B into a {cap} B stream sized by maximum_size"
    );

    // Exactly-sized output buffer, so ASan redzones sit immediately after the
    // last element and any overrun in the unsafe scatter path is caught.
    let mut dst = vec![T::default(); n];
    {
        let mut out = ZfpFieldMut::new(&mut dst, dims);
        bs.rewind();
        let read = bs
            .decompress_with_execution(&config, &mut out, exec)
            .expect("decompress of a stream we just produced must succeed");
        assert_eq!(
            read, written,
            "decompress consumed {read} B but compress wrote {written} B"
        );
    }

    // Decoding must be deterministic and leave no residual stream state.
    let mut again = vec![T::default(); n];
    {
        let mut out = ZfpFieldMut::new(&mut again, dims);
        bs.rewind();
        let _ = bs.decompress_with_execution(&config, &mut out, exec);
    }
    for i in 0..n {
        assert_eq!(
            dst[i].to_bits_u64(),
            again[i].to_bits_u64(),
            "decompression is not deterministic at index {i}"
        );
    }

    // No mode-specific assertion for the lossy modes.
    //
    // Fixed-rate, fixed-precision and expert have no closed-form error bound
    // that survives adversarial input. Fixed-accuracy nominally does, but it
    // does not hold at the extremes this target reaches — a block spanning a
    // wide dynamic range reconstructs its small elements far outside the
    // tolerance, and large magnitudes can reconstruct to infinity from finite
    // input. Whether either is a `zfp-rs` divergence or inherent to zfp needs
    // the C reference to settle, and the C oracle is deliberately kept out of
    // the fuzz targets; see the known-open findings in `fuzz/README.md`.
    //
    // Those modes are still fully fuzzed — the structural invariants,
    // determinism check and ASan coverage above all apply. Reversible mode is
    // the exact correctness oracle, and it is unaffected: it is lossless for
    // every bit pattern, including NaN, infinities and subnormals.
    if config.compression_mode() == ZfpMode::Reversible {
        for i in 0..n {
            assert_eq!(
                src[i].to_bits_u64(),
                dst[i].to_bits_u64(),
                "reversible mode is lossy at index {i} (dims={dims:?}, type={ty})"
            );
        }
    }
}
