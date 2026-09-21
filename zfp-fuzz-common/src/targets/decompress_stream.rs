//! Fuzz target: decompress **untrusted** bytes into a well-formed field.
//!
//! This is the gap the existing test suite cannot reach. Every test in
//! `tests/c/` and `tests/proptest/` decompresses a stream the encoder — Rust or
//! C — just produced. Nothing feeds the decoder bytes it did not write, which
//! is precisely the input a compression library sees in the wild.
//!
//! Input framing is hand-rolled rather than `#[derive(Arbitrary)]` so the byte
//! layout is stable across `arbitrary` releases and `gen_seeds` can emit
//! `HEADER ++ <real compressed stream>` files:
//!
//! ```text
//! [0]       scalar kind          (value % 4)
//! [1]       rank                 (1 + value % 4)
//! [2..6]    per-axis side bytes
//! [6]       mode family          (value % 5)
//! [7..10]   mode parameter       (24-bit LE)
//! [10]      execution policy
//! [11]      flags (stream alignment)
//! [12..]    compressed payload
//! ```
//!
//! The stream is zero-padded out to the decoder's true worst-case budget
//! (`num_blocks * max_bits`, not `maximum_size` — see the comment in `typed`).
//! That deliberately excludes the truncated-stream index panic
//! (`read_word_raw` is `stream.words()[pos]`), which would otherwise be the
//! only thing this target ever reported. **Any panic reaching this target is
//! therefore a real bug.**

use zfp_rs::{ZfpBitStream, ZfpField, ZfpFieldMut};

use crate::input::{ExecSpec, ModeSpec, ScalarKind, Shape};
use crate::limits::MAX_STREAM_BYTES;
use crate::scalar::FuzzScalar;

/// Width of the fixed framing prefix.
pub const HEADER_LEN: usize = 12;

/// Entry point shared by the libFuzzer harness and the stable regression test.
pub fn run(data: &[u8]) {
    if data.len() < HEADER_LEN {
        return;
    }
    let (hdr, payload) = data.split_at(HEADER_LEN);

    let kind = ScalarKind::from_byte(hdr[0]);
    let shape = Shape::from_bytes(hdr[1], [hdr[2], hdr[3], hdr[4], hdr[5]]);
    let mode = ModeSpec::from_bytes(hdr[6], [hdr[7], hdr[8], hdr[9]], hdr[11]);
    let exec = ExecSpec::from_byte(hdr[10]);

    match kind {
        ScalarKind::I32 => typed::<i32>(shape, mode, exec, payload),
        ScalarKind::I64 => typed::<i64>(shape, mode, exec, payload),
        ScalarKind::F32 => typed::<f32>(shape, mode, exec, payload),
        ScalarKind::F64 => typed::<f64>(shape, mode, exec, payload),
    }
}

/// Number of 4^d blocks covering the given dimensions.
fn num_blocks(dims: &[usize]) -> usize {
    dims.iter().map(|&n| n.div_ceil(4)).product()
}

fn typed<T: FuzzScalar>(shape: Shape, mode: ModeSpec, exec: ExecSpec, payload: &[u8]) {
    let dims = shape.dims();
    let rank = shape.rank();
    let n = shape.elements();
    let ty = T::scalar_type();

    let Some(config) = mode.to_config(ty, shape.dimensionality()) else {
        return;
    };
    let cap = config.maximum_size(ty, &dims[..rank]);
    if cap == 0 || cap > MAX_STREAM_BYTES {
        return;
    }

    // `maximum_size` bounds a *well-formed* stream: it clamps the per-block
    // budget to what the scalar type and block shape can actually produce. A
    // malformed stream is not so constrained — the decoder's real per-block
    // budget is `config.max_bits()`, which for reversible and expert modes is
    // `ZFP_MAX_BITS` (16658), far above the well-formed bound. Sizing the
    // buffer to `cap` therefore makes the decoder run off the end of its own
    // words slice on adversarial input, which is a finding in its own right
    // (asserted below) but would otherwise stop this target exploring anything
    // else.
    let blocks = num_blocks(&dims[..rank]);
    let Some(worst_case) = blocks
        .checked_mul(config.max_bits() as usize)
        .map(|bits| bits.div_ceil(8) + 8)
    else {
        return;
    };
    let buffer_bytes = cap.max(worst_case);
    if buffer_bytes > MAX_STREAM_BYTES {
        return;
    }

    // Zero-pad the attacker-controlled payload out to the full buffer. Words
    // are native-endian, matching the crate's documented little-endian
    // bitstream contract — which also means corpora are not portable to a
    // big-endian host.
    let mut words = vec![0u64; buffer_bytes.div_ceil(8)];
    for (word, chunk) in words.iter_mut().zip(payload.chunks(8)) {
        let mut bytes = [0u8; 8];
        bytes[..chunk.len()].copy_from_slice(chunk);
        *word = u64::from_ne_bytes(bytes);
    }
    let mut bs = ZfpBitStream::from_buffer(words);

    // Exactly `n` elements, so ASan redzones abut the last element and any
    // overrun in the unsafe scatter path is caught.
    let mut dst = vec![T::default(); n];
    let exec = exec.to_execution();
    let consumed = {
        let mut out = ZfpFieldMut::new(&mut dst, dims);
        match bs.decompress_with_execution(&config, &mut out, exec) {
            Ok(consumed) => consumed,
            Err(e) => panic!("decompress failed on an exactly-sized field: {e}"),
        }
    };
    assert!(
        consumed <= buffer_bytes,
        "decompress consumed {consumed} B from a {buffer_bytes} B stream"
    );

    // The same bytes must always decode to the same values.
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
            "decompression of untrusted bytes is not deterministic at index {i}"
        );
    }

    // Feed the adversarially-decoded values straight back into the encoder.
    // They may contain NaN, infinities and subnormals in combinations no
    // generator would produce, so this is free coverage of the encode path.
    let field = ZfpField::new(&dst, dims);
    let mut re = ZfpBitStream::new(cap);
    let written = re
        .compress(&config, &field)
        .expect("re-compress of decoded values must succeed");
    assert!(written <= cap, "re-compress wrote {written} B into {cap} B");
}
