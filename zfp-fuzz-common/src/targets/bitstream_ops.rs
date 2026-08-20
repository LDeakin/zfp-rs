//! Fuzz target: deep random sequences of bitstream operations.
//!
//! `tests/proptest/bitstream_compat.rs` compares fixed scenarios against the C
//! implementation. What it does not do is search *sequences* — and the
//! bitstream's cursor state (`word_pos`, `bits`, the read/write buffers) is a
//! small state machine where the interesting bugs live in orderings, not in
//! single calls.
//!
//! Every operand is clamped into a valid range, so an index panic is excluded
//! by construction and anything that does panic is a genuine defect. In
//! particular `read_pos` is `word_pos * 64 - bits`, which underflows whenever
//! the read cursor is in the first word with buffered bits — the fuzz profile
//! enables `overflow-checks` specifically so that surfaces as a crash rather
//! than wrapping to `u64::MAX`.

use zfp_rs::ZfpBitStream;

/// Stream capacity in bytes. Small enough to keep each input fast, large
/// enough for multi-word cursor movement.
const CAPACITY: usize = 256;

/// Cap on operations per input, so a long corpus entry cannot dominate a run.
const MAX_OPS: usize = 512;

/// Bits per bitstream word.
const WORD_BITS: u32 = 64;

/// Entry point shared by the libFuzzer harness and the stable regression test.
pub fn run(data: &[u8]) {
    let mut bs = ZfpBitStream::new(CAPACITY);
    let capacity_bits = (CAPACITY * 8) as u64;

    for chunk in data.chunks(3).take(MAX_OPS) {
        let op = chunk[0];
        let a = chunk.get(1).copied().unwrap_or(0);
        let b = chunk.get(2).copied().unwrap_or(0);

        // Bit counts are 0..=64; offsets are clamped inside the stream.
        let n = u32::from(a) % 65;
        let value = u64::from(a) | (u64::from(b) << 8);
        let offset = (u64::from(a) | (u64::from(b) << 8)) % capacity_bits;

        // Writing past the end of the stream is a caller error, not a defect:
        // C's `stream_write_word` has no bounds check and simply overflows the
        // buffer, so there is no behaviour here worth asserting. Rewind before
        // the cursor reaches capacity so the sequence keeps exercising the
        // cursor logic instead of stopping on a full stream.
        // 8 words of headroom: a single `pad` can advance the cursor by four
        // words, and `write_bits` by two, so a tighter margin still overruns.
        if bs.word_pos() + 8 >= CAPACITY / 8 {
            bs.rewind();
        }

        match op % 12 {
            0 => {
                bs.write_bits(value, n);
            }
            1 => {
                bs.write_bit(u32::from(b & 1));
            }
            2 => {
                bs.write_word(value);
            }
            3 => {
                bs.read_bits(n);
            }
            4 => {
                bs.read_bit();
            }
            5 => {
                bs.read_word();
            }
            6 => bs.seek_write(offset),
            7 => bs.seek_read(offset),
            8 => bs.skip(usize::from(a)),
            9 => {
                // Bounded so one `pad` cannot outrun the headroom check above.
                bs.pad(usize::from(a) % 65);
            }
            10 => {
                bs.flush();
            }
            _ => bs.rewind(),
        }

        // The cursor queries are deliberately *not* bounded here.
        //
        // `bits` is shared between the read and write buffers, so querying the
        // read position on a write-only stream wraps, matching C's
        // `stream_rtell` — which `tests/proptest/bitstream_compat.rs` asserts
        // against. Seeking to a wrapped position then leaves `word_pos` and
        // `write_pos` equally nonsensical. All of that is C-compatible by
        // design; the property under test is that none of it panics or reads
        // out of bounds, which the calls below and ASan cover between them.
        let _ = bs.read_pos();
        let _ = bs.write_pos();
        let _ = bs.word_pos();
        let _ = bs.bits_written();
        let _ = bs.size();
        let _ = bs.as_bytes();

        // Round-trip check, run immediately after a write so no other
        // operation can clobber the bits in between. A deferred shadow model
        // would be unsound here: any later write, seek or rewind may legally
        // overwrite the same offsets.
        if op % 12 == 0 && (1..64).contains(&n) {
            let end = bs.write_pos();
            // Leave a word of headroom: the `flush` below commits the partial
            // buffer and must not run the cursor off the end.
            if end >= u64::from(n) && end + u64::from(WORD_BITS) <= capacity_bits {
                let pos = end - u64::from(n);
                bs.flush();
                bs.seek_read(pos);
                let got = bs.read_bits(n);
                assert_eq!(
                    got,
                    value & mask(n),
                    "wrote {:#x} ({n} bits) at bit {pos} but read back {got:#x}",
                    value & mask(n)
                );
                bs.seek_write(end);
            }
        }
    }
}

fn mask(n: u32) -> u64 {
    if n >= 64 { u64::MAX } else { (1u64 << n) - 1 }
}
