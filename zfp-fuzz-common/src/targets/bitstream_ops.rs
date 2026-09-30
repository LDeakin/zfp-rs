//! Fuzz target: deep random sequences of bitstream operations.
//!
//! `tests/proptest/bitstream_compat.rs` compares fixed scenarios against the C
//! implementation. What it does not do is search *sequences* — and the
//! bitstream's cursor state (`word_pos`, `bits`, the read/write buffers) is a
//! small state machine where the interesting bugs live in orderings, not in
//! single calls.
//!
//! The mixed-operation pass searches for panics and out-of-bounds access with
//! unclamped operands: bit counts above 64, offsets anywhere in `u64`, and
//! pads and copies of up to `u64::MAX` bits. A separate write-only pass, with
//! operands clamped into range, checks that committed writes read back
//! correctly.

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
    run_mixed_ops(data);
    run_write_roundtrips(data);
}

// Exercise arbitrary cursor transitions for panics and out-of-bounds access.
// Read and write positions can wrap after a mixed transition, so they are not
// an oracle for whether a write reached the backing buffer.
fn run_mixed_ops(data: &[u8]) {
    let mut bs = ZfpBitStream::new(CAPACITY).expect("the stream allocates");
    let capacity_bits = (CAPACITY * 8) as u64;

    for chunk in data.chunks(3).take(MAX_OPS) {
        let op = chunk[0];
        let a = chunk.get(1).copied().unwrap_or(0);
        let b = chunk.get(2).copied().unwrap_or(0);

        // Bit counts up to 255, past C's limit of 64. Offsets land inside the
        // stream, or, with the top bit of `b` set, within 2^15 bits of
        // `u64::MAX`; lengths reach `u64::MAX` the same way.
        let n = u32::from(a);
        let value = u64::from(a) | (u64::from(b) << 8);
        let wide = |bits: u64| {
            if b & 0x80 == 0 {
                bits % capacity_bits
            } else {
                u64::MAX - (bits & 0x7fff)
            }
        };
        let offset = wide(value);

        // Rewind once the cursor nears or passes capacity, so the sequence
        // keeps exercising the cursor logic instead of stopping on a full
        // stream. Wide seeks and pads still take it far past the end first.
        if bs.write_pos() / 64 + 8 >= (CAPACITY / 8) as u64 {
            bs.rewind();
        }

        match op % 13 {
            0 => {
                bs.write_bits(value, n);
            }
            1 => {
                bs.write_bit(b & 1 != 0);
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
            8 => bs.skip(wide(u64::from(a))),
            9 => {
                // A huge pad drops every word past the end at once.
                bs.pad(wide(u64::from(a)));
            }
            10 => {
                bs.flush();
            }
            11 => {
                let mut source = ZfpBitStream::from_words(vec![value; 4]);
                source.seek_read(offset);
                bs.copy_from(&mut source, wide(u64::from(a)));
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
        let _ = bs.overflowed();
        let _ = bs.as_bytes();
        let _ = bs.as_words();
    }
}

// Only write-mode transitions participate in the round-trip oracle. The
// mixed-state pass above still exercises every operation in the input.
fn run_write_roundtrips(data: &[u8]) {
    let mut bs = ZfpBitStream::new(CAPACITY).expect("the stream allocates");
    let capacity_bits = (CAPACITY * 8) as u64;

    for chunk in data.chunks(3).take(MAX_OPS) {
        let op = chunk[0] % 12;
        let a = chunk.get(1).copied().unwrap_or(0);
        let b = chunk.get(2).copied().unwrap_or(0);
        let n = u32::from(a) % 65;
        let value = u64::from(a) | (u64::from(b) << 8);
        let offset = value % capacity_bits;

        // Keep enough room for the largest write or pad and a later flush.
        if bs.write_pos() / u64::from(WORD_BITS) + 8 >= (CAPACITY / 8) as u64 {
            bs.rewind();
        }

        match op {
            0 => {
                bs.write_bits(value, n);
            }
            1 => bs.write_bit(b & 1 != 0),
            2 => bs.write_word(value),
            6 => bs.seek_write(offset),
            9 => bs.pad(u64::from(a) % 65),
            10 => {
                bs.flush();
            }
            11 => bs.rewind(),
            _ => continue,
        }

        // Round-trip check, run immediately after a write so no other
        // operation can clobber the bits in between. A deferred shadow model
        // would be unsound here: any later write, seek or rewind may legally
        // overwrite the same offsets.
        if op == 0 && (1..64).contains(&n) && !bs.overflowed() {
            let end = bs.write_pos();
            // Leave a word of headroom: the `flush` below commits the partial
            // buffer and must not run the cursor off the end.
            if end >= u64::from(n) && end <= capacity_bits - u64::from(WORD_BITS) {
                let pos = end - u64::from(n);
                bs.flush();
                // A flush can drop a partial word even if write_bits did not
                // overflow. Only committed bits have a readback guarantee.
                if !bs.overflowed() {
                    bs.seek_read(pos);
                    let got = bs.read_bits(n);
                    assert_eq!(
                        got,
                        value & mask(n),
                        "wrote {:#x} ({n} bits) at bit {pos} but read back {got:#x}",
                        value & mask(n)
                    );
                }
                bs.seek_write(end);
            }
        }
    }
}

fn mask(n: u32) -> u64 {
    if n >= 64 { u64::MAX } else { (1u64 << n) - 1 }
}
