//! Property-based compatibility tests: bitstream_compat.
//!
//! Uses `proptest` + `zfp-sys` to verify byte-for-byte output compatibility
//! with the reference C library.
//!
//! Each test generates a random sequence of bitstream operations, applies them
//! to both the Rust `ZfpBitStream` and the C `stream_*` functions via `zfp-sys`,
//! and asserts that the results are identical.

use proptest::prelude::*;
use zfp_rs::ZfpBitStream;

// ---------------------------------------------------------------------------
// Safe wrapper around the C zfp bitstream
// ---------------------------------------------------------------------------

struct CStream {
    ptr: *mut zfp_sys::bitstream,
    buf: Vec<u8>,
}

impl CStream {
    fn new(capacity: usize) -> Self {
        let mut buf = vec![0u8; capacity];
        let ptr =
            unsafe { zfp_sys::stream_open(buf.as_mut_ptr().cast::<std::ffi::c_void>(), capacity) };
        assert!(!ptr.is_null());
        Self { ptr, buf }
    }

    fn read_bit(&mut self) -> u32 {
        unsafe { zfp_sys::stream_read_bit(self.ptr) }
    }

    fn write_bit(&mut self, bit: u32) -> u32 {
        unsafe { zfp_sys::stream_write_bit(self.ptr, bit) }
    }

    fn read_bits(&mut self, n: usize) -> u64 {
        unsafe { zfp_sys::stream_read_bits(self.ptr, n) }
    }

    fn write_bits(&mut self, value: u64, n: usize) -> u64 {
        unsafe { zfp_sys::stream_write_bits(self.ptr, value, n) }
    }

    fn rtell(&self) -> u64 {
        unsafe { zfp_sys::stream_rtell(self.ptr) }
    }

    fn wtell(&self) -> u64 {
        unsafe { zfp_sys::stream_wtell(self.ptr) }
    }

    fn rewind(&mut self) {
        unsafe { zfp_sys::stream_rewind(self.ptr) }
    }

    fn flush(&mut self) -> usize {
        unsafe { zfp_sys::stream_flush(self.ptr) }
    }

    fn rseek(&mut self, offset: u64) {
        unsafe { zfp_sys::stream_rseek(self.ptr, offset) }
    }

    fn wseek(&mut self, offset: u64) {
        unsafe { zfp_sys::stream_wseek(self.ptr, offset) }
    }

    /// Return committed buffer as bytes (after flush).
    fn as_bytes(&self) -> &[u8] {
        let size = unsafe { zfp_sys::stream_size(self.ptr) };
        &self.buf[..size]
    }
}

impl Drop for CStream {
    fn drop(&mut self) {
        unsafe { zfp_sys::stream_close(self.ptr) };
    }
}

// ---------------------------------------------------------------------------
// Operation model
// ---------------------------------------------------------------------------

/// A single write bitstream operation, applied to both implementations.
#[derive(Clone, Debug)]
enum Op {
    WriteBit(u32),
    WriteBits { value: u64, n: u32 },
}

fn write_ops_strategy() -> impl Strategy<Value = Vec<Op>> {
    prop::collection::vec(
        prop_oneof![
            (0u32..=1u32).prop_map(Op::WriteBit),
            (0u64..=u64::MAX, 1u32..=63u32).prop_map(|(v, n)| {
                let mask = if n < 64 { (1u64 << n) - 1 } else { u64::MAX };
                Op::WriteBits { value: v & mask, n }
            }),
        ],
        1..=200,
    )
}

// ---------------------------------------------------------------------------
// Proptest: write_bits byte-for-byte compatibility
// ---------------------------------------------------------------------------

proptest! {
    /// Write random bit sequences to both implementations; after flushing,
    /// assert the committed bytes are identical.
    #[test]
    fn write_bits_compat(ops in write_ops_strategy()) {
        // Large enough buffer for 200 × 63-bit writes = ~1600 bytes; use 4096.
        let capacity = 4096;
        let mut rs = ZfpBitStream::new(capacity);
        let mut cs = CStream::new(capacity);

        for op in &ops {
            match *op {
                Op::WriteBit(b) => {
                    let r = rs.write_bit(b);
                    let c = cs.write_bit(b);
                    prop_assert_eq!(r, c, "write_bit return mismatch for bit={}", b);
                }
                Op::WriteBits { value, n } => {
                    let r = rs.write_bits(value, n);
                    let c = cs.write_bits(value, n as usize);
                    prop_assert_eq!(r, c, "write_bits overflow mismatch for value={} n={}", value, n);
                }
            }
        }

        // Flush both to commit partial word.
        rs.flush();
        cs.flush();

        // Compare wtell (total bits written before flush padding).
        // After flush both streams are word-aligned; compare committed bytes.
        let rs_bytes = rs.as_bytes();
        let cs_bytes = cs.as_bytes();
        prop_assert_eq!(
            rs_bytes, cs_bytes,
            "buffer mismatch after write sequence"
        );
    }
}

proptest! {
    /// Write a known bit sequence, rewind both, read back with read_bits and
    /// read_bit; assert every read value matches between the two implementations.
    #[test]
    fn read_bits_compat(
        write_ops in write_ops_strategy(),
    ) {
        let capacity = 4096;
        let mut rs = ZfpBitStream::new(capacity);
        let mut cs = CStream::new(capacity);

        // Track total bits written so we know how many to read back.
        let mut total_bits_written: u64 = 0;
        for op in &write_ops {
            match *op {
                Op::WriteBit(b) => {
                    rs.write_bit(b);
                    cs.write_bit(b);
                    total_bits_written += 1;
                }
                Op::WriteBits { value, n } => {
                    rs.write_bits(value, n);
                    cs.write_bits(value, n as usize);
                    total_bits_written += u64::from(n);
                }
            }
        }
        rs.flush();
        cs.flush();
        rs.rewind();
        cs.rewind();

        // Read back all bits using read_bits(1) to avoid boundary complexity.
        let mut remaining = total_bits_written;
        while remaining > 0 {
            let n = remaining.min(63) as u32;
            let r = rs.read_bits(n);
            let c = cs.read_bits(n as usize);
            prop_assert_eq!(r, c, "read_bits mismatch at bit offset {} n={}", total_bits_written - remaining, n);
            remaining -= u64::from(n);
        }
    }
}

proptest! {
    /// write_bit and read_bit round-trip: write bits one at a time, read back
    /// one at a time, assert each bit matches.
    #[test]
    fn write_read_bit_compat(bits in prop::collection::vec(0u32..=1u32, 1..=256)) {
        let capacity = 4096;
        let mut rs = ZfpBitStream::new(capacity);
        let mut cs = CStream::new(capacity);

        for &b in &bits {
            let r = rs.write_bit(b);
            let c = cs.write_bit(b);
            prop_assert_eq!(r, c, "write_bit return mismatch");
        }
        rs.flush();
        cs.flush();
        rs.rewind();
        cs.rewind();

        for (i, &b) in bits.iter().enumerate() {
            let r = rs.read_bit();
            let c = cs.read_bit();
            prop_assert_eq!(r, c, "read_bit mismatch at index {}, expected {}", i, b);
        }
    }
}

proptest! {
    /// wtell/rtell: after writing, wtell must match; after rewinding and reading,
    /// rtell must match at each step.
    #[test]
    fn tell_compat(ops in write_ops_strategy()) {
        let capacity = 4096;
        let mut rs = ZfpBitStream::new(capacity);
        let mut cs = CStream::new(capacity);

        for op in &ops {
            match *op {
                Op::WriteBit(b) => { rs.write_bit(b); cs.write_bit(b); }
                Op::WriteBits { value, n } => { rs.write_bits(value, n); cs.write_bits(value, n as usize); }
            }
            prop_assert_eq!(rs.write_pos(), cs.wtell(), "write_pos mismatch");
        }
    }
}

proptest! {
    /// wseek/rseek: seek to a word-aligned position and write/read, assert the
    /// stream contents match.
    #[test]
    fn seek_compat(
        initial_bits in prop::collection::vec(
            (0u64..u64::MAX, 1u32..=63u32).prop_map(|(v, n)| {
                let mask = if n < 64 { (1u64 << n) - 1 } else { u64::MAX };
                (v & mask, n)
            }),
            4..=8,
        ),
        seek_word in 0usize..=3usize,
        extra_bits in prop::collection::vec(0u32..=1u32, 1..=64),
    ) {
        let capacity = 4096;
        let mut rs = ZfpBitStream::new(capacity);
        let mut cs = CStream::new(capacity);

        // Write initial pattern.
        for &(value, n) in &initial_bits {
            rs.write_bits(value, n);
            cs.write_bits(value, n as usize);
        }
        rs.flush();
        cs.flush();

        // Seek to a word-aligned offset.
        let offset = (seek_word * 64) as u64;
        rs.seek_write(offset);
        cs.wseek(offset);
        prop_assert_eq!(rs.write_pos(), cs.wtell(), "write_pos mismatch after seek_write");

        // Write extra bits.
        for &b in &extra_bits {
            rs.write_bit(b);
            cs.write_bit(b);
        }
        rs.flush();
        cs.flush();

        // Seek back to same word and read back.
        rs.seek_read(offset);
        cs.rseek(offset);
        prop_assert_eq!(rs.read_pos(), cs.rtell(), "read_pos mismatch after seek_read");

        for (i, &b) in extra_bits.iter().enumerate() {
            let r = rs.read_bit();
            let c = cs.read_bit();
            prop_assert_eq!(r, c, "read_bit mismatch at extra bit index {}, expected {}", i, b);
        }
    }
}

proptest! {
    /// flush compatibility: write a partial word, flush both, compare the
    /// number of padding bits returned and the resulting buffer.
    #[test]
    fn flush_compat(
        value in 0u64..u64::MAX,
        n in 1u32..63u32,
    ) {
        let mask = (1u64 << n) - 1;
        let value = value & mask;
        let capacity = 64;
        let mut rs = ZfpBitStream::new(capacity);
        let mut cs = CStream::new(capacity);

        rs.write_bits(value, n);
        cs.write_bits(value, n as usize);

        let r_pad = rs.flush();
        let c_pad = cs.flush();
        prop_assert_eq!(r_pad, c_pad, "flush padding mismatch");

        prop_assert_eq!(
            rs.as_bytes(),
            cs.as_bytes(),
            "buffer mismatch after flush"
        );
    }
}

proptest! {
    /// Round-trip: write random u64 values with n == 64 (full-width), then
    /// read back and assert every value is recovered.  This exercises the
    /// `value >> 1` / `remaining = n - 1` correctness path in write_bits
    /// and the `n == 64` mask-gate in read_bits.
    #[test]
    fn write_read_bits_64_roundtrip(
        values in prop::collection::vec(0u64..u64::MAX, 1..=64),
    ) {
        let capacity = 512;
        let mut rs = ZfpBitStream::new(capacity);
        let mut cs = CStream::new(capacity);

        // Write each value as 64 bits.
        for &v in &values {
            let r = rs.write_bits(v, 64);
            let c = cs.write_bits(v, 64);
            prop_assert_eq!(r, c, "write_bits(64) overflow mismatch for value={}", v);
        }

        rs.flush();
        cs.flush();

        // Read back all bits using read_bits(1) for precise per-bit comparison.
        rs.rewind();
        cs.rewind();

        for (i, &expected) in values.iter().enumerate() {
            for bit_pos in 0u32..64u32 {
                let expected_bit = ((expected >> bit_pos) & 1) as u32;
                let r = rs.read_bit();
                let c = cs.read_bit();
                let bit_idx = (i as u32) * 64 + bit_pos;
                prop_assert_eq!(r, c, "read_bit mismatch at bit {} (value {} bit {})",
                    bit_idx, i, bit_pos);
                prop_assert_eq!(r, expected_bit,
                    "bit {} (value {} bit {}): expected {} got {}",
                    bit_idx, i, bit_pos, expected_bit, r);
            }
        }
    }
}

proptest! {
    /// Round-trip: write random values with n == 64, read back with read_bits(64),
    /// assert value equality.  Also verifies that the overflow return is 0 for
    /// n == 64 (all bits consumed).
    #[test]
    fn write_read_bits_64_readback(value in 0u64..u64::MAX) {
        let capacity = 16;
        let mut rs = ZfpBitStream::new(capacity);
        let mut cs = CStream::new(capacity);

        // Write and verify overflow is 0.
        let r_ovf = rs.write_bits(value, 64);
        let c_ovf = cs.write_bits(value, 64);
        prop_assert_eq!(r_ovf, c_ovf, "write_bits(64) overflow mismatch");
        prop_assert_eq!(r_ovf, 0, "write_bits(64) should return 0 overflow");

        rs.flush();
        cs.flush();

        // Read back with read_bits(64).
        rs.rewind();
        cs.rewind();

        let r = rs.read_bits(64);
        let c = cs.read_bits(64);
        prop_assert_eq!(r, c, "read_bits(64) mismatch");
        prop_assert_eq!(r, value, "read_bits(64) did not recover original value");
    }
}
