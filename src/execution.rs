//! Parallel execution support (optional `rayon` feature).
//!
//! Provides [`ZfpExecution`], which selects between serial and Rayon-based
//! parallel block processing.

// The API and validation layer computes with caller-supplied sizes, so its
// arithmetic and indexing must be checked; see the crate's panic guarantee.
#![warn(clippy::indexing_slicing, clippy::arithmetic_side_effects)]

/// Execution policy for compression / decompression.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ZfpExecution {
    /// Serial (single-threaded) execution: the default.
    #[default]
    Serial,
    /// Parallel execution using Rayon.
    ///
    /// Without the `rayon` feature this runs serially, so code can select it
    /// unconditionally. The output is identical either way.
    ///
    /// Fixed-rate decompression assigns independent chunks to workers.
    /// Variable-rate decompression pipelines a serial plane reader with
    /// reconstruction workers, using a bounded queue of reusable buffers (about
    /// 1 MiB by default). The compressed format is unchanged and needs no block index.
    /// Pipeline speed depends on how much work remains after the plane walk;
    /// small blocks can be slower than serial decoding.
    ///
    /// Decompression runs serially for fields whose strides may alias (where
    /// two blocks could write the same element). The variable-rate pipeline
    /// also runs serially with fewer than two pool threads. Fixed-rate
    /// decompression falls back to serial if the pool or the chunks' buffers
    /// cannot be created, and the pipeline if the pool or its buffers cannot.
    ///
    /// With `threads: 0`, the current pool runs the chunks: Rayon's global
    /// pool, starting it if needed, outside any other. Rayon panics if it
    /// cannot start that pool's threads, and zfp-rs cannot detect this
    /// beforehand; pass a thread count to have a failure fall back to serial
    /// execution instead.
    ///
    /// A nonzero `threads` builds a new pool for every call and drops it
    /// afterwards. That costs tens of microseconds, which is lost in the time
    /// to code a field of 64³ values but can exceed the whole serial time for
    /// one of a few thousand. To code many small fields, build a pool once and
    /// make the calls inside its `install` with `threads: 0`, so that they
    /// share its threads:
    ///
    /// ```
    /// # #[cfg(feature = "rayon")]
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use zfp_rs::{ZfpBitStream, ZfpConfig, ZfpDimensionality, ZfpExecution, ZfpField};
    /// use zfp_rs::{ZfpScalarType, ZfpStreamAlignment};
    ///
    /// let config = ZfpConfig::fixed_rate(
    ///     8.0,
    ///     ZfpScalarType::F64,
    ///     ZfpDimensionality::D2,
    ///     ZfpStreamAlignment::Unaligned,
    /// )?;
    /// let data = vec![0.5f64; 16 * 16];
    /// let field = ZfpField::new(&data, [16usize, 16])?;
    /// let capacity = config.maximum_size(ZfpScalarType::F64, field.dims()).unwrap();
    /// let mut bs = ZfpBitStream::new(capacity)?;
    ///
    /// // One pool for every call, rather than one built for each.
    /// let pool = rayon::ThreadPoolBuilder::new().num_threads(4).build()?;
    /// let execution = ZfpExecution::Rayon {
    ///     threads: 0,
    ///     chunk_size: 0,
    /// };
    /// for _ in 0..100 {
    ///     bs.rewind();
    ///     pool.install(|| bs.compress_with_execution(&config, &field, execution))?;
    /// }
    /// # Ok(())
    /// # }
    /// # #[cfg(not(feature = "rayon"))]
    /// # fn main() {}
    /// ```
    Rayon {
        /// Number of threads; 0 uses the current pool, which is Rayon's global
        /// pool outside any other. A nonzero count builds a new pool for each
        /// call.
        threads: u32,
        /// Number of blocks per chunk. For compression and fixed-rate
        /// decompression, 0 means one chunk per thread. For variable-rate
        /// decompression this sets blocks per batch; 0 selects about
        /// 64 KiB of planes per batch. Up to 16 reusable buffers are allocated.
        chunk_size: u32,
    },
}

/// Run `f` on a new pool of `threads` threads, or with `threads: 0` on the
/// current pool, which is Rayon's global pool outside any other.
///
/// Returns `None` if the pool cannot be built, so the caller can run serially.
#[cfg(feature = "rayon")]
pub(crate) fn install<R: Send>(threads: u32, f: impl FnOnce() -> R + Send) -> Option<R> {
    if threads == 0 {
        return Some(f());
    }
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads as usize)
        .build()
        .ok()?;
    Some(pool.install(f))
}

/// Split `blocks` blocks into chunks, following C OMP semantics:
///
/// - `chunk_size = 0` means one chunk per thread.
/// - Otherwise `chunks = ceil(blocks / chunk_size)`, capped at `blocks`.
///
/// Chunk `i` covers blocks `[blocks * i / chunks, blocks * (i + 1) / chunks)`,
/// as in C. Returns `None` if the ranges cannot be allocated.
#[cfg(feature = "rayon")]
pub(crate) fn chunk_ranges(
    blocks: usize,
    threads: u32,
    chunk_size: u32,
) -> Option<Vec<std::ops::Range<usize>>> {
    let chunks = if chunk_size > 0 {
        blocks.div_ceil(chunk_size as usize).min(blocks)
    } else if threads > 0 {
        (threads as usize).min(blocks)
    } else {
        rayon::current_num_threads().min(blocks)
    }
    .max(1);
    // In `u128`, where `blocks * i` cannot overflow; the quotient is at most
    // `blocks`, so it fits back in `usize`.
    let start = |i: usize| {
        (blocks as u128)
            .checked_mul(i as u128)
            .and_then(|product| product.checked_div(chunks as u128))
            .and_then(|start| usize::try_from(start).ok())
    };
    let mut ranges = crate::bitstream::vec_with_capacity(chunks).ok()?;
    let mut begin = 0;
    for end in 1..=chunks {
        let end = start(end)?;
        ranges.push(begin..end);
        begin = end;
    }
    Some(ranges)
}

#[cfg(test)]
#[cfg(feature = "rayon")]
mod tests {
    use super::chunk_ranges;

    #[test]
    fn chunk_ranges_follow_c_and_cover_every_block() {
        assert_eq!(chunk_ranges(10, 3, 0), Some(vec![0..3, 3..6, 6..10]));
        assert_eq!(chunk_ranges(10, 0, 4), Some(vec![0..3, 3..6, 6..10]));
        assert_eq!(chunk_ranges(2, 8, 0), Some(vec![0..1, 1..2]));
        // An empty field still gets one, empty, chunk.
        let empty = chunk_ranges(0, 8, 0).unwrap();
        assert_eq!((empty.len(), empty[0].clone()), (1, 0..0));
    }

    /// `blocks * i` overflowed `usize` for huge block counts.
    #[test]
    fn chunk_ranges_do_not_overflow_for_huge_block_counts() {
        let blocks = usize::MAX / 3;
        let ranges = chunk_ranges(blocks, 8192, 0).unwrap();
        assert_eq!(ranges.len(), 8192);
        assert_eq!(ranges.last().unwrap().end, blocks);
        assert!(ranges.windows(2).all(|pair| pair[0].end == pair[1].start));
        // One chunk per block: more ranges than can be allocated.
        assert_eq!(chunk_ranges(blocks, 0, 1), None);
    }
}
