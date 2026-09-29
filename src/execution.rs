//! Parallel execution support (optional `rayon` feature).
//!
//! Provides [`ZfpExecution`], which selects between serial and Rayon-based
//! parallel block processing.

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
    /// Decompression runs serially instead for streams that are not fixed-rate,
    /// and for fields whose strides may alias (where two blocks could write the
    /// same element). Both run serially if the thread pool or the chunks'
    /// buffers cannot be created.
    ///
    /// With `threads: 0`, Rayon's global pool runs the chunks, starting it if
    /// needed. Rayon panics if it cannot start that pool's threads, and zfp-rs
    /// cannot detect this beforehand; pass a thread count to have a failure
    /// fall back to serial execution instead.
    Rayon {
        /// Number of threads; 0 uses Rayon's global pool.
        threads: u32,
        /// Number of blocks per chunk; 0 means one chunk per thread.
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

#[cfg(all(test, feature = "rayon"))]
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
