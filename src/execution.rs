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
    /// `threads` = number of threads (0 means use Rayon default).
    /// `chunk_size` = number of blocks per chunk (0 means one chunk per thread).
    Rayon { threads: u32, chunk_size: u32 },
}

/// Resolve the effective number of threads for a Rayon execution.
#[cfg(feature = "rayon")]
#[allow(dead_code)]
#[inline]
pub(crate) fn rayon_thread_count(threads: u32) -> usize {
    if threads > 0 {
        threads as usize
    } else {
        rayon::current_num_threads()
    }
}

/// Resolve the number of chunks for splitting blocks.
///
/// Mirrors C OMP chunk semantics:
/// - `chunk_size = 0` means one chunk per thread.
/// - Otherwise `chunks = ceil(blocks / chunk_size)`, capped at `blocks`.
#[cfg(feature = "rayon")]
#[allow(dead_code)]
#[inline]
pub(crate) fn compute_chunk_count(blocks: usize, chunk_size: u32, threads: usize) -> usize {
    if blocks == 0 {
        return 0;
    }
    if chunk_size == 0 {
        // One chunk per thread
        threads.min(blocks)
    } else {
        // ceil(blocks / chunk_size), capped at blocks
        blocks.div_ceil(chunk_size as usize).min(blocks)
    }
}

/// Compute the block range for a given chunk index.
///
/// Chunk `i` covers blocks `[start, end)` where
/// `start = (blocks * i) / chunks` and `end = (blocks * (i + 1)) / chunks`.
/// This mirrors the C OMP chunking strategy.
#[cfg(feature = "rayon")]
#[allow(dead_code)]
#[inline]
pub(crate) fn chunk_range(
    blocks: usize,
    chunks: usize,
    chunk_idx: usize,
) -> std::ops::Range<usize> {
    let start = (blocks * chunk_idx) / chunks;
    let end = (blocks * (chunk_idx + 1)) / chunks;
    start..end
}
