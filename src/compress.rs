//! Core compression logic.
//!
//! Provides the borrow-based implementation used by
//! [`ZfpBitStream::compress`][crate::ZfpBitStream::compress].

// The API and validation layer computes with caller-supplied sizes, so its
// arithmetic and indexing must be checked; see the crate's panic guarantee.
#![warn(clippy::indexing_slicing, clippy::arithmetic_side_effects)]

use crate::bitstream::ZfpBitStreamMutOps;
use crate::config::ZfpConfig;
use crate::field::ZfpField;
use crate::field_plan::FieldPlan;
use crate::types::{ZfpCompressionError, ZfpScalar, ZfpScalarType};
use std::ops::Range;

// ---------------------------------------------------------------------------
// Serial compression
// ---------------------------------------------------------------------------

/// Compress a field into the bitstream with the given expert parameters.
pub(crate) fn compress(
    bs: &mut (impl ZfpBitStreamMutOps + ?Sized),
    field: &ZfpField,
    config: &ZfpConfig,
) -> Result<usize, ZfpCompressionError> {
    let info = field.plan()?;
    compress_planned(bs, &info, field.data(), config)
}

/// [`compress`] of a field `info` planned, whose data buffer is `buf`.
fn compress_planned(
    bs: &mut (impl ZfpBitStreamMutOps + ?Sized),
    info: &FieldPlan,
    buf: &[u8],
    config: &ZfpConfig,
) -> Result<usize, ZfpCompressionError> {
    // SAFETY: `buf` is the buffer `FieldPlan::new` validated.
    unsafe { compress_blocks(bs, buf.as_ptr(), info, config, 0..info.num_blocks) };
    finish(bs)
}

/// Flush the stream and return its size, or the overflow error.
fn finish(bs: &mut (impl ZfpBitStreamMutOps + ?Sized)) -> Result<usize, ZfpCompressionError> {
    bs.flush();
    if bs.overflowed() {
        return Err(ZfpCompressionError::BufferTooSmall {
            required: bs.byte_len(),
            capacity: bs.capacity(),
        });
    }
    Ok(bs.byte_len())
}

/// Encode blocks `range` into the bitstream, in order.
///
/// # Safety
/// `base` must point to the start of the field's data buffer: at least
/// `checked_size_bytes()` long, aligned for the field's scalar type, and
/// readable for the duration of the call. `FieldPlan::new` validates both
/// properties, so deriving `base` from a field it accepted satisfies this.
// The alignment check is per field, so this cast is not repeated per block.
#[allow(clippy::cast_ptr_alignment)]
unsafe fn compress_blocks(
    bs: &mut (impl ZfpBitStreamMutOps + ?Sized),
    base: *const u8,
    info: &FieldPlan,
    config: &ZfpConfig,
    range: Range<usize>,
) {
    // SAFETY: the caller's contract, with the pointer cast to the scalar type
    // `FieldPlan::new` checked its alignment for.
    unsafe {
        match info.scalar_type {
            ZfpScalarType::I32 => compress_typed(bs, base.cast::<i32>(), info, config, range),
            ZfpScalarType::I64 => compress_typed(bs, base.cast::<i64>(), info, config, range),
            ZfpScalarType::F32 => compress_typed(bs, base.cast::<f32>(), info, config, range),
            ZfpScalarType::F64 => compress_typed(bs, base.cast::<f64>(), info, config, range),
        }
    }
}

/// [`compress_blocks`] for one scalar type.
///
/// # Safety
/// As for [`compress_blocks`], with `base` cast to `T`.
unsafe fn compress_typed<T: ZfpScalar>(
    bs: &mut (impl ZfpBitStreamMutOps + ?Sized),
    base: *const T,
    info: &FieldPlan,
    config: &ZfpConfig,
    range: Range<usize>,
) {
    use crate::codec::block::{encode_lossy, encode_lossy_partial, encode_reversible};

    let dims = info.dims_enum;
    let strides = &info.strides;
    let reversible = config.is_reversible();
    for coords in info.blocks(range) {
        let (offset, lengths) = info.block_geometry(coords);
        // SAFETY: `base` is the field's whole data buffer, and `offset` is
        // the block origin measured from the *lowest* address of the strided
        // span, so every offset the strides generate from it lands inside the
        // buffer.
        unsafe {
            let block = base.add(offset);
            if reversible {
                encode_reversible(bs, block, dims, strides, lengths, config);
            } else if info.is_full(lengths) {
                encode_lossy(bs, block, dims, strides, config);
            } else {
                encode_lossy_partial(bs, block, dims, lengths, strides, config);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Parallel compression (rayon feature)
// ---------------------------------------------------------------------------

/// Parallel compression via Rayon.
///
/// Splits block indices into C-OMP-compatible chunks, compresses each chunk
/// into a local bitstream, then concatenates results at bit-level granularity
/// to match the C OMP implementation's `stream_copy` behavior.
///
/// Compresses serially if the pool, the chunks or their buffers cannot be
/// created. Until every chunk has succeeded nothing is written to `bs`, so the
/// serial fallback starts from a clean stream.
#[cfg(feature = "rayon")]
pub(crate) fn compress_rayon(
    bs: &mut (impl ZfpBitStreamMutOps + ?Sized),
    field: &ZfpField,
    config: &ZfpConfig,
    threads: u32,
    chunk_size: u32,
) -> Result<usize, ZfpCompressionError> {
    let info = field.plan()?;
    let buf = field.data();
    let Some(chunks) = compress_chunks(&info, buf, config, threads, chunk_size) else {
        return compress_planned(bs, &info, buf, config);
    };

    // Concatenate chunks at bit-level granularity, matching C's stream_copy.
    // Write chunks sequentially (no seeking) to avoid buffer clobbering.
    for (bits_written, chunk_words) in chunks.iter().flatten() {
        append_bits(bs, chunk_words, *bits_written);
    }

    finish(bs)
}

/// A compressed chunk of blocks: its bit count and its words.
#[cfg(feature = "rayon")]
type Chunk = (u64, Vec<u64>);

/// Compress each chunk of blocks into its own buffer, in parallel.
///
/// Returns `None` if there are no blocks, or the pool, the chunks or their
/// buffers cannot be created. Otherwise every entry is `Some`: its words and
/// bit count, for bit-level concatenation.
#[cfg(feature = "rayon")]
fn compress_chunks(
    info: &FieldPlan,
    buf: &[u8],
    config: &ZfpConfig,
    threads: u32,
    chunk_size: u32,
) -> Option<Vec<Option<Chunk>>> {
    use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};

    if info.num_blocks == 0 {
        return None;
    }
    let ranges = crate::execution::chunk_ranges(info.num_blocks, threads, chunk_size)?;
    let mut chunk_results = crate::bitstream::vec_with_capacity(ranges.len()).ok()?;
    // `chunk_results` already has room for every chunk.
    let run = || {
        ranges
            .par_iter()
            .map(|range| compress_one_chunk(range.clone(), info, buf, config))
            .collect_into_vec(&mut chunk_results);
    };
    crate::execution::install(threads, run)?;

    // Chunk buffers are sized to fit every block of a valid config, so a chunk
    // fails if its buffer cannot be allocated, or for unvalidated C parameters.
    // Compressing serially needs no buffer, and gets the stream right.
    chunk_results
        .iter()
        .all(Option::is_some)
        .then_some(chunk_results)
}

/// Append the first `bits` bits of `words`, as `copy_from` would, but a word at
/// a time without going through a `dyn` stream. Words past the end read as
/// zero, as they do from a stream.
#[cfg(feature = "rayon")]
#[allow(clippy::cast_possible_truncation)] // `bits / 64` counts words in memory
#[expect(
    clippy::arithmetic_side_effects,
    reason = "`bits / 64` counts whole words and `bits % 64` is below 64"
)]
fn append_bits(bs: &mut (impl ZfpBitStreamMutOps + ?Sized), words: &[u64], bits: u64) {
    let mut w = crate::bitstream::BitWriter::new(bs);
    let word = |i: usize| words.get(i).copied().unwrap_or(0);
    let full = (bits / 64) as usize;
    for i in 0..full {
        w.put(word(i), 64);
    }
    let rest = (bits % 64) as u32;
    if rest != 0 {
        w.put(word(full) & ((1 << rest) - 1), rest);
    }
}

/// Compress blocks `range`, returning (`bits_written`, `compressed_words`),
/// or `None` if the chunk's buffer cannot be allocated or the chunk outgrew it.
///
/// Returns the exact bit count (before flush padding) alongside the flushed
/// words, so the caller can copy at bit-level granularity across chunk boundaries.
#[cfg(feature = "rayon")]
fn compress_one_chunk(
    range: Range<usize>,
    info: &FieldPlan,
    buf: &[u8],
    config: &ZfpConfig,
) -> Option<Chunk> {
    use crate::ZfpBitStream;

    let chunk_bits = u64::try_from(range.len()).ok()?.checked_mul(u64::from(
        config.block_bits(info.scalar_type, info.dims_enum),
    ))?;
    // One word more than the blocks need, for the final flush.
    let chunk_words = (chunk_bits / 64).checked_add(1)?;
    let chunk_bytes = usize::try_from(chunk_words).ok()?.checked_mul(8)?;
    let mut local_bs = ZfpBitStream::new(chunk_bytes).ok()?;
    // SAFETY: `buf` is the field's whole data buffer, validated by `FieldPlan::new`.
    unsafe { compress_blocks(&mut local_bs, buf.as_ptr(), info, config, range) };
    // Record bits written before flushing (flush pads to word boundary).
    let bits_written = local_bs.write_pos();
    local_bs.flush();
    (!local_bs.overflowed()).then(|| (bits_written, local_bs.into_words()))
}
