//! Core decompression logic.
//!
//! Provides the borrow-based implementation used by
//! [`ZfpBitStream::decompress`][crate::ZfpBitStream::decompress].

// The API and validation layer computes with caller-supplied sizes, so its
// arithmetic and indexing must be checked; see the crate's panic guarantee.
#![warn(clippy::indexing_slicing, clippy::arithmetic_side_effects)]

use crate::bitstream::ZfpBitStreamOps;
use crate::config::ZfpConfig;
use crate::field::ZfpFieldMut;
use crate::field_plan::FieldPlan;
use crate::types::{ZFP_MIN_EXP, ZfpDecompressionError, ZfpScalar, ZfpScalarType};
use std::ops::Range;

// ---------------------------------------------------------------------------
// Serial decompression
// ---------------------------------------------------------------------------

/// Decompress from the bitstream into the field with the given expert parameters.
pub(crate) fn decompress(
    bs: &mut (impl ZfpBitStreamOps + ?Sized),
    field: &mut ZfpFieldMut,
    config: &ZfpConfig,
) -> Result<usize, ZfpDecompressionError> {
    let info = plan_mut(field)?;
    // Derived once: a fresh `&mut` retag per block would be needless work, and
    // interleaving it with shared borrows of `field` is a hazard worth avoiding.
    let base = field.data_mut().as_mut_ptr();
    // SAFETY: `FieldPlan::new` validated the buffer's length and alignment,
    // which is exactly `decompress_blocks`' contract.
    unsafe { decompress_blocks(bs, base, &info, config, 0..info.num_blocks, None) };

    bs.align();
    Ok(bs.byte_len())
}

/// Derive the block plan for a field, mapping the layout error.
fn plan_mut(field: &ZfpFieldMut) -> Result<FieldPlan, ZfpDecompressionError> {
    Ok(FieldPlan::new(
        field.scalar_type(),
        field.dims(),
        field.dimensionality(),
        field.effective_strides(),
        field.data(),
        field.checked_size_bytes().unwrap_or(usize::MAX),
    )?)
}

/// Decode blocks `range` from the bitstream, in order.
///
/// With `seek = Some((start, bits))`, block `i` is read from bit
/// `start + i * bits` rather than where the previous block ended.
///
/// # Safety
/// `base` must point to the start of the field's data buffer: at least
/// `checked_size_bytes()` long, aligned for the field's scalar type, and
/// writable for the duration of the call. `FieldPlan::new` validates both
/// properties, so deriving `base` from a field it accepted satisfies this.
// `FieldPlan::new` rejects a field whose buffer is not aligned for its scalar
// type, so this cast is checked once per field rather than once per block.
#[allow(clippy::cast_ptr_alignment)]
unsafe fn decompress_blocks(
    bs: &mut (impl ZfpBitStreamOps + ?Sized),
    base: *mut u8,
    info: &FieldPlan,
    config: &ZfpConfig,
    range: Range<usize>,
    seek: Option<(u64, u64)>,
) {
    // SAFETY: the caller's contract, with the pointer cast to the scalar type
    // `FieldPlan::new` checked its alignment for.
    unsafe {
        match info.scalar_type {
            ZfpScalarType::I32 => {
                decompress_typed(bs, base.cast::<i32>(), info, config, range, seek);
            }
            ZfpScalarType::I64 => {
                decompress_typed(bs, base.cast::<i64>(), info, config, range, seek);
            }
            ZfpScalarType::F32 => {
                decompress_typed(bs, base.cast::<f32>(), info, config, range, seek);
            }
            ZfpScalarType::F64 => {
                decompress_typed(bs, base.cast::<f64>(), info, config, range, seek);
            }
        }
    }
}

/// [`decompress_blocks`] for one scalar type.
///
/// # Safety
/// As for [`decompress_blocks`], with `base` cast to `T`.
#[expect(
    clippy::arithmetic_side_effects,
    reason = "`decompress_rayon` checked that the offset past the last block fits `u64`"
)]
unsafe fn decompress_typed<T: ZfpScalar>(
    bs: &mut (impl ZfpBitStreamOps + ?Sized),
    base: *mut T,
    info: &FieldPlan,
    config: &ZfpConfig,
    range: Range<usize>,
    seek: Option<(u64, u64)>,
) {
    use crate::codec::block::{decode_block_strided, decode_partial, decode_reversible};

    let dims = info.dims_enum;
    let strides = &info.strides;
    // The strided decoders check this per block; hoisting it is faster.
    let reversible = config.min_exp() < ZFP_MIN_EXP;
    for (block_idx, coords) in range.clone().zip(info.blocks(range)) {
        if let Some((start, bits)) = seek {
            bs.seek_read(start + block_idx as u64 * bits);
        }
        let (offset, lengths) = info.block_geometry(coords);
        // SAFETY: `base` is the field's whole data buffer, which
        // `FieldPlan::new` validated to be at least `checked_size_bytes()`
        // long and aligned for the scalar type. `offset` is the block origin
        // measured from the *lowest* address of the strided span, so every
        // offset the strides generate from it lands inside the buffer.
        unsafe {
            let block = base.add(offset);
            if reversible {
                decode_reversible(bs, block, dims, strides, lengths, config);
            } else if info.is_full(lengths) {
                decode_block_strided(bs, block, dims, strides, config);
            } else {
                decode_partial(bs, block, dims, lengths, strides, config);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Parallel decompression (rayon feature)
// ---------------------------------------------------------------------------

/// Parallel decompression via Rayon.
///
/// For **fixed-rate** streams, each thread creates an independent bitstream
/// view and seeks directly to its block's bit position.
///
/// Falls back to serial decompression when block sizes can vary, when field
/// strides may alias (two blocks writing the same element would race), when
/// the blocks would run past the largest bit offset, or when the pool or the
/// chunks cannot be created.
#[cfg(feature = "rayon")]
pub(crate) fn decompress_rayon(
    bs: &mut (impl ZfpBitStreamOps + ?Sized),
    field: &mut ZfpFieldMut,
    config: &ZfpConfig,
    threads: u32,
    chunk_size: u32,
) -> Result<usize, ZfpDecompressionError> {
    let is_fixed_rate = config.mode() == crate::types::ZfpMode::FixedRate;

    if !is_fixed_rate {
        return decompress(bs, field, config);
    }

    let info = plan_mut(field)?;
    // A fixed-rate config only gives each block a fixed physical size if its
    // budget can hold the type's block header. Below 9 bits for f32 or 12 for
    // f64, zero blocks use max_bits but nonzero blocks write the full header.
    // The maximum-size calculation already accounts for this distinction.
    if info.strides_may_alias()
        || config.block_bits(info.scalar_type, info.dims_enum) != config.max_bits()
    {
        return decompress(bs, field, config);
    }

    let blocks = info.num_blocks;
    if blocks == 0 {
        return decompress(bs, field, config);
    }
    let bits_per_block = config.max_bits();
    let start_read_bit = bs.read_pos();
    // Where serial decompression would leave the cursor. Bounding it bounds
    // every block's offset, so the per-block seeks cannot overflow.
    let Some(end) = u64::try_from(blocks)
        .ok()
        .and_then(|blocks| blocks.checked_mul(u64::from(bits_per_block)))
        .and_then(|bits| bits.checked_add(start_read_bit))
    else {
        return decompress(bs, field, config);
    };
    let Some(ranges) = crate::execution::chunk_ranges(blocks, threads, chunk_size) else {
        return decompress(bs, field, config);
    };

    // Shared read-only slice of all words in the bitstream buffer.
    // `word_pos` tracks the current read cursor (reset by rewind), so we use
    // the full buffer length to cover all compressed data.
    let words = bs.backing_words();

    let base = FieldPtr(field.data_mut().as_mut_ptr());

    let run = || {
        decompress_chunks(
            words,
            &ranges,
            &info,
            config,
            bits_per_block,
            start_read_bit,
            base,
        );
    };
    // The pool is built before any block is decoded, so if it cannot be, the
    // serial decoder fills the whole field.
    if crate::execution::install(threads, run).is_none() {
        return decompress(bs, field, config);
    }

    // Leave the cursor where serial decompression would.
    bs.seek_read(end);
    bs.align();
    Ok(bs.byte_len())
}

/// A `*mut u8` into the field's data buffer, handed to worker threads.
///
/// Carrying the real pointer rather than a `usize` round-trip keeps its
/// provenance intact, which `-Zmiri-strict-provenance` requires.
#[cfg(feature = "rayon")]
#[derive(Clone, Copy)]
struct FieldPtr(*mut u8);

#[cfg(feature = "rayon")]
impl FieldPtr {
    /// Take `self` by value so closures capture the whole `FieldPtr` rather
    /// than the bare `*mut u8` inside it: precise capture would otherwise grab
    /// `base.0`, and `&*mut u8` is not `Sync`.
    fn ptr(self) -> *mut u8 {
        self.0
    }
}

// SAFETY: chunks cover disjoint ranges of block indices, and `decompress_rayon`
// only takes this path for non-aliasing strides, under which blocks map to
// disjoint byte ranges. So no two threads write the same byte and no thread
// reads a byte another thread writes.
#[cfg(feature = "rayon")]
unsafe impl Send for FieldPtr {}
#[cfg(feature = "rayon")]
unsafe impl Sync for FieldPtr {}

/// Decompress chunks of blocks in parallel.
///
/// Each chunk gets its own bitstream view and seeks directly to its blocks' bit
/// positions, then defers to the same `decompress_block` the serial path uses.
#[cfg(feature = "rayon")]
fn decompress_chunks(
    words: &[u64],
    ranges: &[Range<usize>],
    info: &FieldPlan,
    config: &ZfpConfig,
    bits_per_block: u32,
    start_read_bit: u64,
    base: FieldPtr,
) {
    use crate::bitstream::borrowed::ZfpBitStreamRef;
    use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

    ranges.par_iter().for_each(|range| {
        let mut local_bs = ZfpBitStreamRef::from_words(words);
        // SAFETY: `base` is the buffer `FieldPlan::new` validated for length
        // and alignment, and chunks cover disjoint blocks, so this thread
        // writes only bytes no other thread touches.
        unsafe {
            decompress_blocks(
                &mut local_bs,
                base.ptr(),
                info,
                config,
                range.clone(),
                Some((start_read_bit, u64::from(bits_per_block))),
            );
        }
    });
}

#[cfg(test)]
#[cfg(feature = "rayon")]
mod tests {
    use crate::config::{ZfpConfig, ZfpStreamAlignment};
    use crate::execution::ZfpExecution;
    use crate::types::{ZfpDimensionality, ZfpScalarType};
    use crate::{ZfpBitStream, ZfpField, ZfpFieldMut};

    /// `[1, 1]` maps index `[x, y]` to `x + y`, so the span is 15 elements.
    fn decode(bs: &mut ZfpBitStream, config: &ZfpConfig, execution: ZfpExecution) -> [u64; 15] {
        let mut out = [0f64; 15];
        let mut field = ZfpFieldMut::new_strided(&mut out, [8usize, 8], [1isize, 1]).unwrap();
        bs.rewind();
        bs.decompress_with_execution(config, &mut field, execution)
            .unwrap();
        out.map(f64::to_bits)
    }

    /// Aliasing strides make two blocks write the same element, which the
    /// parallel path cannot do without a data race. It must fall back to
    /// serial, so its output matches serial decompression exactly.
    #[test]
    fn aliasing_strides_decompress_serially() {
        let config = ZfpConfig::fixed_rate(
            16.0,
            ZfpScalarType::F64,
            ZfpDimensionality::D2,
            ZfpStreamAlignment::Unaligned,
        )
        .unwrap();
        let src: Vec<f64> = (0..64).map(f64::from).collect();
        let mut bs = ZfpBitStream::new(4096).unwrap();
        bs.compress(&config, &ZfpField::new(&src, [8usize, 8]).unwrap())
            .unwrap();

        let parallel = decode(
            &mut bs,
            &config,
            ZfpExecution::Rayon {
                threads: 4,
                chunk_size: 1,
            },
        );
        assert_eq!(parallel, decode(&mut bs, &config, ZfpExecution::Serial));
    }

    /// Block offsets past `u64::MAX` overflowed the per-block seeks; they now
    /// decompress serially.
    #[test]
    fn blocks_past_the_largest_offset_decompress_serially() {
        let config = ZfpConfig::fixed_rate(
            16.0,
            ZfpScalarType::F64,
            ZfpDimensionality::D2,
            ZfpStreamAlignment::Unaligned,
        )
        .unwrap();
        let mut bs = ZfpBitStream::new(64).unwrap();
        let decode = |bs: &mut ZfpBitStream, execution| {
            bs.seek_read(u64::MAX - 100);
            let mut out = [1f64; 64];
            let mut field = ZfpFieldMut::new(&mut out, [8usize, 8]).unwrap();
            let read = bs
                .decompress_with_execution(&config, &mut field, execution)
                .unwrap();
            (read, bs.read_pos(), out.map(f64::to_bits))
        };
        let serial = decode(&mut bs, ZfpExecution::Serial);
        let parallel = ZfpExecution::Rayon {
            threads: 2,
            chunk_size: 1,
        };
        assert_eq!(decode(&mut bs, parallel), serial);
    }

    /// A truncated stream decodes as if the missing words were zeros, in
    /// parallel as serially, and both end where the whole stream does. The
    /// parallel path's final seek once clamped to the buffer instead.
    #[test]
    fn truncated_stream_decompresses_as_if_zero_padded() {
        let config = ZfpConfig::fixed_rate(
            16.0,
            ZfpScalarType::F64,
            ZfpDimensionality::D2,
            ZfpStreamAlignment::Unaligned,
        )
        .unwrap();
        let src: Vec<f64> = (0..64).map(f64::from).collect();
        let mut bs = ZfpBitStream::new(4096).unwrap();
        let size = bs
            .compress(&config, &ZfpField::new(&src, [8usize, 8]).unwrap())
            .unwrap();
        let kept = size / 2;
        let mut padded = bs.as_bytes()[..kept].to_vec();
        padded.resize(size, 0);

        let decode = |bytes: &[u8], execution| {
            let mut out = [0f64; 64];
            let mut field = ZfpFieldMut::new(&mut out, [8usize, 8]).unwrap();
            let read = ZfpBitStream::from_bytes(bytes)
                .unwrap()
                .decompress_with_execution(&config, &mut field, execution)
                .unwrap();
            (read, out.map(f64::to_bits))
        };
        let expect = decode(&padded, ZfpExecution::Serial);
        assert_eq!(expect.0, size);
        let truncated = &bs.as_bytes()[..kept];
        assert_eq!(decode(truncated, ZfpExecution::Serial), expect);
        let parallel = ZfpExecution::Rayon {
            threads: 4,
            chunk_size: 1,
        };
        assert_eq!(decode(truncated, parallel), expect);
    }
}
