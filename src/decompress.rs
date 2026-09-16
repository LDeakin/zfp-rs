//! Core decompression logic.
//!
//! Provides the borrow-based implementation used by
//! [`ZfpBitStream::decompress`][crate::ZfpBitStream::decompress].

use crate::bitstream::ZfpBitStreamOps;
use crate::config::ZfpConfig;
use crate::field::ZfpFieldMut;
use crate::field_plan::{FieldPlan, PlanError};
use crate::types::{ZFP_MIN_EXP, ZfpDecompressionError, ZfpScalarType};

// ---------------------------------------------------------------------------
// Serial decompression
// ---------------------------------------------------------------------------

/// Decompress from the bitstream into the field with the given expert parameters.
pub(crate) fn decompress(
    bs: &mut dyn ZfpBitStreamOps,
    field: &mut ZfpFieldMut,
    config: &ZfpConfig,
) -> Result<usize, ZfpDecompressionError> {
    let info = plan_mut(field)?;
    // Derived once: a fresh `&mut` retag per block would be needless work, and
    // interleaving it with shared borrows of `field` is a hazard worth avoiding.
    let base = field.data_mut().as_mut_ptr();
    for block_idx in 0..info.num_blocks {
        // SAFETY: `FieldPlan::new` validated the buffer's length and
        // alignment, which is exactly `decompress_block`'s contract.
        unsafe { decompress_block(bs, base, &info, config, block_idx) };
    }

    bs.align();
    Ok(bs.size())
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

impl From<PlanError> for ZfpDecompressionError {
    fn from(e: PlanError) -> Self {
        match e {
            PlanError::NoData => ZfpDecompressionError::NoData,
            PlanError::InvalidField { required, actual } => {
                ZfpDecompressionError::InvalidField { required, actual }
            }
            PlanError::MisalignedData { align } => ZfpDecompressionError::MisalignedData { align },
        }
    }
}

/// Decode a single block from the bitstream.
///
/// # Safety
/// `base` must point to the start of the field's data buffer: at least
/// `checked_size_bytes()` long, aligned for the field's scalar type, and
/// writable for the duration of the call. `FieldPlan::new` validates both
/// properties, so deriving `base` from a field it accepted satisfies this.
// `FieldPlan::new` rejects a field whose buffer is not aligned for its scalar
// type, so these casts are checked once per field rather than once per block.
#[allow(clippy::cast_ptr_alignment)]
#[allow(clippy::cast_possible_wrap, clippy::cast_sign_loss)]
unsafe fn decompress_block(
    bs: &mut dyn ZfpBitStreamOps,
    base: *mut u8,
    info: &FieldPlan,
    config: &ZfpConfig,
    block_idx: usize,
) {
    use crate::codec::block::{
        decode_block_strided, decode_block_strided_reversible, decode_partial_block_strided,
    };

    let (ix, iy, iz, iw) = info.block_coords(block_idx);
    let [nx, ny, nz, nw] = info.dims;
    let dim_count = info.dim_count();

    let elem_off = -info.imin
        + (ix as isize) * 4 * info.strides[0]
        + (iy as isize) * 4 * info.strides[1]
        + (iz as isize) * 4 * info.strides[2]
        + (iw as isize) * 4 * info.strides[3];

    let lx = (nx - ix * 4).min(4);
    let ly = if dim_count >= 2 {
        (ny - iy * 4).min(4)
    } else {
        0
    };
    let lz = if dim_count >= 3 {
        (nz - iz * 4).min(4)
    } else {
        0
    };
    let lw = if dim_count >= 4 {
        (nw - iw * 4).min(4)
    } else {
        0
    };

    let full = lx == 4
        && (dim_count < 2 || ly == 4)
        && (dim_count < 3 || lz == 4)
        && (dim_count < 4 || lw == 4);

    let lengths = [lx, ly, lz, lw];
    let dims = info.dims_enum;

    let byte_off = (elem_off as usize) * info.elem_size();

    let ty = info.scalar_type;

    macro_rules! decode_dispatch {
        ($($zfp_ty:path => $elem_ty:ty),* $(,)?) => {{
            match ty {
                $(
                    $zfp_ty => {
                        // SAFETY: `base` is the field's whole data buffer, which
                        // `FieldPlan::new` validated to be at least
                        // `checked_size_bytes()` long and aligned for the scalar
                        // type. `byte_off` is the block origin measured from the
                        // *lowest* address of the strided span (`elem_off`
                        // includes the `-imin` shift), so every offset the
                        // strides generate from it lands inside the buffer.
                        let block = unsafe { base.add(byte_off).cast::<$elem_ty>() };
                        unsafe {
                            if config.min_exp() < ZFP_MIN_EXP {
                                decode_block_strided_reversible(
                                    bs, block, dims, &info.strides, lengths,
                                );
                            } else if full {
                                decode_block_strided(
                                    bs, block, dims, &info.strides, config.min_bits(),
                                    config.max_bits(), config.max_prec(), config.min_exp(),
                                );
                            } else {
                                decode_partial_block_strided(
                                    bs, block, dims, &lengths, &info.strides,
                                    config.min_bits(), config.max_bits(), config.max_prec(),
                                    config.min_exp(),
                                );
                            }
                        }
                    }
                )*
            }
        }};
    }
    decode_dispatch!(
        ZfpScalarType::Int32 => i32,
        ZfpScalarType::Int64 => i64,
        ZfpScalarType::Float => f32,
        ZfpScalarType::Double => f64,
    );
}

// ---------------------------------------------------------------------------
// Parallel decompression (rayon feature)
// ---------------------------------------------------------------------------

/// Parallel decompression via Rayon.
///
/// For **fixed-rate** streams, each thread creates an independent bitstream
/// view and seeks directly to its block's bit position.
///
/// Falls back to serial decompression for non-fixed-rate streams, and for
/// fields whose strides may alias: two blocks writing the same element would
/// race.
#[cfg(feature = "rayon")]
pub(crate) fn decompress_rayon(
    bs: &mut dyn ZfpBitStreamOps,
    field: &mut ZfpFieldMut,
    config: &ZfpConfig,
    threads: u32,
    chunk_size: u32,
) -> Result<usize, ZfpDecompressionError> {
    use crate::config::compression_mode_from_params;

    let is_fixed_rate = compression_mode_from_params(
        config.min_bits(),
        config.max_bits(),
        config.max_prec(),
        config.min_exp(),
    ) == crate::types::ZfpMode::FixedRate;

    if !is_fixed_rate {
        return decompress(bs, field, config);
    }

    let info = plan_mut(field)?;
    if info.strides_may_alias() {
        return decompress(bs, field, config);
    }

    let blocks = info.num_blocks;
    if blocks == 0 {
        return Ok(0);
    }

    let bits_per_block = config.max_bits();
    let start_read_bit = bs.read_pos();
    let (_chunks, chunk_starts) = decompress_compute_chunk_ranges(blocks, threads, chunk_size);

    // Shared read-only slice of all words in the bitstream buffer.
    // `word_pos` tracks the current read cursor (reset by rewind), so we use
    // the full buffer length to cover all compressed data.
    let words = bs.words();

    let base = FieldPtr(field.data_mut().as_mut_ptr());

    let run = || {
        decompress_chunks(
            words,
            &chunk_starts,
            &info,
            config,
            bits_per_block,
            start_read_bit,
            base,
        );
    };
    if threads > 0 {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads as usize)
            .build()
            .expect("rayon thread pool creation failed");
        pool.install(run);
    } else {
        run();
    }

    Ok(bs.size())
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
    chunk_starts: &[usize],
    info: &FieldPlan,
    config: &ZfpConfig,
    bits_per_block: u32,
    start_read_bit: u64,
    base: FieldPtr,
) {
    use crate::bitstream::borrowed::ZfpBitStreamRef;
    use rayon::iter::{IntoParallelIterator, ParallelIterator};

    (0..chunk_starts.len()).into_par_iter().for_each(|chunk| {
        let start_block = chunk_starts[chunk];
        let end_block = if chunk + 1 < chunk_starts.len() {
            chunk_starts[chunk + 1]
        } else {
            info.num_blocks
        };

        let mut local_bs = ZfpBitStreamRef::from_words(words);

        for block_idx in start_block..end_block {
            local_bs.seek_read(start_read_bit + block_idx as u64 * u64::from(bits_per_block));
            // SAFETY: `base` is the buffer `FieldPlan::new` validated for
            // length and alignment, and chunks cover disjoint blocks, so this
            // thread writes only bytes no other thread touches.
            unsafe { decompress_block(&mut local_bs, base.ptr(), info, config, block_idx) };
        }
    });
}

/// Compute chunk ranges for decompression following C OMP semantics.
#[cfg(feature = "rayon")]
#[allow(clippy::cast_sign_loss)]
fn decompress_compute_chunk_ranges(
    blocks: usize,
    threads: u32,
    chunk_size: u32,
) -> (usize, Vec<usize>) {
    let threads = if threads > 0 {
        threads as usize
    } else {
        rayon::current_num_threads()
    };

    let chunks = if chunk_size > 0 {
        let c = blocks.div_ceil(chunk_size as usize);
        c.min(blocks)
    } else {
        threads.min(blocks)
    };

    let chunks = chunks.max(1);
    let mut starts = Vec::with_capacity(chunks);
    for chunk in 0..chunks {
        starts.push((blocks * chunk) / chunks);
    }
    (chunks, starts)
}

#[cfg(all(test, feature = "rayon"))]
mod tests {
    use crate::config::{ZfpConfig, ZfpStreamAlignment};
    use crate::execution::ZfpExecution;
    use crate::types::{ZfpDimensionality, ZfpScalarType};
    use crate::{ZfpBitStream, ZfpField, ZfpFieldMut};

    /// `[1, 1]` maps index `[x, y]` to `x + y`, so the span is 15 elements.
    fn decode(bs: &mut ZfpBitStream, config: &ZfpConfig, execution: ZfpExecution) -> [u64; 15] {
        let mut out = [0f64; 15];
        let mut field = ZfpFieldMut::new_strided(&mut out, [8usize, 8], [1isize, 1]);
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
            ZfpScalarType::Double,
            ZfpDimensionality::D2,
            ZfpStreamAlignment::None,
        );
        let src: Vec<f64> = (0..64).map(f64::from).collect();
        let mut bs = ZfpBitStream::new(4096);
        bs.compress(&config, &ZfpField::new(&src, [8usize, 8]))
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
}
