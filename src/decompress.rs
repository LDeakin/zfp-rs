//! Core decompression logic.
//!
//! Provides the borrow-based implementation used by
//! [`ZfpBitStream::decompress`][crate::ZfpBitStream::decompress].

use crate::bitstream::ZfpBitStreamOps;
use crate::config::ZfpConfig;
use crate::field::ZfpFieldMut;
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
    let info = DecompressInfo::new(field)?;
    for block_idx in 0..info.num_blocks {
        decompress_block(bs, field, &info, config, block_idx);
    }

    bs.align();
    Ok(bs.size())
}

/// Compute block iteration info for decompression.
pub(crate) struct DecompressInfo {
    /// Number of blocks.
    pub num_blocks: usize,
    /// Block grid dimensions.
    pub bx: usize,
    pub by: usize,
    pub bz: usize,
    /// Block count in w-dimension (used to compute `num_blocks`).
    #[allow(dead_code)]
    pub bw: usize,
    /// Dimensionality (1–4).
    pub dim_count: usize,
    /// Element offset minimum (for negative strides).
    pub imin: isize,
    /// Element size in bytes.
    pub elem_size: usize,
    /// Effective strides.
    pub strides: [isize; 4],
    /// Field dimensions `[nx, ny, nz, nw]`.
    pub dims: [usize; 4],
}

impl DecompressInfo {
    #[allow(clippy::cast_possible_wrap, clippy::cast_sign_loss)]
    pub(crate) fn new(field: &mut ZfpFieldMut) -> Result<Self, ZfpDecompressionError> {
        let ty = field.scalar_type();

        if field.data().is_empty() {
            return Err(ZfpDecompressionError::NoData);
        }

        let [nx, ny, nz, nw] = field.dims();
        let dims = field.dimensionality();
        let dim_count = usize::from(dims);
        let strides = field.effective_strides();

        let imin = {
            let mut lo: isize = 0;
            let sizes = field.dims();
            for (s, sz) in strides.iter().zip(sizes.iter()).take(dim_count) {
                if *s < 0 {
                    lo += s * (*sz as isize - 1);
                }
            }
            lo
        };

        let bx = nx.div_ceil(4);
        let by = if dim_count >= 2 { ny.div_ceil(4) } else { 1 };
        let bz = if dim_count >= 3 { nz.div_ceil(4) } else { 1 };
        let bw = if dim_count >= 4 { nw.div_ceil(4) } else { 1 };

        Ok(Self {
            num_blocks: bx * by * bz * bw,
            bx,
            by,
            bz,
            bw,
            dim_count,
            imin,
            elem_size: ty.size(),
            strides,
            dims: [nx, ny, nz, nw],
        })
    }

    #[inline]
    pub(crate) fn block_coords(&self, block_idx: usize) -> (usize, usize, usize, usize) {
        let rem = block_idx;
        let iw = rem / (self.bx * self.by * self.bz);
        let rem = rem % (self.bx * self.by * self.bz);
        let iz = rem / (self.bx * self.by);
        let rem = rem % (self.bx * self.by);
        let iy = rem / self.bx;
        let ix = rem % self.bx;
        (ix, iy, iz, iw)
    }
}

/// Decode a single block from the bitstream.
#[allow(
    clippy::cast_ptr_alignment,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]
#[allow(clippy::ptr_as_ptr)] // data_ptr is already *mut u8 but clippy doesn't know
fn decompress_block(
    bs: &mut dyn ZfpBitStreamOps,
    field: &mut ZfpFieldMut,
    info: &DecompressInfo,
    config: &ZfpConfig,
    block_idx: usize,
) {
    use crate::codec::block::{
        decode_block_strided_reversible, decode_block_strided_with_params,
        decode_partial_block_strided_with_params,
    };

    let (ix, iy, iz, iw) = info.block_coords(block_idx);
    let [nx, ny, nz, nw] = info.dims;

    let elem_off = -info.imin
        + (ix as isize) * 4 * info.strides[0]
        + (iy as isize) * 4 * info.strides[1]
        + (iz as isize) * 4 * info.strides[2]
        + (iw as isize) * 4 * info.strides[3];

    let lx = (nx - ix * 4).min(4);
    let ly = if info.dim_count >= 2 {
        (ny - iy * 4).min(4)
    } else {
        0
    };
    let lz = if info.dim_count >= 3 {
        (nz - iz * 4).min(4)
    } else {
        0
    };
    let lw = if info.dim_count >= 4 {
        (nw - iw * 4).min(4)
    } else {
        0
    };

    let full = lx == 4
        && (info.dim_count < 2 || ly == 4)
        && (info.dim_count < 3 || lz == 4)
        && (info.dim_count < 4 || lw == 4);

    let lengths = [lx, ly, lz, lw];
    let dims = field.dimensionality();
    let data_ptr = field.data_mut().as_mut_ptr();

    let byte_off = (elem_off as usize) * info.elem_size;
    let byte_span = lx * ly * lz * lw * info.elem_size;

    let ty = field.scalar_type();

    macro_rules! decode_dispatch {
        ($($zfp_ty:path => $elem_ty:ty, $div:expr),* $(,)?) => {{
            match ty {
                $(
                    $zfp_ty => {
                        // SAFETY: byte_off + byte_span is within field data bounds.
                        let block: &mut [$elem_ty] = unsafe {
                            std::slice::from_raw_parts_mut(
                                (data_ptr as *mut u8).add(byte_off).cast::<$elem_ty>(),
                                byte_span / $div,
                            )
                        };
                        if config.min_exp() < ZFP_MIN_EXP {
                            decode_block_strided_reversible(
                                bs, block, dims, &info.strides, lengths,
                            );
                        } else if full {
                            decode_block_strided_with_params(
                                bs, block, dims, &info.strides, config.min_bits(),
                                config.max_bits(), config.max_prec(), config.min_exp(),
                            );
                        } else {
                            decode_partial_block_strided_with_params(
                                bs, block, dims, &lengths, &info.strides,
                                config.min_bits(), config.max_bits(), config.max_prec(),
                                config.min_exp(),
                            );
                        }
                    }
                )*
            }
        }};
    }
    decode_dispatch!(
        ZfpScalarType::Int32 => i32, 4,
        ZfpScalarType::Int64 => i64, 8,
        ZfpScalarType::Float => f32, 4,
        ZfpScalarType::Double => f64, 8,
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
/// For non-fixed-rate streams, falls back to serial decompression.
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

    let info = DecompressInfo::new(field)?;
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

    // Capture values needed by parallel closure.
    let dim_count = info.dim_count;
    let imin = info.imin;
    let elem_size = info.elem_size;
    let strides = info.strides;
    let dims_arr = info.dims;
    let dims_enum = field.dimensionality();
    let ty = field.scalar_type();
    // Data pointer as usize (not *mut) so it can be sent to threads.
    let data_ptr = field.data_mut().as_mut_ptr() as usize;

    if threads > 0 {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads as usize)
            .build()
            .expect("rayon thread pool creation failed");
        pool.install(|| {
            decompress_chunks(
                words,
                &chunk_starts,
                &info,
                config,
                blocks,
                bits_per_block,
                start_read_bit,
                dim_count,
                imin,
                elem_size,
                strides,
                dims_arr,
                dims_enum,
                ty,
                data_ptr,
            );
        });
    } else {
        decompress_chunks(
            words,
            &chunk_starts,
            &info,
            config,
            blocks,
            bits_per_block,
            start_read_bit,
            dim_count,
            imin,
            elem_size,
            strides,
            dims_arr,
            dims_enum,
            ty,
            data_ptr,
        );
    }

    Ok(bs.size())
}

/// Run the parallel decompression loop.
#[cfg(feature = "rayon")]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_lines)]
#[allow(
    clippy::cast_ptr_alignment,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]
#[allow(clippy::ptr_as_ptr)] // data_ptr cast from usize needed for thread safety
fn decompress_chunks(
    words: &[u64],
    chunk_starts: &[usize],
    info: &DecompressInfo,
    config: &ZfpConfig,
    blocks: usize,
    bits_per_block: u32,
    start_read_bit: u64,
    dim_count: usize,
    imin: isize,
    elem_size: usize,
    strides: [isize; 4],
    dims_arr: [usize; 4],
    dims_enum: crate::types::ZfpDimensionality,
    ty: ZfpScalarType,
    data_ptr: usize,
) {
    use crate::bitstream::borrowed::ZfpBitStreamRef;
    use crate::codec::block::{
        decode_block_strided_reversible, decode_block_strided_with_params,
        decode_partial_block_strided_with_params,
    };
    use rayon::iter::{IntoParallelIterator, ParallelIterator};

    (0..chunk_starts.len()).into_par_iter().for_each(|chunk| {
        let start_block = chunk_starts[chunk];
        let end_block = if chunk + 1 < chunk_starts.len() {
            chunk_starts[chunk + 1]
        } else {
            blocks
        };

        let mut local_bs = ZfpBitStreamRef::from_words(words);

        for block_idx in start_block..end_block {
            let block_bit_offset = start_read_bit + block_idx as u64 * u64::from(bits_per_block);
            local_bs.seek_read(block_bit_offset);

            let (ix, iy, iz, iw) = info.block_coords(block_idx);
            let [nx, ny, nz, nw] = dims_arr;

            let elem_off = -imin
                + (ix as isize) * 4 * strides[0]
                + (iy as isize) * 4 * strides[1]
                + (iz as isize) * 4 * strides[2]
                + (iw as isize) * 4 * strides[3];

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
            let lengths = [lx, ly, lz, lw];
            let full = lx == 4
                && (dim_count < 2 || ly == 4)
                && (dim_count < 3 || lz == 4)
                && (dim_count < 4 || lw == 4);

            let byte_off = (elem_off as usize) * elem_size;
            let byte_span = lx * ly * lz * lw * elem_size;

            // SAFETY: byte_off + byte_span is within field data bounds.
            // Blocks are non-overlapping, so each thread writes distinct memory.
            match ty {
                ZfpScalarType::Int32 => {
                    let block: &mut [i32] = unsafe {
                        std::slice::from_raw_parts_mut(
                            (data_ptr as *mut u8).add(byte_off).cast::<i32>(),
                            byte_span / 4,
                        )
                    };
                    if config.min_exp() < ZFP_MIN_EXP {
                        decode_block_strided_reversible(
                            &mut local_bs,
                            block,
                            dims_enum,
                            &strides,
                            lengths,
                        );
                    } else if full {
                        decode_block_strided_with_params(
                            &mut local_bs,
                            block,
                            dims_enum,
                            &strides,
                            config.min_bits(),
                            config.max_bits(),
                            config.max_prec(),
                            config.min_exp(),
                        );
                    } else {
                        decode_partial_block_strided_with_params(
                            &mut local_bs,
                            block,
                            dims_enum,
                            &lengths,
                            &strides,
                            config.min_bits(),
                            config.max_bits(),
                            config.max_prec(),
                            config.min_exp(),
                        );
                    }
                }
                ZfpScalarType::Int64 => {
                    let block: &mut [i64] = unsafe {
                        std::slice::from_raw_parts_mut(
                            (data_ptr as *mut u8).add(byte_off).cast::<i64>(),
                            byte_span / 8,
                        )
                    };
                    if config.min_exp() < ZFP_MIN_EXP {
                        decode_block_strided_reversible(
                            &mut local_bs,
                            block,
                            dims_enum,
                            &strides,
                            lengths,
                        );
                    } else if full {
                        decode_block_strided_with_params(
                            &mut local_bs,
                            block,
                            dims_enum,
                            &strides,
                            config.min_bits(),
                            config.max_bits(),
                            config.max_prec(),
                            config.min_exp(),
                        );
                    } else {
                        decode_partial_block_strided_with_params(
                            &mut local_bs,
                            block,
                            dims_enum,
                            &lengths,
                            &strides,
                            config.min_bits(),
                            config.max_bits(),
                            config.max_prec(),
                            config.min_exp(),
                        );
                    }
                }
                ZfpScalarType::Float => {
                    let block: &mut [f32] = unsafe {
                        std::slice::from_raw_parts_mut(
                            (data_ptr as *mut u8).add(byte_off).cast::<f32>(),
                            byte_span / 4,
                        )
                    };
                    if config.min_exp() < ZFP_MIN_EXP {
                        decode_block_strided_reversible(
                            &mut local_bs,
                            block,
                            dims_enum,
                            &strides,
                            lengths,
                        );
                    } else if full {
                        decode_block_strided_with_params(
                            &mut local_bs,
                            block,
                            dims_enum,
                            &strides,
                            config.min_bits(),
                            config.max_bits(),
                            config.max_prec(),
                            config.min_exp(),
                        );
                    } else {
                        decode_partial_block_strided_with_params(
                            &mut local_bs,
                            block,
                            dims_enum,
                            &lengths,
                            &strides,
                            config.min_bits(),
                            config.max_bits(),
                            config.max_prec(),
                            config.min_exp(),
                        );
                    }
                }
                ZfpScalarType::Double => {
                    let block: &mut [f64] = unsafe {
                        std::slice::from_raw_parts_mut(
                            (data_ptr as *mut u8).add(byte_off).cast::<f64>(),
                            byte_span / 8,
                        )
                    };
                    if config.min_exp() < ZFP_MIN_EXP {
                        decode_block_strided_reversible(
                            &mut local_bs,
                            block,
                            dims_enum,
                            &strides,
                            lengths,
                        );
                    } else if full {
                        decode_block_strided_with_params(
                            &mut local_bs,
                            block,
                            dims_enum,
                            &strides,
                            config.min_bits(),
                            config.max_bits(),
                            config.max_prec(),
                            config.min_exp(),
                        );
                    } else {
                        decode_partial_block_strided_with_params(
                            &mut local_bs,
                            block,
                            dims_enum,
                            &lengths,
                            &strides,
                            config.min_bits(),
                            config.max_bits(),
                            config.max_prec(),
                            config.min_exp(),
                        );
                    }
                }
            }
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
