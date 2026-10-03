//! Bounded plane-read / reconstruction pipeline. Only the reader touches the
//! compressed stream; workers own disjoint blocks of the validated output.
//! The stages are in [`crate::codec::decode::staged`].

use std::collections::VecDeque;
use std::sync::{Mutex, MutexGuard};

use super::FieldPtr;
use crate::bitstream::{ZfpBitStreamRef, overread, vec_with_capacity};
use crate::codec::decode::staged::{Block, Layout, Scalar, Scatter, read_batch, reconstruct_batch};
use crate::config::ZfpConfig;
use crate::field_plan::FieldPlan;
use crate::types::{ZfpDimensionality, ZfpScalarType};

const BATCH_BYTES: usize = 64 * 1024;
const QUEUE_DEPTH: usize = 16;

/// All buffers and recycling slots are allocated before the first read.
/// A buffer belongs either to this FIFO or to exactly one lease. Rotate free
/// buffers instead of immediately reusing the one a worker just returned;
/// immediate reuse was substantially slower for cheap 1-D precision blocks.
struct BufferPool<B> {
    free: Mutex<VecDeque<Vec<B>>>,
}

impl<B> BufferPool<B> {
    fn new(capacity: usize, depth: usize, mut init: impl FnMut() -> B) -> Option<Self> {
        let mut free = VecDeque::new();
        free.try_reserve_exact(depth).ok()?;
        for _ in 0..depth {
            let mut buffer = allocate_buffer(capacity)?;
            buffer.resize_with(capacity, &mut init);
            free.push_back(buffer);
        }
        Some(Self {
            free: Mutex::new(free),
        })
    }

    fn free(&self) -> MutexGuard<'_, VecDeque<Vec<B>>> {
        // The lock only protects ownership transfers, never codec work.
        // Recovering poison also lets leases return during unwinding.
        self.free
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn acquire(&self) -> BufferLease<'_, B> {
        loop {
            // Drop the lock before yielding: a worker needs it to return a
            // buffer, and yield_now can recursively execute another decoder.
            let buffer = self.free().pop_front();
            if let Some(buffer) = buffer {
                return BufferLease { pool: self, buffer };
            }
            // Blocking a Rayon worker here could deadlock nested decodes.
            rayon::yield_now();
            std::thread::yield_now();
        }
    }
}

struct BufferLease<'a, B> {
    pool: &'a BufferPool<B>,
    buffer: Vec<B>,
}

impl<B> Drop for BufferLease<'_, B> {
    fn drop(&mut self) {
        // The list has room for every buffer. A lease returns its buffer
        // exactly once, so push_back cannot allocate, including during unwinding.
        self.pool.free().push_back(std::mem::take(&mut self.buffer));
    }
}

fn allocate_buffer<B>(capacity: usize) -> Option<Vec<B>> {
    #[cfg(test)]
    if FAIL_BUFFER_AFTER
        .with(|remaining| {
            let n = remaining.get()?;
            remaining.set(n.checked_sub(1));
            Some(n == 0)
        })
        .unwrap_or(false)
    {
        return None;
    }
    vec_with_capacity(capacity).ok()
}

#[cfg(test)]
thread_local! {
    static FAIL_BUFFER_AFTER: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
}

/// A decode ready to run. Everything that can fail is in [`Self::new`], which
/// touches neither the stream nor the field, so a failure leaves them for the
/// serial fallback; [`Self::run`] cannot fail.
struct Pipeline<T: Scalar<N>, const N: usize> {
    buffers: BufferPool<Block<T, N>>,
    /// Blocks per batch.
    batch_len: usize,
}

impl<T: Scalar<N>, const N: usize> Pipeline<T, N>
where
    Layout<N>: Scatter<N>,
{
    /// `batch_size` is the blocks per batch, or 0 for [`BATCH_BYTES`] of planes.
    ///
    /// `None` for a field of one batch, which is read in full before any of it
    /// is reconstructed, so that nothing overlaps and decoding is slower than
    /// serial.
    #[expect(
        clippy::arithmetic_side_effects,
        reason = "a block is never zero-sized, and the batch count is at most num_blocks"
    )]
    fn new(info: &FieldPlan, batch_size: u32) -> Option<Self> {
        let batch_len = if batch_size == 0 {
            (BATCH_BYTES / size_of::<Block<T, N>>()).max(1)
        } else {
            batch_size as usize
        };
        if batch_len >= info.num_blocks {
            return None;
        }
        let depth = QUEUE_DEPTH.min(info.num_blocks.div_ceil(batch_len));
        Some(Self {
            buffers: BufferPool::new(batch_len, depth, Block::new)?,
            batch_len,
        })
    }

    #[expect(
        clippy::arithmetic_side_effects,
        clippy::indexing_slicing,
        reason = "batches are capped at the blocks left, and no more than `batch_len` blocks are read into a buffer of that length"
    )]
    fn run(
        &self,
        bs: &mut ZfpBitStreamRef<'_>,
        base: FieldPtr,
        info: &FieldPlan,
        config: &ZfpConfig,
    ) {
        rayon::scope_fifo(|scope| {
            let mut start = 0;
            while start < info.num_blocks {
                let mut lease = self.buffers.acquire();
                let count = self.batch_len.min(info.num_blocks - start);
                read_batch::<T, N>(bs, &mut lease.buffer[..count], config, info.dims_enum);
                scope.spawn_fifo(move |_| {
                    // SAFETY: `FieldPtr::new` excluded aliasing strides, and
                    // `info` was planned for the buffer `base` points to. Each
                    // batch holds different blocks, and the scope joins every
                    // write before `run` returns.
                    unsafe {
                        reconstruct_batch::<T, N>(
                            &lease.buffer[..count],
                            start,
                            base.ptr(),
                            info,
                            config,
                        );
                    }
                    // Dropping the lease recycles it, also if reconstruction unwinds.
                });
                start += count;
            }
        });
    }
}

/// Build a [`Pipeline`] and run it, or return `None` having done nothing.
fn pipelined<T: Scalar<N>, const N: usize>(
    bs: &mut ZfpBitStreamRef<'_>,
    base: FieldPtr,
    info: &FieldPlan,
    config: &ZfpConfig,
    batch_size: u32,
) -> Option<()>
where
    Layout<N>: Scatter<N>,
{
    Pipeline::<T, N>::new(info, batch_size)?.run(bs, base, info, config);
    Some(())
}

/// Returns None before consuming input or writing output on fallback.
///
/// 1-D blocks are too cheap to reconstruct for handing them to another thread
/// to pay: it decoded between 0.8 and 1.1 times as fast as serial.
pub(super) fn decode(
    words: &[u64],
    start: u64,
    base: FieldPtr,
    info: &FieldPlan,
    config: &ZfpConfig,
    batch_size: u32,
) -> Option<(u64, bool)> {
    if rayon::current_num_threads() < 2 {
        return None;
    }
    let mut bs = ZfpBitStreamRef::from_words(words);
    bs.seek_read(start);
    macro_rules! dispatch {
        ($t:ty) => {
            match info.dims_enum {
                ZfpDimensionality::D1 => None,
                ZfpDimensionality::D2 => {
                    pipelined::<$t, 16>(&mut bs, base, info, config, batch_size)
                }
                ZfpDimensionality::D3 => {
                    pipelined::<$t, 64>(&mut bs, base, info, config, batch_size)
                }
                ZfpDimensionality::D4 => {
                    pipelined::<$t, 256>(&mut bs, base, info, config, batch_size)
                }
            }
        };
    }
    match info.scalar_type {
        ZfpScalarType::I32 => dispatch!(i32),
        ZfpScalarType::I64 => dispatch!(i64),
        ZfpScalarType::F32 => dispatch!(f32),
        ZfpScalarType::F64 => dispatch!(f64),
    }?;
    Some((bs.read_pos(), overread(&bs)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ZfpBitStream, ZfpExecution, ZfpField, ZfpFieldMut};

    #[test]
    fn returned_buffers_rotate_before_being_reused() {
        let mut identity = 0;
        let buffers = BufferPool::new(1, 3, || {
            let next = identity;
            identity += 1;
            next
        })
        .unwrap();
        // Preserve the recycling order that avoids the measured regression
        // when one reconstruction worker can keep up with the reader.
        for expected in (0..3).cycle().take(9) {
            assert_eq!(buffers.acquire().buffer[0], expected);
        }
    }

    #[test]
    fn one_buffer_recycles_through_saturated_scope() {
        let workers = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();
        workers.install(|| {
            let buffers = BufferPool::new(1, 1, || 0usize).unwrap();
            let completed = std::sync::atomic::AtomicUsize::new(0);
            rayon::scope_fifo(|scope| {
                for expected in 0..64 {
                    let mut lease = buffers.acquire();
                    assert_eq!(lease.buffer[0], expected);
                    let completed = &completed;
                    scope.spawn_fifo(move |_| {
                        lease.buffer[0] += 1;
                        completed.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    });
                }
            });
            assert_eq!(completed.load(std::sync::atomic::Ordering::Relaxed), 64);
            assert_eq!(buffers.free().len(), 1);
            assert_eq!(buffers.acquire().buffer[0], 64);
        });
    }

    #[test]
    fn leases_return_on_unwind_and_recover_poison() {
        let buffers = BufferPool::new(1, 1, || 7u32).unwrap();
        let result = std::panic::catch_unwind(|| {
            let _lease = buffers.acquire();
            let _lock = buffers.free();
            panic!("test lease return while the pool lock becomes poisoned");
        });
        assert!(result.is_err());
        assert_eq!(buffers.acquire().buffer, [7]);
        assert_eq!(buffers.free().len(), 1);
    }

    #[test]
    fn allocation_failure_falls_back_to_serial() {
        let data: Vec<i32> = (0..512).map(|i| i * 127 - 19).collect();
        let dims = [32, 16];
        let config = ZfpConfig::reversible();
        let mut stream = ZfpBitStream::new(16_384).unwrap();
        stream.write_bits(0x123, 13);
        stream
            .compress(&config, &ZfpField::new(&data, dims).unwrap())
            .unwrap();
        let mut output = vec![0i32; data.len()];
        let mut reference = ZfpBitStreamRef::from_words(stream.as_words());
        reference.seek_read(13);
        reference
            .decompress(&config, &mut ZfpFieldMut::new(&mut output, dims).unwrap())
            .unwrap();
        let workers = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();
        workers.install(|| {
            for fail_after in [0, 2] {
                let info = ZfpFieldMut::new(&mut output, dims).unwrap().plan().unwrap();
                FAIL_BUFFER_AFTER.set(Some(fail_after));
                assert!(Pipeline::<i32, 16>::new(&info, 1).is_none());
                output.fill(-1);
                let mut reader = ZfpBitStreamRef::from_words(stream.as_words());
                reader.seek_read(13);
                FAIL_BUFFER_AFTER.set(Some(fail_after));
                reader
                    .decompress_with_execution(
                        &config,
                        &mut ZfpFieldMut::new(&mut output, dims).unwrap(),
                        ZfpExecution::Rayon {
                            threads: 0,
                            chunk_size: 1,
                        },
                    )
                    .unwrap();
                assert_eq!(output, data);
                assert_eq!(reader.read_pos(), reference.read_pos());
                // The failed pipeline consumed the injected failure.
                assert_eq!(FAIL_BUFFER_AFTER.get(), None);
            }
        });
    }

    #[test]
    fn one_batch_and_1d_fields_are_not_pipelined() {
        let workers = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();
        workers.install(|| {
            // Four blocks.
            let mut output = vec![0i32; 64];
            let mut field = ZfpFieldMut::new(&mut output, [8, 8]).unwrap();
            let info = field.plan().unwrap();
            assert!(Pipeline::<i32, 16>::new(&info, 4).is_none());
            assert!(Pipeline::<i32, 16>::new(&info, 0).is_none());
            assert!(Pipeline::<i32, 16>::new(&info, 3).is_some());

            let config = ZfpConfig::fixed_precision(16);
            let base = FieldPtr::new(field.data_mut().as_mut_ptr(), &info).unwrap();
            assert!(decode(&[], 0, base, &info, &config, 1).is_some());

            let mut output = vec![0i32; 64];
            let mut field = ZfpFieldMut::new(&mut output, [64]).unwrap();
            let info = field.plan().unwrap();
            let base = FieldPtr::new(field.data_mut().as_mut_ptr(), &info).unwrap();
            assert!(decode(&[], 0, base, &info, &config, 1).is_none());
        });
    }
}
