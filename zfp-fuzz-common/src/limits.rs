//! Bounds that keep fuzz inputs cheap.
//!
//! Without these the fuzzer spends nearly all of its time in multi-GB
//! allocations and multi-second encodes instead of exploring the codec.

/// Maximum number of scalars in a generated field (32 KiB at `f64`).
pub const MAX_ELEMENTS: usize = 4096;

/// Maximum compressed-stream capacity a target will allocate.
///
/// Because [`ZfpConfig::maximum_size`] is `num_blocks * max_bits`, capping it
/// bounds runtime as well as memory: a target that skips inputs exceeding this
/// cannot be handed a field that takes seconds to encode.
///
/// [`ZfpConfig::maximum_size`]: zfp_rs::ZfpConfig::maximum_size
pub const MAX_STREAM_BYTES: usize = 1 << 20;

/// Maximum length of a single axis, indexed by `rank - 1`.
///
/// Chosen so the product stays at or under [`MAX_ELEMENTS`] for a cube of the
/// given rank while still allowing long, thin 1-D fields.
pub const MAX_SIDE: [usize; 4] = [256, 64, 16, 8];

/// Maximum absolute stride multiplier used when building strided layouts.
pub const MAX_STRIDE_GAP: u8 = 4;
