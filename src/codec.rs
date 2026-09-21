//! Public block-codec API: dispatch by scalar type and dimensionality.
//!
//! [`block`] is the stable surface: [`block::encode_block`] and
//! [`block::decode_block`] are safe and validate their input.
//!
//! The remaining modules are the monomorphised innards. [`encode`] and
//! [`decode`] are public only with the `internals` feature, which the C-port
//! and proptest suites enable; [`promote`] with `ffi`, which the C-ABI layer
//! enables. Otherwise they stay crate-internal. Their strided entry points are
//! `unsafe`: they index through caller-supplied strides without bounds checks.

pub mod block;

#[cfg(feature = "internals")]
pub mod decode;
#[cfg(not(feature = "internals"))]
pub(crate) mod decode;

#[cfg(feature = "internals")]
pub mod encode;
#[cfg(not(feature = "internals"))]
pub(crate) mod encode;

// Type promotion is used only by the C ABI (`zfp_promote_*`) and its tests.
#[cfg(feature = "ffi")]
pub mod promote;

// Never part of the public API: the lifting/decorrelation primitives are used
// only by `encode` and `decode`.
pub(crate) mod transform;
