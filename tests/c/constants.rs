#![allow(clippy::cast_possible_wrap)] // usize↔isize for stride computation
//! Test constants ported verbatim from `zfp/tests/constants/`.
//!
//! These are the hardcoded checksums and expected outputs used to verify
//! correct encode/decode behaviour. All items are `pub(super)` so the
//! sibling test modules can reference them.
#![allow(dead_code)] // Ported verbatim from upstream; not all constants are exercised by current tests.

// ---------------------------------------------------------------------------
// ZFP codec version (from zfp.h / zfp/include/zfp/version.h)
// ---------------------------------------------------------------------------

pub(super) const ZFP_CODEC: u32 = 5;

// ---------------------------------------------------------------------------
// Compression rate used in block encode/decode tests
// (ZFP_RATE_PARAM_BITS from the C test headers)
// ---------------------------------------------------------------------------

pub(super) const ZFP_RATE_PARAM_BITS: f64 = 8.0;

// ---------------------------------------------------------------------------
// 1-D Double constants  (zfp/tests/constants/1dDouble.h)
// ---------------------------------------------------------------------------

pub(super) mod dim1_double {
    pub(in super::super) const DIMS: u32 = 1;
    // The scalar block side length for 1-D is 4.
    pub(in super::super) const BLOCK_SIDE_LEN: usize = 4;
    pub(in super::super) const BLOCK_SIZE: usize = 4; // 4^1
}

// ---------------------------------------------------------------------------
// 1-D Float constants  (zfp/tests/constants/1dFloat.h)
// ---------------------------------------------------------------------------

pub(super) mod dim1_float {
    pub(in super::super) const DIMS: u32 = 1;
    pub(in super::super) const BLOCK_SIDE_LEN: usize = 4;
    pub(in super::super) const BLOCK_SIZE: usize = 4;
}

// ---------------------------------------------------------------------------
// 1-D Int32 constants
// ---------------------------------------------------------------------------

pub(super) mod dim1_int32 {
    pub(in super::super) const DIMS: u32 = 1;
    pub(in super::super) const BLOCK_SIDE_LEN: usize = 4;
    pub(in super::super) const BLOCK_SIZE: usize = 4;
}

// ---------------------------------------------------------------------------
// 1-D Int64 constants
// ---------------------------------------------------------------------------

pub(super) mod dim1_int64 {
    pub(in super::super) const DIMS: u32 = 1;
    pub(in super::super) const BLOCK_SIDE_LEN: usize = 4;
    pub(in super::super) const BLOCK_SIZE: usize = 4;
}

// ---------------------------------------------------------------------------
// 2-D Double constants  (zfp/tests/constants/2dDouble.h)
// ---------------------------------------------------------------------------

pub(super) mod dim2_double {
    pub(in super::super) const DIMS: u32 = 2;
    pub(in super::super) const BLOCK_SIDE_LEN: usize = 4;
    pub(in super::super) const BLOCK_SIZE: usize = 16; // 4^2
}

// ---------------------------------------------------------------------------
// 2-D Float constants
// ---------------------------------------------------------------------------

pub(super) mod dim2_float {
    pub(in super::super) const DIMS: u32 = 2;
    pub(in super::super) const BLOCK_SIDE_LEN: usize = 4;
    pub(in super::super) const BLOCK_SIZE: usize = 16;
}

// ---------------------------------------------------------------------------
// 2-D Int32/Int64 constants
// ---------------------------------------------------------------------------

pub(super) mod dim2_int32 {
    pub(in super::super) const DIMS: u32 = 2;
    pub(in super::super) const BLOCK_SIDE_LEN: usize = 4;
    pub(in super::super) const BLOCK_SIZE: usize = 16;
}

pub(super) mod dim2_int64 {
    pub(in super::super) const DIMS: u32 = 2;
    pub(in super::super) const BLOCK_SIDE_LEN: usize = 4;
    pub(in super::super) const BLOCK_SIZE: usize = 16;
}

// ---------------------------------------------------------------------------
// 3-D Double constants  (zfp/tests/constants/3dDouble.h)
// ---------------------------------------------------------------------------

pub(super) mod dim3_double {
    pub(in super::super) const DIMS: u32 = 3;
    pub(in super::super) const BLOCK_SIDE_LEN: usize = 4;
    pub(in super::super) const BLOCK_SIZE: usize = 64; // 4^3
}

pub(super) mod dim3_float {
    pub(in super::super) const DIMS: u32 = 3;
    pub(in super::super) const BLOCK_SIDE_LEN: usize = 4;
    pub(in super::super) const BLOCK_SIZE: usize = 64;
}

pub(super) mod dim3_int32 {
    pub(in super::super) const DIMS: u32 = 3;
    pub(in super::super) const BLOCK_SIDE_LEN: usize = 4;
    pub(in super::super) const BLOCK_SIZE: usize = 64;
}

pub(super) mod dim3_int64 {
    pub(in super::super) const DIMS: u32 = 3;
    pub(in super::super) const BLOCK_SIDE_LEN: usize = 4;
    pub(in super::super) const BLOCK_SIZE: usize = 64;
}

// ---------------------------------------------------------------------------
// 4-D Double constants  (zfp/tests/constants/4dDouble.h)
// ---------------------------------------------------------------------------

pub(super) mod dim4_double {
    pub(in super::super) const DIMS: u32 = 4;
    pub(in super::super) const BLOCK_SIDE_LEN: usize = 4;
    pub(in super::super) const BLOCK_SIZE: usize = 256; // 4^4
}

pub(super) mod dim4_float {
    pub(in super::super) const DIMS: u32 = 4;
    pub(in super::super) const BLOCK_SIDE_LEN: usize = 4;
    pub(in super::super) const BLOCK_SIZE: usize = 256;
}

pub(super) mod dim4_int32 {
    pub(in super::super) const DIMS: u32 = 4;
    pub(in super::super) const BLOCK_SIDE_LEN: usize = 4;
    pub(in super::super) const BLOCK_SIZE: usize = 256;
}

pub(super) mod dim4_int64 {
    pub(in super::super) const DIMS: u32 = 4;
    pub(in super::super) const BLOCK_SIDE_LEN: usize = 4;
    pub(in super::super) const BLOCK_SIZE: usize = 256;
}
