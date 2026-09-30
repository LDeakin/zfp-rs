//! Float strategies shared by the proptest binaries.
//!
//! C's scale factor overflows for a block whose largest magnitude is below
//! 2^-98 (`f32`) or 2^-962 (`f64`), so every value casts to the minimum
//! integer. zfp-rs scales such blocks exactly, and its bytes differ. Nonzero
//! values are at least that large here, subnormals included, so no block is
//! below it.

use proptest::prelude::*;

pub(crate) fn comparable_f32() -> impl Strategy<Value = f32> {
    // 2^-98
    let min = f32::from_bits(29 << 23);
    any::<f32>().prop_filter("zero, NaN or at least 2^-98", move |f| {
        *f == 0.0 || f.is_nan() || f.abs() >= min
    })
}

pub(crate) fn comparable_f64() -> impl Strategy<Value = f64> {
    // 2^-962
    let min = f64::from_bits(61 << 52);
    any::<f64>().prop_filter("zero, NaN or at least 2^-962", move |f| {
        *f == 0.0 || f.is_nan() || f.abs() >= min
    })
}
