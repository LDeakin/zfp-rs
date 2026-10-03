//! Block decoding in two stages: [`read_batch`] parses headers and bit planes
//! from the stream, and [`reconstruct_batch`] turns what it read into values
//! in the field. The stages can run on different threads, which the serial
//! decoders, fusing both, cannot.
//!
//! The block grammar is the one in `core.rs` and `reversible.rs`: a change to
//! it there needs the same change in [`read_block`].

use crate::bitstream::ZfpBitStreamRef;
use crate::codec::bitplane::{DecodedPlanes, PlaneBlock};
use crate::codec::decode::reversible::{
    NO_ROUNDING, rev_inv_reinterpret_f32, rev_inv_reinterpret_f64,
};
use crate::codec::decode::{
    core::{inv_cast_f32, inv_cast_f64, inv_order_i32, inv_order_i64},
    dim2, dim3, dim4,
};
use crate::codec::encode::core::{
    Budget, EBIAS_F32, EBIAS_F64, EBITS_F32, EBITS_F64, PBITS_32, PBITS_64, PERM_2, PERM_3, PERM_4,
    precision_f, skip_to,
};
use crate::codec::transform::{inv_xform, rev_inv_xform};
use crate::config::ZfpConfig;
use crate::field_plan::FieldPlan;
use crate::types::{ZfpDimensionality, ZfpScalar};

/// Values needed after reading a floating-point block's header.
#[derive(Clone, Copy)]
pub(crate) enum FloatEncoding {
    Zero,
    BlockFloat(i32),
    Reinterpret,
}

/// What [`read_block`] leaves for [`Scalar::reconstruct`].
pub(crate) struct Block<T: Scalar<N>, const N: usize> {
    planes: DecodedPlanes<T::Unsigned>,
    header: T::Header,
}

impl<T: Scalar<N>, const N: usize> Block<T, N> {
    pub(crate) fn new() -> Self {
        Self {
            planes: DecodedPlanes::new(),
            header: FloatEncoding::Zero.into(),
        }
    }
}

/// A scalar type's blocks of `N` values, decoded in stages.
pub(crate) trait Scalar<const N: usize>: ZfpScalar {
    type Unsigned: PlaneBlock;
    /// What a block's header leaves for [`Self::reconstruct`]: nothing for an
    /// integer, which has no header.
    type Header: Copy + Send + Sync + From<FloatEncoding>;
    const FLOAT: bool;
    const EBITS: u32;
    const EBIAS: i32;
    const PBITS: u32;
    fn reconstruct(
        planes: &DecodedPlanes<Self::Unsigned>,
        header: Self::Header,
        reversible: bool,
        perm: &[u8; N],
    ) -> [Self; N];
}

impl From<FloatEncoding> for () {
    fn from(_: FloatEncoding) {}
}

macro_rules! integer {
    ($int:ty, $uint:ty, $pbits:expr, $order:ident) => {
        impl<const N: usize> Scalar<N> for $int
        where
            [$uint; N]: PlaneBlock,
        {
            type Unsigned = [$uint; N];
            type Header = ();
            const FLOAT: bool = false;
            const EBITS: u32 = 0;
            const EBIAS: i32 = 0;
            const PBITS: u32 = $pbits;
            #[inline]
            fn reconstruct(
                planes: &DecodedPlanes<Self::Unsigned>,
                (): (),
                reversible: bool,
                perm: &[u8; N],
            ) -> [Self; N] {
                let (unsigned, zero) = planes.reconstruct();
                let mut out = [0; N];
                if !zero {
                    $order(&unsigned, &mut out, perm);
                    if reversible {
                        rev_inv_xform(&mut out);
                    } else {
                        inv_xform(&mut out);
                    }
                }
                out
            }
        }
    };
}
integer!(i32, u32, PBITS_32, inv_order_i32);
integer!(i64, u64, PBITS_64, inv_order_i64);

macro_rules! float {
    ($float:ty, $int:ty, $ebits:ident, $ebias:ident, $cast:ident, $reinterpret:ident) => {
        impl<const N: usize> Scalar<N> for $float
        where
            $int: Scalar<N, Header = ()>,
        {
            type Unsigned = <$int as Scalar<N>>::Unsigned;
            type Header = FloatEncoding;
            const FLOAT: bool = true;
            const EBITS: u32 = $ebits;
            const EBIAS: i32 = $ebias;
            const PBITS: u32 = <$int as Scalar<N>>::PBITS;
            #[inline]
            fn reconstruct(
                planes: &DecodedPlanes<Self::Unsigned>,
                header: FloatEncoding,
                reversible: bool,
                perm: &[u8; N],
            ) -> [Self; N] {
                let mut out = [0.0; N];
                match header {
                    FloatEncoding::Zero => {}
                    FloatEncoding::BlockFloat(exponent) => {
                        let ints = <$int as Scalar<N>>::reconstruct(planes, (), reversible, perm);
                        $cast(&ints, &mut out, exponent);
                    }
                    FloatEncoding::Reinterpret => {
                        let ints = <$int as Scalar<N>>::reconstruct(planes, (), reversible, perm);
                        $reinterpret(&ints, &mut out);
                    }
                }
                out
            }
        }
    };
}
float!(
    f32,
    i32,
    EBITS_F32,
    EBIAS_F32,
    inv_cast_f32,
    rev_inv_reinterpret_f32
);
float!(
    f64,
    i64,
    EBITS_F64,
    EBIAS_F64,
    inv_cast_f64,
    rev_inv_reinterpret_f64
);

/// The coefficient order and scatter of blocks of `N` values.
pub(crate) struct Layout<const N: usize>;
pub(crate) trait Scatter<const N: usize> {
    const PERM: [u8; N];
    /// Safety: `data` and the layout must cover every output address.
    unsafe fn scatter<T: Copy>(block: &[T; N], data: *mut T, info: &FieldPlan, lengths: [usize; 4]);
}
macro_rules! layout {
    ($n:literal, $perm:ident, $dim:ident, $full:ident, $partial:ident, [$($s:ident),+], [$($l:ident),+]) => {
        impl Scatter<$n> for Layout<$n> {
            const PERM: [u8; $n] = $perm;
            #[inline]
            unsafe fn scatter<T: Copy>(block: &[T; $n], data: *mut T, info: &FieldPlan, lengths: [usize; 4]) {
                let [$($s,)+ ..] = info.strides;
                let [$($l,)+ ..] = lengths;
                // SAFETY: the caller supplies a validated, disjoint block.
                unsafe {
                    if info.is_full(lengths) { $dim::$full(block, data, $($s),+); }
                    else { $dim::$partial(block, data, $($l,)+ $($s),+); }
                }
            }
        }
    };
}
layout!(
    16,
    PERM_2,
    dim2,
    scatter_2d,
    scatter_partial_2d,
    [sx, sy],
    [lx, ly]
);
layout!(
    64,
    PERM_3,
    dim3,
    scatter_3d,
    scatter_partial_3d,
    [sx, sy, sz],
    [lx, ly, lz]
);
layout!(
    256,
    PERM_4,
    dim4,
    scatter_4d,
    scatter_partial_4d,
    [sx, sy, sz, sw],
    [lx, ly, lz, lw]
);

/// Read headers and planes, retaining everything reconstruction needs.
#[inline]
fn read_block<T: Scalar<N>, const N: usize>(
    bs: &mut ZfpBitStreamRef<'_>,
    block: &mut Block<T, N>,
    config: &ZfpConfig,
    dims: ZfpDimensionality,
) {
    let reversible = config.is_reversible();
    let mut bits = u32::from(T::FLOAT);
    let mut precision = config.max_prec();
    if T::FLOAT {
        if bs.read_bits(1) == 0 {
            block.header = FloatEncoding::Zero.into();
            skip_to(bs, bits, config.min_bits());
            return;
        }
        let reinterpret = reversible && bs.read_bits(1) != 0;
        bits += u32::from(reversible);
        if reinterpret {
            block.header = FloatEncoding::Reinterpret.into();
        } else {
            #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
            let exponent = bs.read_bits(T::EBITS) as i32 - T::EBIAS;
            bits += T::EBITS;
            block.header = FloatEncoding::BlockFloat(exponent).into();
            if !reversible {
                precision = precision_f(
                    exponent,
                    precision,
                    config.min_exp(),
                    u32::from(dims),
                    config.rounding().tight_error(),
                );
            }
        }
    }
    if reversible {
        #[allow(clippy::cast_possible_truncation)]
        {
            precision = bs.read_bits(T::PBITS) as u32 + 1;
        }
        bits += T::PBITS;
    }
    let rounding = if reversible {
        NO_ROUNDING
    } else {
        config.rounding()
    };
    bits += block
        .planes
        .read(bs, Budget::of(config).after(bits).max, precision, rounding);
    skip_to(bs, bits, config.min_bits());
}

/// Read one block into each element of `batch`.
pub(crate) fn read_batch<T: Scalar<N>, const N: usize>(
    bs: &mut ZfpBitStreamRef<'_>,
    batch: &mut [Block<T, N>],
    config: &ZfpConfig,
    dims: ZfpDimensionality,
) {
    for block in batch {
        read_block::<T, N>(bs, block, config, dims);
    }
}

/// Reconstruct `batch`, which holds the field's blocks from index `start`,
/// using the block-coordinate odometer.
///
/// # Safety
/// `base` must point to the buffer `info` was planned for, and no other thread
/// may access the blocks `start..start + batch.len()`, which must not overlap
/// one another, so `info`'s strides must not alias.
pub(crate) unsafe fn reconstruct_batch<T: Scalar<N>, const N: usize>(
    batch: &[Block<T, N>],
    start: usize,
    base: *mut u8,
    info: &FieldPlan,
    config: &ZfpConfig,
) where
    Layout<N>: Scatter<N>,
{
    let reversible = config.is_reversible();
    for (block, coords) in batch.iter().zip(info.blocks(start..start + batch.len())) {
        let values = T::reconstruct(&block.planes, block.header, reversible, &Layout::<N>::PERM);
        let (offset, lengths) = info.block_geometry(coords);
        // SAFETY: the caller's contract.
        unsafe {
            Layout::<N>::scatter(&values, base.cast::<T>().add(offset), info, lengths);
        }
    }
}
