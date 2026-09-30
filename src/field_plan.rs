//! Per-field block iteration plan, shared by compression and decompression.

// The API and validation layer computes with caller-supplied sizes, so its
// arithmetic and indexing must be checked; see the crate's panic guarantee.
#![warn(clippy::indexing_slicing, clippy::arithmetic_side_effects)]

use std::ops::Range;

use crate::field::{checked_num_blocks, dimensionality, field_index_span, logical_shape_fits};
use crate::types::{ZfpDimensionality, ZfpFieldError, ZfpScalarType};

/// Block grid and memory layout, derived once per field.
pub(crate) struct FieldPlan {
    /// Number of blocks.
    pub num_blocks: usize,
    /// Block grid dimensions.
    pub bx: usize,
    pub by: usize,
    pub bz: usize,
    /// Lowest element index the strides reach, relative to index `[0, 0, 0, 0]`
    /// (`<= 0`). The logical origin sits `-imin` from the buffer's low end.
    ///
    /// Zero unless some stride is negative. See [`ZfpField::from_raw`].
    pub imin: isize,
    /// Effective strides.
    pub strides: [isize; 4],
    /// Field dimensions `[nx, ny, nz, nw]`.
    pub dims: [usize; 4],
    /// Dimensionality (1-4).
    pub dims_enum: ZfpDimensionality,
    /// Scalar type of the field data.
    pub scalar_type: ZfpScalarType,
}

impl FieldPlan {
    /// Validate a field's buffer and derive its block grid.
    ///
    /// `required` is the field's `checked_size_bytes`, saturated to `usize::MAX`
    /// when the span overflows.
    pub(crate) fn new(
        scalar_type: ZfpScalarType,
        dims: [usize; 4],
        dims_enum: ZfpDimensionality,
        strides: [isize; 4],
        data: &[u8],
        required: usize,
    ) -> Result<Self, ZfpFieldError> {
        debug_assert_eq!(
            dims_enum,
            dimensionality(&dims),
            "the block grid and the field's span must agree on which axes are active"
        );

        if !logical_shape_fits(&dims) {
            return Err(ZfpFieldError::ShapeTooLarge { dims });
        }

        let actual = data.len();
        if actual < required {
            return Err(ZfpFieldError::InsufficientData { required, actual });
        }

        // The codec reinterprets this buffer as the scalar type and walks it
        // with raw pointer offsets, so it must be correctly aligned. The
        // field constructors check this too, but the C ABI bypasses them.
        if !scalar_type.is_aligned(data.as_ptr()) {
            return Err(ZfpFieldError::MisalignedData {
                align: scalar_type.align(),
            });
        }

        let dim_count = usize::from(dims_enum);
        let [nx, ny, nz, nw] = dims;
        let bx = nx.div_ceil(4);
        let by = if dim_count >= 2 { ny.div_ceil(4) } else { 1 };
        let bz = if dim_count >= 3 { nz.div_ceil(4) } else { 1 };
        let bw = if dim_count >= 4 { nw.div_ceil(4) } else { 1 };

        let num_blocks = checked_num_blocks(&dims).ok_or(ZfpFieldError::ShapeTooLarge { dims })?;
        debug_assert_eq!(
            Some(num_blocks),
            bx.checked_mul(by)
                .and_then(|n| n.checked_mul(bz))
                .and_then(|n| n.checked_mul(bw))
        );

        Ok(Self {
            num_blocks,
            bx,
            by,
            bz,
            imin: field_index_span(&dims, &strides).0,
            strides,
            dims,
            dims_enum,
            scalar_type,
        })
    }

    /// Dimensionality as a count (1-4).
    #[inline]
    pub(crate) fn dim_count(&self) -> usize {
        usize::from(self.dims_enum)
    }

    /// Whether two distinct index tuples can address the same element.
    ///
    /// Sorts the active axes by `|stride|` and checks each one clears the span
    /// of the smaller ones, so `true` only means "cannot be ruled out".
    /// Parallel decompression requires this to be false.
    // Only the rayon path consults it; the unit tests below cover it either way.
    #[cfg_attr(not(feature = "rayon"), allow(dead_code))]
    pub(crate) fn strides_may_alias(&self) -> bool {
        // Axes of length 0 or 1 cannot alias. They sort last with a stride no
        // reach exceeds and an extent of nothing, so they never decide.
        let mut axes = [(usize::MAX, 1usize); 4];
        for ((axis, &stride), &dim) in axes
            .iter_mut()
            .zip(&self.strides)
            .zip(&self.dims)
            .take(self.dim_count())
        {
            if dim >= 2 {
                *axis = (stride.unsigned_abs(), dim);
            }
        }
        axes.sort_unstable();

        let mut reach = 1usize;
        for (stride, dim) in axes {
            if stride < reach {
                return true;
            }
            reach = stride
                .saturating_mul(dim.saturating_sub(1))
                .saturating_add(reach);
        }
        false
    }

    /// Block grid coordinates of a linear block index.
    ///
    /// Only for an index below `num_blocks`, so the grid is not empty.
    #[inline]
    #[expect(
        clippy::arithmetic_side_effects,
        reason = "the grid's dimensions are nonzero and multiply out to `num_blocks`, which fits"
    )]
    pub(crate) fn block_coords(&self, block_idx: usize) -> [usize; 4] {
        let rem = block_idx;
        let iw = rem / (self.bx * self.by * self.bz);
        let rem = rem % (self.bx * self.by * self.bz);
        let iz = rem / (self.bx * self.by);
        let rem = rem % (self.bx * self.by);
        let iy = rem / self.bx;
        let ix = rem % self.bx;
        [ix, iy, iz, iw]
    }

    /// Grid coordinates of blocks `range`, in index order.
    ///
    /// Divides once, for the first block, then steps like an odometer: a
    /// division per block is a measurable share of coding a 2-D block.
    #[expect(
        clippy::arithmetic_side_effects,
        reason = "each coordinate stays below its grid dimension, and the last below `num_blocks`"
    )]
    pub(crate) fn blocks(&self, range: Range<usize>) -> impl Iterator<Item = [usize; 4]> {
        // An empty grid has nothing to divide by.
        let mut next = if range.is_empty() {
            [0; 4]
        } else {
            self.block_coords(range.start)
        };
        let (bx, by, bz) = (self.bx, self.by, self.bz);
        range.map(move |_| {
            let coords = next;
            next[0] += 1;
            if next[0] == bx {
                next[0] = 0;
                next[1] += 1;
                if next[1] == by {
                    next[1] = 0;
                    next[2] += 1;
                    if next[2] == bz {
                        next[2] = 0;
                        next[3] += 1;
                    }
                }
            }
            coords
        })
    }

    /// Where block `[ix, iy, iz, iw]` starts, as an element offset from the
    /// buffer's low end, and how many of its values lie inside the field
    /// along each axis (zero past the dimensionality).
    #[inline]
    #[allow(clippy::cast_possible_wrap)] // block indices fit in isize for a valid field
    #[expect(
        clippy::arithmetic_side_effects,
        reason = "`FieldPlan::new` checked the field's index span fits `isize`, and a block's origin lies inside the field"
    )]
    pub(crate) fn block_geometry(&self, coords: [usize; 4]) -> (usize, [usize; 4]) {
        let mut offset = -self.imin;
        let mut lengths = [0; 4];
        for (((length, &coord), &stride), &dim) in lengths
            .iter_mut()
            .zip(&coords)
            .zip(&self.strides)
            .zip(&self.dims)
            .take(self.dim_count())
        {
            let origin = 4 * coord;
            offset += origin as isize * stride;
            *length = (dim - origin).min(4);
        }
        // `-imin` is where index zero sits, so every block starts at or above
        // the buffer's low end.
        (offset.cast_unsigned(), lengths)
    }

    /// Whether `lengths`, from [`Self::block_geometry`], is a whole block.
    #[inline]
    pub(crate) fn is_full(&self, lengths: [usize; 4]) -> bool {
        lengths.iter().take(self.dim_count()).all(|&n| n == 4)
    }
}

#[cfg(test)]
#[allow(clippy::arithmetic_side_effects, reason = "test spans are small")]
mod tests {
    use super::*;

    /// `imin` used to be computed inline in `compress`/`decompress` by summing
    /// the negative strides over the active axes. Pin the equivalence.
    #[allow(clippy::cast_possible_wrap)]
    fn imin_by_hand(dims: [usize; 4], strides: [isize; 4], dim_count: usize) -> isize {
        let mut lo: isize = 0;
        for (s, sz) in strides.iter().zip(dims.iter()).take(dim_count) {
            if *s < 0 {
                lo += s * (*sz as isize - 1);
            }
        }
        lo
    }

    #[test]
    fn field_index_span_matches_the_inline_imin() {
        let cases: [([usize; 4], [isize; 4], usize); 9] = [
            ([8, 0, 0, 0], [1, 0, 0, 0], 1),
            ([8, 0, 0, 0], [-1, 0, 0, 0], 1),
            ([8, 5, 0, 0], [1, 8, 0, 0], 2),
            ([8, 5, 0, 0], [-1, -8, 0, 0], 2),
            ([8, 5, 0, 0], [1, -8, 0, 0], 2),
            ([4, 3, 2, 0], [1, -4, 12, 0], 3),
            ([4, 3, 2, 2], [-1, 4, -12, 24], 4),
            ([4, 3, 2, 2], [-1, -4, -12, -24], 4),
            // Dims past `dim_count` must not shift the origin.
            ([5, 0, 5, 0], [1, 0, -100, 0], 1),
        ];
        for (dims, strides, dim_count) in cases {
            assert_eq!(
                crate::field::field_index_span(&dims, &strides).0,
                imin_by_hand(dims, strides, dim_count),
                "dims {dims:?} strides {strides:?}"
            );
        }
    }

    #[test]
    fn blocks_of_an_empty_grid_is_empty() {
        let plan = FieldPlan {
            num_blocks: 0,
            bx: 0,
            by: 1,
            bz: 1,
            imin: 0,
            strides: [1, 0, 0, 0],
            dims: [0; 4],
            dims_enum: ZfpDimensionality::D1,
            scalar_type: ZfpScalarType::F32,
        };
        assert_eq!(plan.blocks(0..0).count(), 0);
    }

    #[test]
    fn blocks_step_through_block_coords() {
        let data = vec![0u8; 4 * 9 * 5 * 6 * 7];
        let plan = FieldPlan::new(
            ZfpScalarType::F32,
            [9, 5, 6, 7],
            ZfpDimensionality::D4,
            [1, 9, 45, 270],
            &data,
            data.len(),
        )
        .unwrap();
        for range in [0..plan.num_blocks, 0..0, 3..4, 7..plan.num_blocks, 11..40] {
            let expect: Vec<_> = range.clone().map(|i| plan.block_coords(i)).collect();
            assert_eq!(plan.blocks(range).collect::<Vec<_>>(), expect);
        }
    }

    fn may_alias(dims: [usize; 4], strides: [isize; 4], dims_enum: ZfpDimensionality) -> bool {
        FieldPlan {
            num_blocks: 0,
            bx: 0,
            by: 0,
            bz: 0,
            imin: 0,
            strides,
            dims,
            dims_enum,
            scalar_type: ZfpScalarType::F32,
        }
        .strides_may_alias()
    }

    #[test]
    fn aliasing_strides_are_detected() {
        assert!(!may_alias(
            [8, 8, 0, 0],
            [1, 8, 0, 0],
            ZfpDimensionality::D2
        ));
        assert!(!may_alias(
            [8, 8, 0, 0],
            [-1, 8, 0, 0],
            ZfpDimensionality::D2
        ));
        assert!(!may_alias(
            [8, 8, 0, 0],
            [8, 1, 0, 0],
            ZfpDimensionality::D2
        ));
        assert!(!may_alias(
            [8, 0, 0, 0],
            [1, 0, 0, 0],
            ZfpDimensionality::D1
        ));
        // Both axes step by one element.
        assert!(may_alias([8, 8, 0, 0], [1, 1, 0, 0], ZfpDimensionality::D2));
        // A stride that lands inside the span of a smaller one.
        assert!(may_alias([8, 8, 0, 0], [1, 4, 0, 0], ZfpDimensionality::D2));
        // A zero stride repeats the same element.
        assert!(may_alias([8, 8, 0, 0], [1, 0, 0, 0], ZfpDimensionality::D2));
        // An axis outside `dim_count` does not count.
        assert!(!may_alias(
            [8, 8, 0, 0],
            [1, 1, 0, 0],
            ZfpDimensionality::D1
        ));
    }
}
