//! Per-field block iteration plan, shared by compression and decompression.

use crate::field::{ZfpField, dimensionality};
use crate::types::{ZfpDimensionality, ZfpScalarType};

/// Why a field cannot be walked block by block.
///
/// `compress`/`decompress` map this onto their own error enums.
pub(crate) enum PlanError {
    NoData,
    InvalidField { required: usize, actual: usize },
    MisalignedData { align: usize },
}

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
    ) -> Result<Self, PlanError> {
        debug_assert_eq!(
            dims_enum,
            dimensionality(&dims),
            "the block grid and the field's span must agree on which axes are active"
        );

        if data.is_empty() {
            return Err(PlanError::NoData);
        }

        let actual = data.len();
        if actual < required {
            return Err(PlanError::InvalidField { required, actual });
        }

        // The codec reinterprets this buffer as the scalar type and walks it
        // with raw pointer offsets, so it must be correctly aligned. Only
        // reachable via `from_raw`: `ZfpField::new` goes through
        // `bytemuck::cast_slice`, which is always aligned.
        if !scalar_type.is_aligned(data.as_ptr()) {
            return Err(PlanError::MisalignedData {
                align: scalar_type.align(),
            });
        }

        let dim_count = usize::from(dims_enum);
        let [nx, ny, nz, nw] = dims;
        let bx = nx.div_ceil(4);
        let by = if dim_count >= 2 { ny.div_ceil(4) } else { 1 };
        let bz = if dim_count >= 3 { nz.div_ceil(4) } else { 1 };
        let bw = if dim_count >= 4 { nw.div_ceil(4) } else { 1 };

        Ok(Self {
            num_blocks: bx * by * bz * bw,
            bx,
            by,
            bz,
            imin: ZfpField::field_index_span_static(&dims, &strides).0,
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

    /// Element size in bytes.
    #[inline]
    pub(crate) fn elem_size(&self) -> usize {
        self.scalar_type.size()
    }

    /// Whether two distinct index tuples can address the same element.
    ///
    /// Sorts the active axes by `|stride|` and checks each one clears the span
    /// of the smaller ones, so `true` only means "cannot be ruled out".
    /// Parallel decompression requires this to be false.
    // Only the rayon path consults it; the unit tests below cover it either way.
    #[cfg_attr(not(feature = "rayon"), allow(dead_code))]
    pub(crate) fn strides_may_alias(&self) -> bool {
        let mut axes = [(0usize, 0usize); 4];
        let mut count = 0;
        for (&stride, &dim) in self.strides.iter().zip(&self.dims).take(self.dim_count()) {
            if dim >= 2 {
                axes[count] = (stride.unsigned_abs(), dim);
                count += 1;
            }
        }
        let axes = &mut axes[..count];
        axes.sort_unstable();

        let mut reach = 1usize;
        for &(stride, dim) in &*axes {
            if stride < reach {
                return true;
            }
            reach = stride.saturating_mul(dim - 1).saturating_add(reach);
        }
        false
    }

    /// Block grid coordinates of a linear block index.
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

#[cfg(test)]
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
                ZfpField::field_index_span_static(&dims, &strides).0,
                imin_by_hand(dims, strides, dim_count),
                "dims {dims:?} strides {strides:?}"
            );
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
            scalar_type: ZfpScalarType::Float,
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
