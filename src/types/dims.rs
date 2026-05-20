mod sealed {
    pub trait Sealed {}
    impl Sealed for usize {}
    impl Sealed for [usize; 1] {}
    impl Sealed for [usize; 2] {}
    impl Sealed for [usize; 3] {}
    impl Sealed for [usize; 4] {}
}

/// Marker trait for dimension arrays accepted by [`crate::ZfpField::new`] and
/// [`crate::ZfpFieldMut::new`].
///
/// Implemented for `usize`, `[usize; 1]` through `[usize; 4]`.
pub trait ZfpDims: sealed::Sealed + Copy {
    /// Return the dimensions as a fixed-size array of length 4, with trailing entries set to 0.
    fn to_array(self) -> [usize; 4];

    /// Return the dimensionality (1–4).
    fn dimensionality(&self) -> usize;
}

impl ZfpDims for usize {
    fn to_array(self) -> [usize; 4] {
        [self, 0, 0, 0]
    }

    fn dimensionality(&self) -> usize {
        1
    }
}

impl ZfpDims for [usize; 1] {
    fn to_array(self) -> [usize; 4] {
        [self[0], 0, 0, 0]
    }

    fn dimensionality(&self) -> usize {
        1
    }
}
impl ZfpDims for [usize; 2] {
    fn to_array(self) -> [usize; 4] {
        [self[0], self[1], 0, 0]
    }

    fn dimensionality(&self) -> usize {
        2
    }
}
impl ZfpDims for [usize; 3] {
    fn to_array(self) -> [usize; 4] {
        [self[0], self[1], self[2], 0]
    }

    fn dimensionality(&self) -> usize {
        3
    }
}
impl ZfpDims for [usize; 4] {
    fn to_array(self) -> [usize; 4] {
        self
    }

    fn dimensionality(&self) -> usize {
        4
    }
}
