mod sealed_strides {
    pub trait Sealed {}
    impl Sealed for isize {}
    impl Sealed for [isize; 1] {}
    impl Sealed for [isize; 2] {}
    impl Sealed for [isize; 3] {}
    impl Sealed for [isize; 4] {}
}

/// Marker trait for stride arrays accepted by [`crate::ZfpField::new_strided`]
/// and [`crate::ZfpFieldMut::new_strided`].
///
/// Implemented for `isize`, `[isize; 1]` through `[isize; 4]`.
/// A stride of `0` means "use the default (contiguous) stride for that axis."
pub trait ZfpStrides: sealed_strides::Sealed + Copy {
    /// Return the strides as a fixed-size array of length 4, with trailing entries set to 0.
    fn to_array(self) -> [isize; 4];

    /// Return the dimensionality (1–4).
    fn dimensionality(&self) -> usize;
}

impl ZfpStrides for isize {
    fn to_array(self) -> [isize; 4] {
        [self, 0, 0, 0]
    }

    fn dimensionality(&self) -> usize {
        1
    }
}

impl ZfpStrides for [isize; 1] {
    fn to_array(self) -> [isize; 4] {
        [self[0], 0, 0, 0]
    }

    fn dimensionality(&self) -> usize {
        1
    }
}
impl ZfpStrides for [isize; 2] {
    fn to_array(self) -> [isize; 4] {
        [self[0], self[1], 0, 0]
    }

    fn dimensionality(&self) -> usize {
        2
    }
}
impl ZfpStrides for [isize; 3] {
    fn to_array(self) -> [isize; 4] {
        [self[0], self[1], self[2], 0]
    }

    fn dimensionality(&self) -> usize {
        3
    }
}
impl ZfpStrides for [isize; 4] {
    fn to_array(self) -> [isize; 4] {
        self
    }

    fn dimensionality(&self) -> usize {
        4
    }
}
