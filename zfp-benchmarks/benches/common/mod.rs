//! Code shared by the benchmark binaries.
//!
//! `zfp-sys` and `zfp-rs-ffi` both define the C symbols `zfp_*` and `stream_*`,
//! so no binary may link both: every `zfp_sys` call would bind to the `zfp-rs-ffi`
//! definitions and the "C" results would silently be the Rust ones.
//! Nothing here may name either crate, as that would link it into every binary.

use zfp_rs::{ZfpConfig, ZfpDimensionality, ZfpScalar, ZfpScalarType, ZfpStreamAlignment};

pub const RATE: f64 = 8.0;
pub const PRECISION: u32 = 16;
pub const ACCURACY: f64 = 0.003_906_25;
pub const STREAM_PAD_BYTES: usize = 8;

#[derive(Clone, Copy)]
pub enum ScalarKind {
    I32,
    I64,
    F32,
    F64,
}

impl ScalarKind {
    const fn label(self) -> &'static str {
        match self {
            Self::I32 => "i32",
            Self::I64 => "i64",
            Self::F32 => "f32",
            Self::F64 => "f64",
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
pub enum ModeKind {
    FixedRate,
    FixedPrecision,
    FixedAccuracy,
    Reversible,
}

impl ModeKind {
    const fn label(self) -> &'static str {
        match self {
            Self::FixedRate => "fixed_rate",
            Self::FixedPrecision => "fixed_precision",
            Self::FixedAccuracy => "fixed_accuracy",
            Self::Reversible => "reversible",
        }
    }
}

#[derive(Clone, Copy)]
pub struct Case {
    pub scalar: ScalarKind,
    pub dims: u32,
    pub mode: ModeKind,
}

impl Case {
    pub fn label(self) -> String {
        format!(
            "{}_d{}_{}",
            self.scalar.label(),
            self.dims,
            self.mode.label()
        )
    }

    fn dimensionality(self) -> ZfpDimensionality {
        ZfpDimensionality::try_from(self.dims).expect("benchmark dimensions are valid")
    }
}

pub const SCALARS: &[ScalarKind] = &[
    ScalarKind::I32,
    ScalarKind::I64,
    ScalarKind::F32,
    ScalarKind::F64,
];
pub const DIMS: &[u32] = &[2, 3, 4];
pub const MODES: &[ModeKind] = &[
    ModeKind::FixedRate,
    ModeKind::FixedPrecision,
    ModeKind::FixedAccuracy,
    ModeKind::Reversible,
];
pub const OMP_THREADS: &[u32] = &[2];

pub trait BenchScalar: ZfpScalar + bytemuck::Pod + Default + Copy + 'static {
    const RUST_TYPE: ZfpScalarType;

    fn sample(index: usize) -> Self;

    fn generate(dims: u32) -> (Vec<Self>, Vec<usize>) {
        let shape = shape_for_dims(dims);
        let elements = shape.iter().product();
        let data = (0..elements).map(Self::sample).collect();
        (data, shape)
    }
}

fn shape_for_dims(dims: u32) -> Vec<usize> {
    match dims {
        2 => vec![1_024, 1_024],
        3 => vec![128, 128, 64],
        4 => vec![64, 64, 16, 16],
        _ => panic!("unsupported dimensionality"),
    }
}

impl BenchScalar for i32 {
    const RUST_TYPE: ZfpScalarType = ZfpScalarType::I32;

    fn sample(index: usize) -> Self {
        let mixed = index.wrapping_mul(1_103_515_245).wrapping_add(12_345);
        ((mixed & 0x000f_ffff) as i32) - 524_288
    }
}

impl BenchScalar for i64 {
    const RUST_TYPE: ZfpScalarType = ZfpScalarType::I64;

    fn sample(index: usize) -> Self {
        let mixed = (index as i64)
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (mixed & 0x0000_ffff_ffff_ffff) - 140_737_488_355_328
    }
}

impl BenchScalar for f32 {
    const RUST_TYPE: ZfpScalarType = ZfpScalarType::F32;

    fn sample(index: usize) -> Self {
        let x = index as f64;
        ((x * 0.001).sin() * 32.0 + (x * 0.000_037).cos() * 4.0) as f32
    }
}

impl BenchScalar for f64 {
    const RUST_TYPE: ZfpScalarType = ZfpScalarType::F64;

    fn sample(index: usize) -> Self {
        let x = index as f64;
        (x * 0.001).sin() * 32.0 + (x * 0.000_037).cos() * 4.0
    }
}

pub fn rust_config<T: BenchScalar>(case: Case) -> ZfpConfig {
    match case.mode {
        ModeKind::FixedRate => ZfpConfig::fixed_rate(
            RATE,
            T::RUST_TYPE,
            case.dimensionality(),
            ZfpStreamAlignment::Unaligned,
        )
        .expect("the benchmark rate is valid"),
        ModeKind::FixedPrecision => ZfpConfig::fixed_precision(PRECISION),
        ModeKind::FixedAccuracy => ZfpConfig::fixed_accuracy(ACCURACY),
        ModeKind::Reversible => ZfpConfig::reversible(),
    }
}

/// Zero-pad 1-4 dimensions to the `[nx, ny, nz, nw]` form.
pub fn dims4(dims: &[usize]) -> [usize; 4] {
    assert!((1..=4).contains(&dims.len()), "unsupported dimensionality");
    let mut out = [0; 4];
    out[..dims.len()].copy_from_slice(dims);
    out
}

/// Call `$bench_case::<T>(criterion, case)` for every scalar type, dimensionality and mode.
///
/// A macro rather than a generic function, as each binary bounds `T` by its own
/// extension of [`BenchScalar`].
macro_rules! run_cases {
    ($criterion:expr, $bench_case:ident) => {
        for &scalar in common::SCALARS {
            for &dims in common::DIMS {
                for &mode in common::MODES {
                    let case = common::Case { scalar, dims, mode };
                    match scalar {
                        common::ScalarKind::I32 => $bench_case::<i32>($criterion, case),
                        common::ScalarKind::I64 => $bench_case::<i64>($criterion, case),
                        common::ScalarKind::F32 => $bench_case::<f32>($criterion, case),
                        common::ScalarKind::F64 => $bench_case::<f64>($criterion, case),
                    }
                }
            }
        }
    };
}
