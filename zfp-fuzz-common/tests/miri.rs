//! Miri regression tests for the strided block gather/scatter paths.
//!
//! Before the raw-pointer refactor, `src/compress.rs` handed the codec a slice
//! manufactured with `slice::from_raw_parts(data_ptr, lx*ly*lz*lw*elem_size)`.
//! `ly`/`lz`/`lw` are 0 for unused axes, so for every 1-D, 2-D and 3-D field
//! that slice was *empty*, and each `*p.offset(x*sx + ...)` read or wrote
//! outside its provenance. Miri (Stacked Borrows) rejects it:
//!
//! ```text
//! error: Undefined Behavior: attempting a read access using <tag> at
//! alloc[0x0], but that tag does not exist in the borrow stack
//! ```
//!
//! In 4-D the same expression has the opposite problem: every length is
//! non-zero, so `byte_span` reaches past the end of the buffer for boundary
//! blocks of a field whose dimensions are not multiples of 4. The `4d_partial`
//! case below pins that.
//!
//! These cases are deliberately tiny. Miri is roughly three orders of magnitude
//! slower than native, so the C-port checksum suites are not viable here; this
//! file lives in `zfp-fuzz-common` rather than `zfp-rs/tests/` because
//! `zfp-rs`'s dev-dependencies (`zfp-sys`, `zfp-test-utils`) would drag cmake,
//! bindgen and a C toolchain into the run for no benefit.
//!
//! It is *not* `#[cfg(miri)]`-gated: it also runs natively in
//! `cargo test --workspace`, where it costs milliseconds and keeps itself
//! honest.

use zfp_rs::types::ZfpScalarType;
use zfp_rs::{
    ZfpBitStream, ZfpConfig, ZfpField, ZfpFieldMut, ZfpMode, ZfpScalar, ZfpStreamAlignment,
};

#[derive(Clone, Copy, Debug)]
struct Case {
    name: &'static str,
    /// `[nx, ny, nz, nw]`; 0 marks an unused axis.
    dims: [usize; 4],
    /// `[sx, sy, sz, sw]`; 0 means "use the contiguous default".
    strides: [isize; 4],
}

/// Every case is at most a handful of blocks. Negative and permuted strides are
/// over-represented on purpose: those are the offsets a slice-typed parameter
/// can never carry provenance for.
const CASES: &[Case] = &[
    // 1-D
    Case {
        name: "1d_contiguous",
        dims: [8, 0, 0, 0],
        strides: [0, 0, 0, 0],
    },
    Case {
        name: "1d_gap_partial",
        dims: [5, 0, 0, 0],
        strides: [2, 0, 0, 0],
    },
    Case {
        name: "1d_negative",
        dims: [5, 0, 0, 0],
        strides: [-1, 0, 0, 0],
    },
    // 2-D — `2d_gap` is the shape that first reproduced the UB.
    Case {
        name: "2d_gap",
        dims: [4, 4, 0, 0],
        strides: [1, 8, 0, 0],
    },
    Case {
        name: "2d_partial",
        dims: [5, 3, 0, 0],
        strides: [1, 5, 0, 0],
    },
    Case {
        name: "2d_negative_y",
        dims: [4, 4, 0, 0],
        strides: [1, -4, 0, 0],
    },
    Case {
        name: "2d_transposed",
        dims: [4, 4, 0, 0],
        strides: [4, 1, 0, 0],
    },
    // 3-D
    Case {
        name: "3d_contiguous",
        dims: [4, 4, 4, 0],
        strides: [0, 0, 0, 0],
    },
    Case {
        name: "3d_gap_partial",
        dims: [5, 2, 2, 0],
        strides: [2, 10, 20, 0],
    },
    Case {
        name: "3d_negative_x",
        dims: [4, 4, 2, 0],
        strides: [-1, 4, 16, 0],
    },
    // 4-D — `4d_partial` pins the buffer overrun described above.
    Case {
        name: "4d_full",
        dims: [4, 4, 4, 4],
        strides: [0, 0, 0, 0],
    },
    Case {
        name: "4d_partial",
        dims: [5, 2, 2, 2],
        strides: [0, 0, 0, 0],
    },
    Case {
        name: "4d_negative_w",
        dims: [4, 2, 2, 2],
        strides: [1, 4, 8, -16],
    },
];

/// Number of scalars the field's dims and strides span, gaps included.
fn span_of(case: &Case) -> usize {
    let (imin, imax) = ZfpField::field_index_span_static(&case.dims, &case.strides);
    usize::try_from(imax - imin + 1).expect("span fits in usize")
}

fn rank_of(case: &Case) -> usize {
    case.dims.iter().take_while(|&&d| d != 0).count()
}

/// Indices into the span buffer that the field actually addresses.
///
/// Non-unit strides leave gaps, and decompression never writes them, so the
/// reversible oracle below must skip them. Index 0 of the buffer is the
/// *lowest* address of the span, which is what the codec assumes, hence the
/// `-imin` shift.
fn covered_indices(case: &Case) -> Vec<usize> {
    let (imin, _) = ZfpField::field_index_span_static(&case.dims, &case.strides);
    let [nx, ny, nz, nw] = case.dims;
    let s = [
        if case.strides[0] != 0 {
            case.strides[0]
        } else {
            1
        },
        if case.strides[1] != 0 {
            case.strides[1]
        } else {
            nx as isize
        },
        if case.strides[2] != 0 {
            case.strides[2]
        } else {
            (nx * ny) as isize
        },
        if case.strides[3] != 0 {
            case.strides[3]
        } else {
            (nx * ny * nz) as isize
        },
    ];
    let mut out = Vec::new();
    for w in 0..nw.max(1) as isize {
        for z in 0..nz.max(1) as isize {
            for y in 0..ny.max(1) as isize {
                for x in 0..nx.max(1) as isize {
                    let off = -imin + x * s[0] + y * s[1] + z * s[2] + w * s[3];
                    out.push(usize::try_from(off).expect("offset is within the span"));
                }
            }
        }
    }
    out
}

/// A distinct, exactly-representable value per index, so a scatter landing on
/// the wrong index is visible even when the access itself is in bounds.
trait Sample: ZfpScalar + PartialEq + std::fmt::Debug {
    fn sample(i: usize) -> Self;
}

impl Sample for f64 {
    fn sample(i: usize) -> Self {
        // Small integers are exact in f64 and survive every mode losslessly
        // enough for the reversible oracle below.
        (i as f64) - 8.0
    }
}

impl Sample for i32 {
    fn sample(i: usize) -> Self {
        (i as i32) - 8
    }
}

fn roundtrip<T: Sample>(case: &Case, config: &ZfpConfig) {
    let span = span_of(case);
    let rank = rank_of(case);
    let src: Vec<T> = (0..span).map(T::sample).collect();

    let cap = config
        .maximum_size(T::scalar_type(), &case.dims[..rank])
        .max(64);
    let mut bs = ZfpBitStream::new(cap);

    let written = {
        let field = ZfpField::new_strided(&src, case.dims, case.strides);
        bs.compress(config, &field)
            .unwrap_or_else(|e| panic!("{}: compress failed: {e}", case.name))
    };
    assert!(
        written <= cap,
        "{}: compress wrote {written} B into a {cap} B stream",
        case.name
    );

    let mut dst: Vec<T> = vec![T::default(); span];
    bs.rewind();
    {
        let mut out = ZfpFieldMut::new_strided(&mut dst, case.dims, case.strides);
        bs.decompress(config, &mut out)
            .unwrap_or_else(|e| panic!("{}: decompress failed: {e}", case.name));
    }

    // Reversible mode is bit-exact, so it doubles as a correctness oracle:
    // it catches a gather/scatter that stays in bounds but hits the wrong index.
    if config.compression_mode() == ZfpMode::Reversible {
        for i in covered_indices(case) {
            assert_eq!(
                src[i], dst[i],
                "{}: reversible mode differs at index {i}",
                case.name
            );
        }
    }
}

/// `gather_block` / `scatter_block`, via reversible mode.
#[test]
fn strided_reversible() {
    for case in CASES {
        roundtrip::<f64>(case, &ZfpConfig::reversible());
        roundtrip::<i32>(case, &ZfpConfig::reversible());
    }
}

/// `gather_{1,2,3,4}d` / `scatter_*` and their partial-block variants, via a
/// lossy mode. Partial blocks are reached by the cases whose dimensions are not
/// multiples of 4.
#[test]
fn strided_fixed_rate() {
    for case in CASES {
        let rank = rank_of(case);
        let dims = zfp_fuzz_common::input::dimensionality_of(rank);
        roundtrip::<f64>(
            case,
            &ZfpConfig::fixed_rate(16.0, ZfpScalarType::Double, dims, ZfpStreamAlignment::None),
        );
        roundtrip::<i32>(
            case,
            &ZfpConfig::fixed_rate(16.0, ZfpScalarType::Int32, dims, ZfpStreamAlignment::None),
        );
    }
}

/// Fixed-accuracy exercises a third parameter path through the same helpers.
#[test]
fn strided_fixed_accuracy() {
    for case in CASES {
        roundtrip::<f64>(case, &ZfpConfig::fixed_accuracy(1e-3));
    }
}

/// A field whose buffer is shorter than its index span must be rejected before
/// any pointer arithmetic happens — this is the check the whole raw-pointer
/// scheme rests on.
#[test]
fn undersized_field_is_rejected() {
    let src = vec![0.0f64; 4];
    let field = ZfpField::new(&src, [1000usize]);
    let config = ZfpConfig::reversible();
    let mut bs = ZfpBitStream::new(4096);
    assert!(
        bs.compress(&config, &field).is_err(),
        "compressing a 4-element buffer declared as 1000 elements must fail"
    );

    let mut dst = vec![0.0f64; 4];
    let mut out = ZfpFieldMut::new(&mut dst, [1000usize]);
    assert!(
        bs.decompress(&config, &mut out).is_err(),
        "decompressing into a 4-element buffer declared as 1000 elements must fail"
    );
}
