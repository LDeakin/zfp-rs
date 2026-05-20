#![allow(clippy::cast_sign_loss)] // i32↔u32 and i64↔u64 for transform
//! Forward and inverse decorrelating transforms.
//!
//! Implements the lifting (wavelet-like) transform and the orthogonal transform
//! used by the ZFP codec.
//!
//! Reference: `zfp/src/template/encode.c` (`fwd_lift`),
//!            `zfp/src/template/decode.c` (`inv_lift`),
//!            `zfp/src/template/encode{1-4}.c` (`fwd_xform`),
//!            `zfp/src/template/decode{1-4}.c` (`inv_xform`).

// ---------------------------------------------------------------------------
// Lifting transform (i32)
// ---------------------------------------------------------------------------

/// Forward lifting step on `p[0]`, `p[s]`, `p[2*s]`, `p[3*s]`.
///
/// Non-orthogonal transform:
/// ```text
///        ( 4  4  4  4) (x)
/// 1/16 * ( 5  1 -1 -5) (y)
///        (-4  4  4 -4) (z)
///        (-2  6 -6  2) (w)
/// ```
///
/// Wrapping arithmetic is used to match C's implicit two's-complement behaviour.
#[allow(clippy::many_single_char_names)]
pub fn fwd_lift_i32(p: &mut [i32], s: usize) {
    let mut x = p[0];
    let mut y = p[s];
    let mut z = p[2 * s];
    let mut w = p[3 * s];

    x = x.wrapping_add(w);
    x >>= 1;
    w = w.wrapping_sub(x);
    z = z.wrapping_add(y);
    z >>= 1;
    y = y.wrapping_sub(z);
    x = x.wrapping_add(z);
    x >>= 1;
    z = z.wrapping_sub(x);
    w = w.wrapping_add(y);
    w >>= 1;
    y = y.wrapping_sub(w);
    w = w.wrapping_add(y >> 1);
    y = y.wrapping_sub(w >> 1);

    p[0] = x;
    p[s] = y;
    p[2 * s] = z;
    p[3 * s] = w;
}

/// Inverse lifting step on `p[0]`, `p[s]`, `p[2*s]`, `p[3*s]`.
///
/// Non-orthogonal transform:
/// ```text
///       ( 4  6 -4 -1) (x)
/// 1/4 * ( 4  2  4  5) (y)
///       ( 4 -2  4 -5) (z)
///       ( 4 -6 -4  1) (w)
/// ```
///
/// Wrapping arithmetic is required: these operations can overflow by design,
/// matching C's two's-complement implicit wrap behaviour.
#[allow(clippy::many_single_char_names)]
pub fn inv_lift_i32(p: &mut [i32], s: usize) {
    let mut x = p[0];
    let mut y = p[s];
    let mut z = p[2 * s];
    let mut w = p[3 * s];

    y = y.wrapping_add(w >> 1);
    w = w.wrapping_sub(y >> 1);
    y = y.wrapping_add(w);
    w = w.wrapping_sub(y.wrapping_sub(w));
    z = z.wrapping_add(x);
    x = x.wrapping_sub(z.wrapping_sub(x));
    y = y.wrapping_add(z);
    z = z.wrapping_sub(y.wrapping_sub(z));
    w = w.wrapping_add(x);
    x = x.wrapping_sub(w.wrapping_sub(x));

    p[0] = x;
    p[s] = y;
    p[2 * s] = z;
    p[3 * s] = w;
}

// ---------------------------------------------------------------------------
// Lifting transform (i64)
// ---------------------------------------------------------------------------

/// Forward lifting step on `p[0]`, `p[s]`, `p[2*s]`, `p[3*s]` (64-bit).
///
/// Wrapping arithmetic is used to match C's implicit two's-complement behaviour.
#[allow(clippy::many_single_char_names)]
pub fn fwd_lift_i64(p: &mut [i64], s: usize) {
    let mut x = p[0];
    let mut y = p[s];
    let mut z = p[2 * s];
    let mut w = p[3 * s];

    x = x.wrapping_add(w);
    x >>= 1;
    w = w.wrapping_sub(x);
    z = z.wrapping_add(y);
    z >>= 1;
    y = y.wrapping_sub(z);
    x = x.wrapping_add(z);
    x >>= 1;
    z = z.wrapping_sub(x);
    w = w.wrapping_add(y);
    w >>= 1;
    y = y.wrapping_sub(w);
    w = w.wrapping_add(y >> 1);
    y = y.wrapping_sub(w >> 1);

    p[0] = x;
    p[s] = y;
    p[2 * s] = z;
    p[3 * s] = w;
}

/// Inverse lifting step on `p[0]`, `p[s]`, `p[2*s]`, `p[3*s]` (64-bit).
///
/// Wrapping arithmetic is required: these operations can overflow by design,
/// matching C's two's-complement implicit wrap behaviour.
#[allow(clippy::many_single_char_names)]
pub fn inv_lift_i64(p: &mut [i64], s: usize) {
    let mut x = p[0];
    let mut y = p[s];
    let mut z = p[2 * s];
    let mut w = p[3 * s];

    y = y.wrapping_add(w >> 1);
    w = w.wrapping_sub(y >> 1);
    y = y.wrapping_add(w);
    w = w.wrapping_sub(y.wrapping_sub(w));
    z = z.wrapping_add(x);
    x = x.wrapping_sub(z.wrapping_sub(x));
    y = y.wrapping_add(z);
    z = z.wrapping_sub(y.wrapping_sub(z));
    w = w.wrapping_add(x);
    x = x.wrapping_sub(w.wrapping_sub(x));

    p[0] = x;
    p[s] = y;
    p[2 * s] = z;
    p[3 * s] = w;
}

// ---------------------------------------------------------------------------
// Orthogonal transforms (i32): composed from lifting steps
// ---------------------------------------------------------------------------

/// Forward orthogonal transform on a 4-element (4^1) block.
pub fn fwd_xform_1d(p: &mut [i32; 4]) {
    fwd_lift_i32(p, 1);
}

/// Inverse orthogonal transform on a 4-element (4^1) block.
pub fn inv_xform_1d(p: &mut [i32; 4]) {
    inv_lift_i32(p, 1);
}

/// Forward orthogonal transform on a 16-element (4×4) block.
pub fn fwd_xform_2d(p: &mut [i32; 16]) {
    // Transform along x (stride 1, 4 rows).
    for y in 0..4_usize {
        fwd_lift_i32(&mut p[4 * y..], 1);
    }
    // Transform along y (stride 4, 4 columns).
    for x in 0..4_usize {
        // Build a temporary flat slice view with stride 4.
        let mut col = [p[x], p[4 + x], p[8 + x], p[12 + x]];
        fwd_lift_i32(&mut col, 1);
        p[x] = col[0];
        p[4 + x] = col[1];
        p[8 + x] = col[2];
        p[12 + x] = col[3];
    }
}

/// Inverse orthogonal transform on a 16-element (4×4) block.
pub fn inv_xform_2d(p: &mut [i32; 16]) {
    // Transform along y first (inverse order of forward).
    for x in 0..4_usize {
        let mut col = [p[x], p[4 + x], p[8 + x], p[12 + x]];
        inv_lift_i32(&mut col, 1);
        p[x] = col[0];
        p[4 + x] = col[1];
        p[8 + x] = col[2];
        p[12 + x] = col[3];
    }
    // Transform along x.
    for y in 0..4_usize {
        inv_lift_i32(&mut p[4 * y..], 1);
    }
}

/// Forward orthogonal transform on a 64-element (4×4×4) block.
/// Block layout: `p[x + 4*y + 16*z]`.
pub fn fwd_xform_3d(p: &mut [i32; 64]) {
    // Transform along x (stride 1).
    for z in 0..4_usize {
        for y in 0..4_usize {
            fwd_lift_i32(&mut p[4 * y + 16 * z..], 1);
        }
    }
    // Transform along y (stride 4).
    for z in 0..4_usize {
        for x in 0..4_usize {
            let base = 16 * z + x;
            let mut col = [p[base], p[base + 4], p[base + 8], p[base + 12]];
            fwd_lift_i32(&mut col, 1);
            p[base] = col[0];
            p[base + 4] = col[1];
            p[base + 8] = col[2];
            p[base + 12] = col[3];
        }
    }
    // Transform along z (stride 16).
    for y in 0..4_usize {
        for x in 0..4_usize {
            let base = x + 4 * y;
            let mut col = [p[base], p[base + 16], p[base + 32], p[base + 48]];
            fwd_lift_i32(&mut col, 1);
            p[base] = col[0];
            p[base + 16] = col[1];
            p[base + 32] = col[2];
            p[base + 48] = col[3];
        }
    }
}

/// Inverse orthogonal transform on a 64-element (4×4×4) block.
pub fn inv_xform_3d(p: &mut [i32; 64]) {
    // Transform along z.
    for y in 0..4_usize {
        for x in 0..4_usize {
            let base = x + 4 * y;
            let mut col = [p[base], p[base + 16], p[base + 32], p[base + 48]];
            inv_lift_i32(&mut col, 1);
            p[base] = col[0];
            p[base + 16] = col[1];
            p[base + 32] = col[2];
            p[base + 48] = col[3];
        }
    }
    // Transform along y.
    for z in 0..4_usize {
        for x in 0..4_usize {
            let base = 16 * z + x;
            let mut col = [p[base], p[base + 4], p[base + 8], p[base + 12]];
            inv_lift_i32(&mut col, 1);
            p[base] = col[0];
            p[base + 4] = col[1];
            p[base + 8] = col[2];
            p[base + 12] = col[3];
        }
    }
    // Transform along x.
    for z in 0..4_usize {
        for y in 0..4_usize {
            inv_lift_i32(&mut p[4 * y + 16 * z..], 1);
        }
    }
}

/// Forward orthogonal transform on a 256-element (4×4×4×4) block.
/// Block layout: `p[x + 4*y + 16*z + 64*w]`.
pub fn fwd_xform_4d(p: &mut [i32; 256]) {
    // Transform along x (stride 1).
    for w in 0..4_usize {
        for z in 0..4_usize {
            for y in 0..4_usize {
                fwd_lift_i32(&mut p[4 * y + 16 * z + 64 * w..], 1);
            }
        }
    }
    // Transform along y (stride 4).
    for x in 0..4_usize {
        for w in 0..4_usize {
            for z in 0..4_usize {
                let base = 16 * z + 64 * w + x;
                let mut col = [p[base], p[base + 4], p[base + 8], p[base + 12]];
                fwd_lift_i32(&mut col, 1);
                p[base] = col[0];
                p[base + 4] = col[1];
                p[base + 8] = col[2];
                p[base + 12] = col[3];
            }
        }
    }
    // Transform along z (stride 16).
    for y in 0..4_usize {
        for x in 0..4_usize {
            for w in 0..4_usize {
                let base = 64 * w + x + 4 * y;
                let mut col = [p[base], p[base + 16], p[base + 32], p[base + 48]];
                fwd_lift_i32(&mut col, 1);
                p[base] = col[0];
                p[base + 16] = col[1];
                p[base + 32] = col[2];
                p[base + 48] = col[3];
            }
        }
    }
    // Transform along w (stride 64).
    for z in 0..4_usize {
        for y in 0..4_usize {
            for x in 0..4_usize {
                let base = x + 4 * y + 16 * z;
                let mut col = [p[base], p[base + 64], p[base + 128], p[base + 192]];
                fwd_lift_i32(&mut col, 1);
                p[base] = col[0];
                p[base + 64] = col[1];
                p[base + 128] = col[2];
                p[base + 192] = col[3];
            }
        }
    }
}

/// Inverse orthogonal transform on a 256-element (4×4×4×4) block.
pub fn inv_xform_4d(p: &mut [i32; 256]) {
    // Transform along w.
    for z in 0..4_usize {
        for y in 0..4_usize {
            for x in 0..4_usize {
                let base = x + 4 * y + 16 * z;
                let mut col = [p[base], p[base + 64], p[base + 128], p[base + 192]];
                inv_lift_i32(&mut col, 1);
                p[base] = col[0];
                p[base + 64] = col[1];
                p[base + 128] = col[2];
                p[base + 192] = col[3];
            }
        }
    }
    // Transform along z.
    for y in 0..4_usize {
        for x in 0..4_usize {
            for w in 0..4_usize {
                let base = 64 * w + x + 4 * y;
                let mut col = [p[base], p[base + 16], p[base + 32], p[base + 48]];
                inv_lift_i32(&mut col, 1);
                p[base] = col[0];
                p[base + 16] = col[1];
                p[base + 32] = col[2];
                p[base + 48] = col[3];
            }
        }
    }
    // Transform along y.
    for x in 0..4_usize {
        for w in 0..4_usize {
            for z in 0..4_usize {
                let base = 16 * z + 64 * w + x;
                let mut col = [p[base], p[base + 4], p[base + 8], p[base + 12]];
                inv_lift_i32(&mut col, 1);
                p[base] = col[0];
                p[base + 4] = col[1];
                p[base + 8] = col[2];
                p[base + 12] = col[3];
            }
        }
    }
    // Transform along x.
    for w in 0..4_usize {
        for z in 0..4_usize {
            for y in 0..4_usize {
                inv_lift_i32(&mut p[4 * y + 16 * z + 64 * w..], 1);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Reversible (Lorenzo) lifting transform (i32)
// ---------------------------------------------------------------------------

/// Reversible forward lifting step on `p[0]`, `p[s]`, `p[2*s]`, `p[3*s]` (i32).
///
/// Lorenzo difference transform (lossless):
/// ```text
/// y -= x; z -= y; w -= z; z -= y; w -= z; w -= z;
/// ```
/// Uses unsigned arithmetic to avoid signed-overflow UB.
#[allow(clippy::cast_possible_wrap, clippy::cast_sign_loss)] // i32↔u32 for transform
#[allow(clippy::many_single_char_names)]
pub fn rev_fwd_lift_i32(p: &mut [i32], s: usize) {
    let x = p[0] as u32;
    let mut y = p[s] as u32;
    let mut z = p[2 * s] as u32;
    let mut w = p[3 * s] as u32;

    w = w.wrapping_sub(z);
    z = z.wrapping_sub(y);
    y = y.wrapping_sub(x);
    w = w.wrapping_sub(z);
    z = z.wrapping_sub(y);
    w = w.wrapping_sub(z);

    p[0] = x as i32;
    p[s] = y as i32;
    p[2 * s] = z as i32;
    p[3 * s] = w as i32;
}

/// Reversible forward orthogonal transform on a 4-element block (i32).
#[allow(clippy::cast_possible_wrap, clippy::cast_sign_loss)] // i32↔u32 for transform
pub fn rev_fwd_xform_1d(p: &mut [i32; 4]) {
    rev_fwd_lift_i32(p, 1);
}

/// Reversible forward orthogonal transform on a 16-element (4×4) block (i32).
#[allow(clippy::cast_possible_wrap, clippy::cast_sign_loss)] // i32↔u32 for transform
pub fn rev_fwd_xform_2d(p: &mut [i32; 16]) {
    for y in 0..4_usize {
        rev_fwd_lift_i32(&mut p[4 * y..], 1);
    }
    for x in 0..4_usize {
        let mut col = [p[x], p[4 + x], p[8 + x], p[12 + x]];
        rev_fwd_lift_i32(&mut col, 1);
        p[x] = col[0];
        p[4 + x] = col[1];
        p[8 + x] = col[2];
        p[12 + x] = col[3];
    }
}

/// Reversible forward orthogonal transform on a 64-element (4×4×4) block (i32).
pub fn rev_fwd_xform_3d(p: &mut [i32; 64]) {
    for z in 0..4_usize {
        for y in 0..4_usize {
            rev_fwd_lift_i32(&mut p[4 * y + 16 * z..], 1);
        }
    }
    for z in 0..4_usize {
        for x in 0..4_usize {
            let base = 16 * z + x;
            let mut col = [p[base], p[base + 4], p[base + 8], p[base + 12]];
            rev_fwd_lift_i32(&mut col, 1);
            p[base] = col[0];
            p[base + 4] = col[1];
            p[base + 8] = col[2];
            p[base + 12] = col[3];
        }
    }
    for y in 0..4_usize {
        for x in 0..4_usize {
            let base = x + 4 * y;
            let mut col = [p[base], p[base + 16], p[base + 32], p[base + 48]];
            rev_fwd_lift_i32(&mut col, 1);
            p[base] = col[0];
            p[base + 16] = col[1];
            p[base + 32] = col[2];
            p[base + 48] = col[3];
        }
    }
}

/// Reversible forward orthogonal transform on a 256-element (4×4×4×4) block (i32).
pub fn rev_fwd_xform_4d(p: &mut [i32; 256]) {
    for w in 0..4_usize {
        for z in 0..4_usize {
            for y in 0..4_usize {
                rev_fwd_lift_i32(&mut p[4 * y + 16 * z + 64 * w..], 1);
            }
        }
    }
    for x in 0..4_usize {
        for w in 0..4_usize {
            for z in 0..4_usize {
                let base = 16 * z + 64 * w + x;
                let mut col = [p[base], p[base + 4], p[base + 8], p[base + 12]];
                rev_fwd_lift_i32(&mut col, 1);
                p[base] = col[0];
                p[base + 4] = col[1];
                p[base + 8] = col[2];
                p[base + 12] = col[3];
            }
        }
    }
    for y in 0..4_usize {
        for x in 0..4_usize {
            for w in 0..4_usize {
                let base = 64 * w + x + 4 * y;
                let mut col = [p[base], p[base + 16], p[base + 32], p[base + 48]];
                rev_fwd_lift_i32(&mut col, 1);
                p[base] = col[0];
                p[base + 16] = col[1];
                p[base + 32] = col[2];
                p[base + 48] = col[3];
            }
        }
    }
    for z in 0..4_usize {
        for y in 0..4_usize {
            for x in 0..4_usize {
                let base = x + 4 * y + 16 * z;
                let mut col = [p[base], p[base + 64], p[base + 128], p[base + 192]];
                rev_fwd_lift_i32(&mut col, 1);
                p[base] = col[0];
                p[base + 64] = col[1];
                p[base + 128] = col[2];
                p[base + 192] = col[3];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Reversible inverse (Lorenzo) lifting transform (i32)
// ---------------------------------------------------------------------------

/// Reversible inverse lifting step on `p[0]`, `p[s]`, `p[2*s]`, `p[3*s]` (i32).
///
/// Inverse Lorenzo (P4 Pascal matrix):
/// ```text
/// w += z; z += y; w += z; y += x; z += y; w += z;
/// ```
#[allow(clippy::many_single_char_names)]
#[allow(clippy::cast_possible_wrap, clippy::cast_sign_loss)] // i32↔u32 for transform
pub fn rev_inv_lift_i32(p: &mut [i32], s: usize) {
    let x = p[0] as u32;
    let mut y = p[s] as u32;
    let mut z = p[2 * s] as u32;
    let mut w = p[3 * s] as u32;

    w = w.wrapping_add(z);
    z = z.wrapping_add(y);
    w = w.wrapping_add(z);
    y = y.wrapping_add(x);
    z = z.wrapping_add(y);
    w = w.wrapping_add(z);

    p[0] = x as i32;
    p[s] = y as i32;
    p[2 * s] = z as i32;
    p[3 * s] = w as i32;
}

/// Reversible inverse orthogonal transform on a 4-element block (i32).
#[allow(clippy::cast_possible_wrap, clippy::cast_sign_loss)] // i32↔u32 for transform
pub fn rev_inv_xform_1d(p: &mut [i32; 4]) {
    rev_inv_lift_i32(p, 1);
}

/// Reversible inverse orthogonal transform on a 16-element (4×4) block (i32).
#[allow(clippy::cast_possible_wrap, clippy::cast_sign_loss)] // i32↔u32 for transform
pub fn rev_inv_xform_2d(p: &mut [i32; 16]) {
    // Inverse: y-dimension first, then x
    for x in 0..4_usize {
        let mut col = [p[x], p[4 + x], p[8 + x], p[12 + x]];
        rev_inv_lift_i32(&mut col, 1);
        p[x] = col[0];
        p[4 + x] = col[1];
        p[8 + x] = col[2];
        p[12 + x] = col[3];
    }
    for y in 0..4_usize {
        rev_inv_lift_i32(&mut p[4 * y..], 1);
    }
}

/// Reversible inverse orthogonal transform on a 64-element (4×4×4) block (i32).
pub fn rev_inv_xform_3d(p: &mut [i32; 64]) {
    for y in 0..4_usize {
        for x in 0..4_usize {
            let base = x + 4 * y;
            let mut col = [p[base], p[base + 16], p[base + 32], p[base + 48]];
            rev_inv_lift_i32(&mut col, 1);
            p[base] = col[0];
            p[base + 16] = col[1];
            p[base + 32] = col[2];
            p[base + 48] = col[3];
        }
    }
    for z in 0..4_usize {
        for x in 0..4_usize {
            let base = 16 * z + x;
            let mut col = [p[base], p[base + 4], p[base + 8], p[base + 12]];
            rev_inv_lift_i32(&mut col, 1);
            p[base] = col[0];
            p[base + 4] = col[1];
            p[base + 8] = col[2];
            p[base + 12] = col[3];
        }
    }
    for z in 0..4_usize {
        for y in 0..4_usize {
            rev_inv_lift_i32(&mut p[4 * y + 16 * z..], 1);
        }
    }
}

/// Reversible inverse orthogonal transform on a 256-element (4×4×4×4) block (i32).
pub fn rev_inv_xform_4d(p: &mut [i32; 256]) {
    for z in 0..4_usize {
        for y in 0..4_usize {
            for x in 0..4_usize {
                let base = x + 4 * y + 16 * z;
                let mut col = [p[base], p[base + 64], p[base + 128], p[base + 192]];
                rev_inv_lift_i32(&mut col, 1);
                p[base] = col[0];
                p[base + 64] = col[1];
                p[base + 128] = col[2];
                p[base + 192] = col[3];
            }
        }
    }
    for y in 0..4_usize {
        for x in 0..4_usize {
            for w in 0..4_usize {
                let base = 64 * w + x + 4 * y;
                let mut col = [p[base], p[base + 16], p[base + 32], p[base + 48]];
                rev_inv_lift_i32(&mut col, 1);
                p[base] = col[0];
                p[base + 16] = col[1];
                p[base + 32] = col[2];
                p[base + 48] = col[3];
            }
        }
    }
    for x in 0..4_usize {
        for w in 0..4_usize {
            for z in 0..4_usize {
                let base = 16 * z + 64 * w + x;
                let mut col = [p[base], p[base + 4], p[base + 8], p[base + 12]];
                rev_inv_lift_i32(&mut col, 1);
                p[base] = col[0];
                p[base + 4] = col[1];
                p[base + 8] = col[2];
                p[base + 12] = col[3];
            }
        }
    }
    for w in 0..4_usize {
        for z in 0..4_usize {
            for y in 0..4_usize {
                rev_inv_lift_i32(&mut p[4 * y + 16 * z + 64 * w..], 1);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Reversible (Lorenzo) lifting transform (i64)
// ---------------------------------------------------------------------------

/// Reversible forward lifting step on `p[0]`, `p[s]`, `p[2*s]`, `p[3*s]` (i64).
#[allow(clippy::cast_possible_wrap, clippy::cast_sign_loss)] // i64↔u64 for transform
#[allow(clippy::many_single_char_names)]
pub fn rev_fwd_lift_i64(p: &mut [i64], s: usize) {
    let x = p[0] as u64;
    let mut y = p[s] as u64;
    let mut z = p[2 * s] as u64;
    let mut w = p[3 * s] as u64;

    w = w.wrapping_sub(z);
    z = z.wrapping_sub(y);
    y = y.wrapping_sub(x);
    w = w.wrapping_sub(z);
    z = z.wrapping_sub(y);
    w = w.wrapping_sub(z);

    p[0] = x as i64;
    p[s] = y as i64;
    p[2 * s] = z as i64;
    p[3 * s] = w as i64;
}

/// Reversible forward orthogonal transform on a 4-element block (i64).
#[allow(clippy::cast_possible_wrap, clippy::cast_sign_loss)] // i64↔u64 for transform
pub fn rev_fwd_xform_1d_i64(p: &mut [i64; 4]) {
    rev_fwd_lift_i64(p, 1);
}

/// Reversible forward orthogonal transform on a 16-element block (i64).
#[allow(clippy::cast_possible_wrap, clippy::cast_sign_loss)] // i64↔u64 for transform
pub fn rev_fwd_xform_2d_i64(p: &mut [i64; 16]) {
    for y in 0..4_usize {
        rev_fwd_lift_i64(&mut p[4 * y..], 1);
    }
    for x in 0..4_usize {
        let mut col = [p[x], p[4 + x], p[8 + x], p[12 + x]];
        rev_fwd_lift_i64(&mut col, 1);
        p[x] = col[0];
        p[4 + x] = col[1];
        p[8 + x] = col[2];
        p[12 + x] = col[3];
    }
}

/// Reversible forward orthogonal transform on a 64-element block (i64).
pub fn rev_fwd_xform_3d_i64(p: &mut [i64; 64]) {
    for z in 0..4_usize {
        for y in 0..4_usize {
            rev_fwd_lift_i64(&mut p[4 * y + 16 * z..], 1);
        }
    }
    for z in 0..4_usize {
        for x in 0..4_usize {
            let base = 16 * z + x;
            let mut col = [p[base], p[base + 4], p[base + 8], p[base + 12]];
            rev_fwd_lift_i64(&mut col, 1);
            p[base] = col[0];
            p[base + 4] = col[1];
            p[base + 8] = col[2];
            p[base + 12] = col[3];
        }
    }
    for y in 0..4_usize {
        for x in 0..4_usize {
            let base = x + 4 * y;
            let mut col = [p[base], p[base + 16], p[base + 32], p[base + 48]];
            rev_fwd_lift_i64(&mut col, 1);
            p[base] = col[0];
            p[base + 16] = col[1];
            p[base + 32] = col[2];
            p[base + 48] = col[3];
        }
    }
}

/// Reversible forward orthogonal transform on a 256-element block (i64).
pub fn rev_fwd_xform_4d_i64(p: &mut [i64; 256]) {
    for w in 0..4_usize {
        for z in 0..4_usize {
            for y in 0..4_usize {
                rev_fwd_lift_i64(&mut p[4 * y + 16 * z + 64 * w..], 1);
            }
        }
    }
    for x in 0..4_usize {
        for w in 0..4_usize {
            for z in 0..4_usize {
                let base = 16 * z + 64 * w + x;
                let mut col = [p[base], p[base + 4], p[base + 8], p[base + 12]];
                rev_fwd_lift_i64(&mut col, 1);
                p[base] = col[0];
                p[base + 4] = col[1];
                p[base + 8] = col[2];
                p[base + 12] = col[3];
            }
        }
    }
    for y in 0..4_usize {
        for x in 0..4_usize {
            for w in 0..4_usize {
                let base = 64 * w + x + 4 * y;
                let mut col = [p[base], p[base + 16], p[base + 32], p[base + 48]];
                rev_fwd_lift_i64(&mut col, 1);
                p[base] = col[0];
                p[base + 16] = col[1];
                p[base + 32] = col[2];
                p[base + 48] = col[3];
            }
        }
    }
    for z in 0..4_usize {
        for y in 0..4_usize {
            for x in 0..4_usize {
                let base = x + 4 * y + 16 * z;
                let mut col = [p[base], p[base + 64], p[base + 128], p[base + 192]];
                rev_fwd_lift_i64(&mut col, 1);
                p[base] = col[0];
                p[base + 64] = col[1];
                p[base + 128] = col[2];
                p[base + 192] = col[3];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// i64 variants of xform (codec uses Int = int64 for int64 scalar type)
// ---------------------------------------------------------------------------

/// Forward orthogonal transform on a 4-element block (i64).
pub fn fwd_xform_1d_i64(p: &mut [i64; 4]) {
    fwd_lift_i64(p, 1);
}

/// Inverse orthogonal transform on a 4-element block (i64).
pub fn inv_xform_1d_i64(p: &mut [i64; 4]) {
    inv_lift_i64(p, 1);
}

/// Forward orthogonal transform on a 16-element block (i64).
pub fn fwd_xform_2d_i64(p: &mut [i64; 16]) {
    for y in 0..4_usize {
        fwd_lift_i64(&mut p[4 * y..], 1);
    }
    for x in 0..4_usize {
        let mut col = [p[x], p[4 + x], p[8 + x], p[12 + x]];
        fwd_lift_i64(&mut col, 1);
        p[x] = col[0];
        p[4 + x] = col[1];
        p[8 + x] = col[2];
        p[12 + x] = col[3];
    }
}

/// Inverse orthogonal transform on a 16-element block (i64).
pub fn inv_xform_2d_i64(p: &mut [i64; 16]) {
    for x in 0..4_usize {
        let mut col = [p[x], p[4 + x], p[8 + x], p[12 + x]];
        inv_lift_i64(&mut col, 1);
        p[x] = col[0];
        p[4 + x] = col[1];
        p[8 + x] = col[2];
        p[12 + x] = col[3];
    }
    for y in 0..4_usize {
        inv_lift_i64(&mut p[4 * y..], 1);
    }
}

/// Forward orthogonal transform on a 64-element block (i64).
pub fn fwd_xform_3d_i64(p: &mut [i64; 64]) {
    for z in 0..4_usize {
        for y in 0..4_usize {
            fwd_lift_i64(&mut p[4 * y + 16 * z..], 1);
        }
    }
    for z in 0..4_usize {
        for x in 0..4_usize {
            let base = 16 * z + x;
            let mut col = [p[base], p[base + 4], p[base + 8], p[base + 12]];
            fwd_lift_i64(&mut col, 1);
            p[base] = col[0];
            p[base + 4] = col[1];
            p[base + 8] = col[2];
            p[base + 12] = col[3];
        }
    }
    for y in 0..4_usize {
        for x in 0..4_usize {
            let base = x + 4 * y;
            let mut col = [p[base], p[base + 16], p[base + 32], p[base + 48]];
            fwd_lift_i64(&mut col, 1);
            p[base] = col[0];
            p[base + 16] = col[1];
            p[base + 32] = col[2];
            p[base + 48] = col[3];
        }
    }
}

/// Inverse orthogonal transform on a 64-element block (i64).
pub fn inv_xform_3d_i64(p: &mut [i64; 64]) {
    for y in 0..4_usize {
        for x in 0..4_usize {
            let base = x + 4 * y;
            let mut col = [p[base], p[base + 16], p[base + 32], p[base + 48]];
            inv_lift_i64(&mut col, 1);
            p[base] = col[0];
            p[base + 16] = col[1];
            p[base + 32] = col[2];
            p[base + 48] = col[3];
        }
    }
    for z in 0..4_usize {
        for x in 0..4_usize {
            let base = 16 * z + x;
            let mut col = [p[base], p[base + 4], p[base + 8], p[base + 12]];
            inv_lift_i64(&mut col, 1);
            p[base] = col[0];
            p[base + 4] = col[1];
            p[base + 8] = col[2];
            p[base + 12] = col[3];
        }
    }
    for z in 0..4_usize {
        for y in 0..4_usize {
            inv_lift_i64(&mut p[4 * y + 16 * z..], 1);
        }
    }
}

/// Forward orthogonal transform on a 256-element block (i64).
pub fn fwd_xform_4d_i64(p: &mut [i64; 256]) {
    for w in 0..4_usize {
        for z in 0..4_usize {
            for y in 0..4_usize {
                fwd_lift_i64(&mut p[4 * y + 16 * z + 64 * w..], 1);
            }
        }
    }
    for x in 0..4_usize {
        for w in 0..4_usize {
            for z in 0..4_usize {
                let base = 16 * z + 64 * w + x;
                let mut col = [p[base], p[base + 4], p[base + 8], p[base + 12]];
                fwd_lift_i64(&mut col, 1);
                p[base] = col[0];
                p[base + 4] = col[1];
                p[base + 8] = col[2];
                p[base + 12] = col[3];
            }
        }
    }
    for y in 0..4_usize {
        for x in 0..4_usize {
            for w in 0..4_usize {
                let base = 64 * w + x + 4 * y;
                let mut col = [p[base], p[base + 16], p[base + 32], p[base + 48]];
                fwd_lift_i64(&mut col, 1);
                p[base] = col[0];
                p[base + 16] = col[1];
                p[base + 32] = col[2];
                p[base + 48] = col[3];
            }
        }
    }
    for z in 0..4_usize {
        for y in 0..4_usize {
            for x in 0..4_usize {
                let base = x + 4 * y + 16 * z;
                let mut col = [p[base], p[base + 64], p[base + 128], p[base + 192]];
                fwd_lift_i64(&mut col, 1);
                p[base] = col[0];
                p[base + 64] = col[1];
                p[base + 128] = col[2];
                p[base + 192] = col[3];
            }
        }
    }
}

/// Inverse orthogonal transform on a 256-element block (i64).
pub fn inv_xform_4d_i64(p: &mut [i64; 256]) {
    for z in 0..4_usize {
        for y in 0..4_usize {
            for x in 0..4_usize {
                let base = x + 4 * y + 16 * z;
                let mut col = [p[base], p[base + 64], p[base + 128], p[base + 192]];
                inv_lift_i64(&mut col, 1);
                p[base] = col[0];
                p[base + 64] = col[1];
                p[base + 128] = col[2];
                p[base + 192] = col[3];
            }
        }
    }
    for y in 0..4_usize {
        for x in 0..4_usize {
            for w in 0..4_usize {
                let base = 64 * w + x + 4 * y;
                let mut col = [p[base], p[base + 16], p[base + 32], p[base + 48]];
                inv_lift_i64(&mut col, 1);
                p[base] = col[0];
                p[base + 16] = col[1];
                p[base + 32] = col[2];
                p[base + 48] = col[3];
            }
        }
    }
    for x in 0..4_usize {
        for w in 0..4_usize {
            for z in 0..4_usize {
                let base = 16 * z + 64 * w + x;
                let mut col = [p[base], p[base + 4], p[base + 8], p[base + 12]];
                inv_lift_i64(&mut col, 1);
                p[base] = col[0];
                p[base + 4] = col[1];
                p[base + 8] = col[2];
                p[base + 12] = col[3];
            }
        }
    }
    for w in 0..4_usize {
        for z in 0..4_usize {
            for y in 0..4_usize {
                inv_lift_i64(&mut p[4 * y + 16 * z + 64 * w..], 1);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Reversible inverse (Lorenzo) lifting transform (i64)
// ---------------------------------------------------------------------------

/// Reversible inverse lifting step on `p[0]`, `p[s]`, `p[2*s]`, `p[3*s]` (i64).
#[allow(clippy::cast_possible_wrap)] // u64→i64 for transform
#[allow(clippy::many_single_char_names)]
pub fn rev_inv_lift_i64(p: &mut [i64], s: usize) {
    let x = p[0] as u64;
    let mut y = p[s] as u64;
    let mut z = p[2 * s] as u64;
    let mut w = p[3 * s] as u64;

    w = w.wrapping_add(z);
    z = z.wrapping_add(y);
    w = w.wrapping_add(z);
    y = y.wrapping_add(x);
    z = z.wrapping_add(y);
    w = w.wrapping_add(z);

    p[0] = x as i64;
    p[s] = y as i64;
    p[2 * s] = z as i64;
    p[3 * s] = w as i64;
}

/// Reversible inverse orthogonal transform on a 4-element block (i64).
pub fn rev_inv_xform_1d_i64(p: &mut [i64; 4]) {
    rev_inv_lift_i64(p, 1);
}

/// Reversible inverse orthogonal transform on a 16-element block (i64).
pub fn rev_inv_xform_2d_i64(p: &mut [i64; 16]) {
    for x in 0..4_usize {
        let mut col = [p[x], p[4 + x], p[8 + x], p[12 + x]];
        rev_inv_lift_i64(&mut col, 1);
        p[x] = col[0];
        p[4 + x] = col[1];
        p[8 + x] = col[2];
        p[12 + x] = col[3];
    }
    for y in 0..4_usize {
        rev_inv_lift_i64(&mut p[4 * y..], 1);
    }
}

/// Reversible inverse orthogonal transform on a 64-element block (i64).
pub fn rev_inv_xform_3d_i64(p: &mut [i64; 64]) {
    for y in 0..4_usize {
        for x in 0..4_usize {
            let base = x + 4 * y;
            let mut col = [p[base], p[base + 16], p[base + 32], p[base + 48]];
            rev_inv_lift_i64(&mut col, 1);
            p[base] = col[0];
            p[base + 16] = col[1];
            p[base + 32] = col[2];
            p[base + 48] = col[3];
        }
    }
    for z in 0..4_usize {
        for x in 0..4_usize {
            let base = 16 * z + x;
            let mut col = [p[base], p[base + 4], p[base + 8], p[base + 12]];
            rev_inv_lift_i64(&mut col, 1);
            p[base] = col[0];
            p[base + 4] = col[1];
            p[base + 8] = col[2];
            p[base + 12] = col[3];
        }
    }
    for z in 0..4_usize {
        for y in 0..4_usize {
            rev_inv_lift_i64(&mut p[4 * y + 16 * z..], 1);
        }
    }
}

/// Reversible inverse orthogonal transform on a 256-element block (i64).
pub fn rev_inv_xform_4d_i64(p: &mut [i64; 256]) {
    for z in 0..4_usize {
        for y in 0..4_usize {
            for x in 0..4_usize {
                let base = x + 4 * y + 16 * z;
                let mut col = [p[base], p[base + 64], p[base + 128], p[base + 192]];
                rev_inv_lift_i64(&mut col, 1);
                p[base] = col[0];
                p[base + 64] = col[1];
                p[base + 128] = col[2];
                p[base + 192] = col[3];
            }
        }
    }
    for y in 0..4_usize {
        for x in 0..4_usize {
            for w in 0..4_usize {
                let base = 64 * w + x + 4 * y;
                let mut col = [p[base], p[base + 16], p[base + 32], p[base + 48]];
                rev_inv_lift_i64(&mut col, 1);
                p[base] = col[0];
                p[base + 16] = col[1];
                p[base + 32] = col[2];
                p[base + 48] = col[3];
            }
        }
    }
    for x in 0..4_usize {
        for w in 0..4_usize {
            for z in 0..4_usize {
                let base = 16 * z + 64 * w + x;
                let mut col = [p[base], p[base + 4], p[base + 8], p[base + 12]];
                rev_inv_lift_i64(&mut col, 1);
                p[base] = col[0];
                p[base + 4] = col[1];
                p[base + 8] = col[2];
                p[base + 12] = col[3];
            }
        }
    }
    for w in 0..4_usize {
        for z in 0..4_usize {
            for y in 0..4_usize {
                rev_inv_lift_i64(&mut p[4 * y + 16 * z + 64 * w..], 1);
            }
        }
    }
}
