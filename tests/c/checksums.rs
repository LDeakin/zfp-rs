#![allow(clippy::cast_possible_wrap)] // usize↔isize for stride computation
//! Port of `zfp/tests/utils/zfpChecksums.{h,c}` and `zfp/tests/utils/zfpHash.c`.
//!
//! Provides:
//! - Jenkins one-at-a-time hash functions for input arrays and bitstreams
//! - Key computation matching `computeKey` / `computeKeyOriginalInput`
//! - Lookup tables ported from all 16 `zfp/tests/constants/checksums/*.h` files
//! - `get_checksum` for looking up expected values

// ---------------------------------------------------------------------------
// Hash functions (zfpHash.c)
// ---------------------------------------------------------------------------

#[inline]
fn hash_value(val: u32, h: &mut u32) {
    *h = h.wrapping_add(val);
    *h = h.wrapping_add(*h << 10);
    *h ^= *h >> 6;
}

#[inline]
fn hash_finish(h: u32) -> u32 {
    let mut h = h;
    h = h.wrapping_add(h << 3);
    h ^= h >> 11;
    h = h.wrapping_add(h << 15);
    h
}

/// Hash a bitstream buffer (matches `hashBitstream`).
/// `buf` is the raw committed byte slice; treated as little-endian u64 words.
pub fn hash_bitstream(buf: &[u8]) -> u64 {
    let mut h1: u32 = 0;
    let mut h2: u32 = 0;
    for chunk in buf.as_chunks::<8>().0 {
        let word = u64::from_le_bytes(*chunk);
        let lo = (word & 0xffff_ffff) as u32;
        let hi = ((word >> 32) & 0xffff_ffff) as u32;
        hash_value(lo, &mut h1);
        hash_value(hi, &mut h2);
    }
    let r1 = u64::from(hash_finish(h1));
    let r2 = u64::from(hash_finish(h2));
    r1 + (r2 << 32)
}

/// Hash a 32-bit array with given stride (matches `hashArray32`).
/// `stride` is the element stride (not byte stride).
pub fn hash_array32(arr: &[u32], stride: usize) -> u32 {
    let mut h: u32 = 0;
    let mut idx = 0;
    let count = arr.len().checked_div(stride).unwrap_or(0);
    for _ in 0..count {
        hash_value(arr[idx], &mut h);
        idx += stride;
    }
    hash_finish(h)
}

/// Hash a 64-bit array with given stride (matches `hashArray64`).
/// `stride` is the element stride (not byte stride).
pub fn hash_array64(arr: &[u64], stride: usize) -> u64 {
    let mut h1: u32 = 0;
    let mut h2: u32 = 0;
    let mut idx = 0;
    let count = arr.len().checked_div(stride).unwrap_or(0);
    for _ in 0..count {
        let val = arr[idx];
        hash_value((val & 0xffff_ffff) as u32, &mut h1);
        hash_value(((val >> 32) & 0xffff_ffff) as u32, &mut h2);
        idx += stride;
    }
    let r1 = u64::from(hash_finish(h1));
    let r2 = u64::from(hash_finish(h2));
    r1 + (r2 << 32)
}

/// Hash a strided 32-bit array (matches `hashStridedArray32`).
/// `n` is the logical array dimensions (unused dims are 0).
/// `s` is the element stride per dimension.
pub fn hash_strided_array32(arr: &[u32], n: [usize; 4], s: [isize; 4]) -> u32 {
    let mut h: u32 = 0;
    let n0 = n[0].max(1);
    let n1 = if n[1] > 0 { n[1] } else { 1 };
    let n2 = if n[2] > 0 { n[2] } else { 1 };
    let n3 = if n[3] > 0 { n[3] } else { 1 };

    // Mirrors C: single `arr` pointer advancing through all loops.
    // ptr_j/k/l track the start-of-row/plane/hyperplane positions.
    // After each inner loop, ptr advances by n_inner * s_inner (from the inner loop)
    // plus (s_outer - n_inner * s_inner) (the post-increment), totalling s_outer.
    let base = arr.as_ptr();
    let mut ptr_l = base as isize;
    for _l in 0..n3 {
        let mut ptr_k = ptr_l;
        for _k in 0..n2 {
            let mut ptr_j = ptr_k;
            for _j in 0..n1 {
                let mut ptr_i = ptr_j;
                for _i in 0..n0 {
                    // SAFETY: strides and dimensions match what C test expects; array sized accordingly
                    let val = unsafe { *(ptr_i as *const u32) };
                    hash_value(val, &mut h);
                    ptr_i += s[0] * 4;
                }
                // ptr_i is now at ptr_j + n0*s[0]*4; C adds (s[1]-n0*s[0]) to arr,
                // so next j-start = ptr_i + (s[1]-n0*s[0])*4 = ptr_j + s[1]*4
                ptr_j += s[1] * 4;
            }
            ptr_k += s[2] * 4;
        }
        ptr_l += s[3] * 4;
    }
    hash_finish(h)
}

/// Hash a strided 64-bit array (matches `hashStridedArray64`).
/// `n` is the logical array dimensions (unused dims are 0).
/// `s` is the element stride per dimension.
pub fn hash_strided_array64(arr: &[u64], n: [usize; 4], s: [isize; 4]) -> u64 {
    let mut h1: u32 = 0;
    let mut h2: u32 = 0;
    let n0 = n[0].max(1);
    let n1 = if n[1] > 0 { n[1] } else { 1 };
    let n2 = if n[2] > 0 { n[2] } else { 1 };
    let n3 = if n[3] > 0 { n[3] } else { 1 };

    let base = arr.as_ptr();
    let mut ptr_l = base as isize;
    for _l in 0..n3 {
        let mut ptr_k = ptr_l;
        for _k in 0..n2 {
            let mut ptr_j = ptr_k;
            for _j in 0..n1 {
                let mut ptr_i = ptr_j;
                for _i in 0..n0 {
                    // SAFETY: strides and dimensions match what C test expects; array sized accordingly
                    let val = unsafe { *(ptr_i as *const u64) };
                    hash_value((val & 0xffff_ffff) as u32, &mut h1);
                    hash_value(((val >> 32) & 0xffff_ffff) as u32, &mut h2);
                    ptr_i += s[0] * 8;
                }
                ptr_j += s[1] * 8;
            }
            ptr_k += s[2] * 8;
        }
        ptr_l += s[3] * 8;
    }
    let r1 = u64::from(hash_finish(h1));
    let r2 = u64::from(hash_finish(h2));
    r1 + (r2 << 32)
}

// ---------------------------------------------------------------------------
// Key computation (zfpChecksums.c: computeKey / computeKeyOriginalInput)
// ---------------------------------------------------------------------------

// The block variants are only constructed by the block-codec tests, which are
// `ffi`-gated; the default-feature build of this binary uses the array ones.
#[allow(dead_code)]
#[derive(Clone, Copy)]
pub enum TestType {
    BlockFull = 0,
    BlockPartial = 1,
    Array = 2,
}

#[derive(Clone, Copy)]
pub enum Subject {
    OriginalInput = 0,
    CompressedBitstream = 1,
    DecompressedArray = 2,
}

#[derive(Clone, Copy)]
#[allow(dead_code)] // Keep parity with upstream zfp mode constants; not all modes are exercised yet.
pub enum ZfpMode {
    Null = 0,
    Expert = 1,
    FixedRate = 2,
    FixedPrecision = 3,
    FixedAccuracy = 4,
    Reversible = 5,
}

/// Compute `(key1, key2)` for a given test configuration.
/// `n` is `[nx, ny, nz, nw]` with unused dimensions set to 0.
/// `misc_param` is `specialValueIndex` for block tests or `compressParamNum` for endtoend.
pub fn compute_key(
    tt: TestType,
    sjt: Subject,
    n: [usize; 4],
    mode: ZfpMode,
    misc_param: u64,
) -> (u64, u64) {
    let mut result: u64 = 0;
    result += tt as u64;
    result <<= 2;
    result += sjt as u64;
    result <<= 3;
    result += mode as u64;
    result <<= 4;
    result += misc_param;
    let key1 = result;

    // key2 stores dimensions
    let dims = if n[3] != 0 {
        4
    } else if n[2] != 0 {
        3
    } else if n[1] != 0 {
        2
    } else {
        1
    };

    let key2 = match dims {
        1 => n[0] as u64 - 1,
        2 => {
            let mut k: u64 = n[0] as u64 - 1;
            k <<= 24;
            k += n[1] as u64 - 1;
            k
        }
        3 => {
            let mut k: u64 = n[0] as u64 - 1;
            k <<= 16;
            k += n[1] as u64 - 1;
            k <<= 16;
            k += n[2] as u64 - 1;
            k
        }
        4 => {
            let mut k: u64 = n[0] as u64 - 1;
            k <<= 12;
            k += n[1] as u64 - 1;
            k <<= 12;
            k += n[2] as u64 - 1;
            k <<= 12;
            k += n[3] as u64 - 1;
            k
        }
        _ => unreachable!(),
    };

    (key1, key2)
}

/// Shorthand for computing the key for original-input checksums.
pub fn compute_key_original_input(tt: TestType, n: [usize; 4]) -> (u64, u64) {
    compute_key(tt, Subject::OriginalInput, n, ZfpMode::Null, 0)
}

// ---------------------------------------------------------------------------
// Checksum tables (ported from zfp/tests/constants/checksums/*.h)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
pub enum ZfpScalarType {
    Int32,
    Int64,
    Float,
    Double,
}

// --- 1-D ---

static CHECKSUMS_1D_INT32: &[(u64, u64, u64)] = &[
    (0x0, 0x3, 0xf3e7c054),
    (0xa0, 0x3, 0xc9d92bd5bdfd2c41),
    (0x2a0, 0x3, 0x2b7ac04c5f2c27f9),
    (0x120, 0x3, 0x4b38a824),
    (0x320, 0x3, 0xfbfb6da8),
    (0x400, 0x1000, 0x224cbf63),
    (0x4b0, 0x1000, 0x0d31e1d4f3028cea),
    (0x530, 0x1000, 0xae502d39),
    (0x4b1, 0x1000, 0x2d76d29099fb22ec),
    (0x531, 0x1000, 0xdf369702),
    (0x4b2, 0x1000, 0xb90d9da736a534a9),
    (0x532, 0x1000, 0x8e2310b0),
    (0x4a0, 0x1000, 0x804c71c729a559cf),
    (0x520, 0x1000, 0x0ff2890c),
    (0x4a1, 0x1000, 0xbe1ef33c903369a4),
    (0x521, 0x1000, 0x35a6f08e),
    (0x4a2, 0x1000, 0x8c1e4b2bdfca4bca),
    (0x522, 0x1000, 0x8e2310b0),
    (0x4d0, 0x1000, 0xcd449c2be8c8a337),
];

static CHECKSUMS_1D_INT64: &[(u64, u64, u64)] = &[
    (0x0, 0x3, 0x10decbfab896db77),
    (0xa0, 0x3, 0x103c2fc57809b590),
    (0x2a0, 0x3, 0x5a808f85fa746948),
    (0x120, 0x3, 0x321e0ab000000000),
    (0x320, 0x3, 0x1e0e4631271d520e),
    (0x400, 0x1000, 0x261f22581146db18),
    (0x4b0, 0x1000, 0x0d31e1d4f3028cea),
    (0x530, 0x1000, 0xae502d3900000000),
    (0x4b1, 0x1000, 0x2d76d29099fb22ec),
    (0x531, 0x1000, 0xdf36970200000000),
    (0x4b2, 0x1000, 0x2fa06f3672c34330),
    (0x532, 0x1000, 0xc64d5c7c923c2a4e),
    (0x4a0, 0x1000, 0x804c71c729a559cf),
    (0x520, 0x1000, 0x0ff2890c00000000),
    (0x4a1, 0x1000, 0xdf50079b903369a4),
    (0x521, 0x1000, 0xea935b1000000000),
    (0x4a2, 0x1000, 0x9de253002800ea54),
    (0x522, 0x1000, 0x0ebb9a3b522e681e),
    (0x4d0, 0x1000, 0x718abd28a6b2f034),
];

static CHECKSUMS_1D_FLOAT: &[(u64, u64, u64)] = &[
    (0x0, 0x3, 0xa35730c2),
    (0xa0, 0x3, 0x40bdf65ac73b115c),
    (0xd1, 0x3, 0x0),
    (0xd2, 0x3, 0x7baad4bceaf3d5c4),
    (0xd3, 0x3, 0xe3aab612eaf3d5c4),
    (0xd4, 0x3, 0x71d55b09a42ac833),
    (0xd5, 0x3, 0x8ca6efaedb5cd44e),
    (0xd6, 0x3, 0x15845a4aad908133),
    (0xd7, 0x3, 0xd277135bdcf92823),
    (0xd8, 0x3, 0xed4fc9b68f3c00b8),
    (0xd9, 0x3, 0x9fd129bea6d1bbd5),
    (0xda, 0x3, 0xaee692dd4f340c9f),
    (0x2a0, 0x3, 0xb4fe1804c2e28f46),
    (0x120, 0x3, 0xe7f64d14),
    (0x320, 0x3, 0x6bc01e0),
    (0x400, 0x100000, 0x81123c83),
    (0x4b0, 0x100000, 0x6698eeddef2c576f),
    (0x530, 0x100000, 0x7582fd98),
    (0x4b1, 0x100000, 0xd2f86ca7a1a270e6),
    (0x531, 0x100000, 0xa6e9b884),
    (0x4b2, 0x100000, 0x5b428a8dde9cc0c1),
    (0x532, 0x100000, 0x81123c83),
    (0x4a0, 0x100000, 0x4271157f4d1561e4),
    (0x520, 0x100000, 0xc298af05),
    (0x4a1, 0x100000, 0x450fcb1330dab01a),
    (0x521, 0x100000, 0xfe1c110c),
    (0x4a2, 0x100000, 0xae3f40d6903e54eb),
    (0x522, 0x100000, 0x81123c83),
    (0x4c0, 0x100000, 0xe8cef6c8c8ac1e62),
    (0x540, 0x100000, 0xaef278e8),
    (0x4c1, 0x100000, 0x83fe1bf6d49a1b6e),
    (0x541, 0x100000, 0x60361242),
    (0x4c2, 0x100000, 0xe7ab29faf14866d1),
    (0x542, 0x100000, 0x67e8c596),
    (0x4d0, 0x100000, 0xed4507b04c0b7919),
];

static CHECKSUMS_1D_DOUBLE: &[(u64, u64, u64)] = &[
    (0x0, 0x3, 0xb519ca1b83e2b23f),
    (0xa0, 0x3, 0xd1a4b883363919a6),
    (0xd1, 0x3, 0x0),
    (0xd2, 0x3, 0xc1de1da8),
    (0xd3, 0x3, 0xeb469308),
    (0xd4, 0x3, 0x97d201d8),
    (0xd5, 0x3, 0x49dccd6ddfc3e6d0),
    (0xd6, 0x3, 0xcfe894df52ba0b77),
    (0xd7, 0x3, 0x0dac7d74cdcc77f2),
    (0xd8, 0x3, 0xaea40aaff9d6d766),
    (0xd9, 0x3, 0xadd892805c539502),
    (0xda, 0x3, 0x30cf22a9e4dafb50),
    (0x2a0, 0x3, 0x7e0c5012d3011a34),
    (0x120, 0x3, 0xf034a06e00000000),
    (0x320, 0x3, 0x907a60b70d3a1692),
    (0x400, 0x100000, 0x49d66cd3c1044484),
    (0x4b0, 0x100000, 0xd19e1bd58ae7b771),
    (0x530, 0x100000, 0xe823070a00000000),
    (0x4b1, 0x100000, 0xd1de17cee7c8de3b),
    (0x531, 0x100000, 0x174e113400000000),
    (0x4b2, 0x100000, 0x89204000682034e7),
    (0x532, 0x100000, 0xa8c3e7eef220a0e4),
    (0x4a0, 0x100000, 0x713fc507f37f624d),
    (0x520, 0x100000, 0xeecbcbd400000000),
    (0x4a1, 0x100000, 0xa9c0457b722fce7c),
    (0x521, 0x100000, 0x5763edcdef7122e3),
    (0x4a2, 0x100000, 0xb6569815387d0248),
    (0x522, 0x100000, 0x84a661bb59df99b6),
    (0x4c0, 0x100000, 0x492797c144b2f5aa),
    (0x540, 0x100000, 0x033931390025a6c7),
    (0x4c1, 0x100000, 0x5f530b841d8ad3b2),
    (0x541, 0x100000, 0x1245fb8d26d1004b),
    (0x4c2, 0x100000, 0x8aaa2c3635420ca1),
    (0x542, 0x100000, 0x495d680180ba02ab),
    (0x4d0, 0x100000, 0x268fd6fbede5ed59),
];

// --- 2-D ---

static CHECKSUMS_2D_INT32: &[(u64, u64, u64)] = &[
    (0x0, 0x3000003, 0x94aada73),
    (0xa0, 0x3000003, 0x01264830f387e560),
    (0x2a0, 0x3000003, 0xf09d2faf2ba66c16),
    (0x120, 0x3000003, 0x3ece4105),
    (0x320, 0x3000003, 0xbb514638),
    (0x400, 0x40000040, 0xbafc4f7c),
    (0x4b0, 0x40000040, 0x2a2ecc3532b9e47c),
    (0x530, 0x40000040, 0xa0b51de9),
    (0x4b1, 0x40000040, 0xa68050a6f03bbeac),
    (0x531, 0x40000040, 0x8d1227ea),
    (0x4b2, 0x40000040, 0x298cca6049cda102),
    (0x532, 0x40000040, 0xb331c139),
    (0x4a0, 0x40000040, 0x419666be07f8fd5b),
    (0x520, 0x40000040, 0xc955273b),
    (0x4a1, 0x40000040, 0x1bb735117e4b84c0),
    (0x521, 0x40000040, 0xb2cff311),
    (0x4a2, 0x40000040, 0x45e684a399d342bf),
    (0x522, 0x40000040, 0xb331c139),
    (0x4d0, 0x40000040, 0x55be045ea7268027),
];

static CHECKSUMS_2D_INT64: &[(u64, u64, u64)] = &[
    (0x0, 0x3000003, 0x60569371027435a7),
    (0xa0, 0x3000003, 0x74905e21b1d68ae2),
    (0x2a0, 0x3000003, 0xc83e2f319f07372e),
    (0x120, 0x3000003, 0x8cdc228000000000),
    (0x320, 0x3000003, 0x6bd17a493be325d1),
    (0x400, 0x40000040, 0xf57fe1822b2a33c8),
    (0x4b0, 0x40000040, 0x2a2ecc3532b9e47c),
    (0x530, 0x40000040, 0xa0b51de900000000),
    (0x4b1, 0x40000040, 0xa68050a6f03bbeac),
    (0x531, 0x40000040, 0x8d1227ea00000000),
    (0x4b2, 0x40000040, 0x07abe90820ae730a),
    (0x532, 0x40000040, 0x4384aefdd310e015),
    (0x4a0, 0x40000040, 0x419666be07f8fd5b),
    (0x520, 0x40000040, 0xc955273b00000000),
    (0x4a1, 0x40000040, 0xd9cd09fd7e4b84c0),
    (0x521, 0x40000040, 0x74b2370100000000),
    (0x4a2, 0x40000040, 0x8bed6d7ee10836ae),
    (0x522, 0x40000040, 0xe0b475056c768219),
    (0x4d0, 0x40000040, 0x0aced76ad1c2ebb9),
];

static CHECKSUMS_2D_FLOAT: &[(u64, u64, u64)] = &[
    (0x0, 0x3000003, 0xd61ebeeb),
    (0xa0, 0x3000003, 0xda4c301a8e0f8cee),
    (0xd1, 0x3000003, 0x0),
    (0xd2, 0x3000003, 0xeabd0942eaf3d5c4),
    (0xd3, 0x3000003, 0x37364bbaeaf3d5c4),
    (0xd4, 0x3000003, 0x1bab25dda42ac833),
    (0xd5, 0x3000003, 0x15efe9eb467df9de),
    (0xd6, 0x3000003, 0x0646c26ae1386d3f),
    (0xd7, 0x3000003, 0x927724ec9f90816d),
    (0xd8, 0x3000003, 0x8d27f49059a9fe98),
    (0xd9, 0x3000003, 0x9d4930c42f82c1fb),
    (0xda, 0x3000003, 0x11acf6d756257748),
    (0x2a0, 0x3000003, 0x584942f81bec40fb),
    (0x120, 0x3000003, 0xd183b619),
    (0x320, 0x3000003, 0x713809a7),
    (0x400, 0x400000400, 0x0e4bfe4e),
    (0x4b0, 0x400000400, 0x8417e3e4287f38b5),
    (0x530, 0x400000400, 0x9b77e022),
    (0x4b1, 0x400000400, 0xf5356ab8f5b59e8a),
    (0x531, 0x400000400, 0x541a3433),
    (0x4b2, 0x400000400, 0xa537f64220d1fc1d),
    (0x532, 0x400000400, 0x0e4bfe4e),
    (0x4a0, 0x400000400, 0xd183e05b7c3be5eb),
    (0x520, 0x400000400, 0x5198a34b),
    (0x4a1, 0x400000400, 0x254679da05758c1a),
    (0x521, 0x400000400, 0x0b9126f4),
    (0x4a2, 0x400000400, 0x72cd5c52aa46c2da),
    (0x522, 0x400000400, 0x0e4bfe4e),
    (0x4c0, 0x400000400, 0x211f16ea5922b678),
    (0x540, 0x400000400, 0x2e0e3c8b),
    (0x4c1, 0x400000400, 0xf2a1526474d8ee29),
    (0x541, 0x400000400, 0xb6a7efcb),
    (0x4c2, 0x400000400, 0x53ed9feb9ca6dd1a),
    (0x542, 0x400000400, 0x18bad4a1),
    (0x4d0, 0x400000400, 0x0e91cd56d5db78ef),
];

static CHECKSUMS_2D_DOUBLE: &[(u64, u64, u64)] = &[
    (0x0, 0x3000003, 0x1c772c230f3ccbb4),
    (0xa0, 0x3000003, 0xc0a1814da6ce303b),
    (0xd1, 0x3000003, 0x0),
    (0xd2, 0x3000003, 0x83b0d73d),
    (0xd3, 0x3000003, 0x289bae9d),
    (0xd4, 0x3000003, 0x5d6e57cf),
    (0xd5, 0x3000003, 0x52a77478bd871422),
    (0xd6, 0x3000003, 0x5f99e005089267e0),
    (0xd7, 0x3000003, 0x7d108dc9451d2cb7),
    (0xd8, 0x3000003, 0x34c72b14e6ed9d8c),
    (0xd9, 0x3000003, 0xba67c09098a3d01a),
    (0xda, 0x3000003, 0x7d108dc9271c8a60),
    (0x2a0, 0x3000003, 0xf47a9d0740fd12f1),
    (0x120, 0x3000003, 0x7ac02ede00000000),
    (0x320, 0x3000003, 0x12ef3cd64903bcca),
    (0x400, 0x400000400, 0x0856a073a7252dd4),
    (0x4b0, 0x400000400, 0xe4efc0e6e0c4937f),
    (0x530, 0x400000400, 0x8e010bbc00000000),
    (0x4b1, 0x400000400, 0x26ab1ab12b69d8e7),
    (0x531, 0x400000400, 0xa296ec5400000000),
    (0x4b2, 0x400000400, 0xd7605316605ae257),
    (0x532, 0x400000400, 0x626c78e0852013ee),
    (0x4a0, 0x400000400, 0x10288c2054631266),
    (0x520, 0x400000400, 0xd5495117b8fe1c02),
    (0x4a1, 0x400000400, 0xb1d8865622fe6fc0),
    (0x521, 0x400000400, 0x9437836903fc33a1),
    (0x4a2, 0x400000400, 0x816b6359b90eaba1),
    (0x522, 0x400000400, 0x124ac89d7f6e6511),
    (0x4c0, 0x400000400, 0x7cb428be5481bd7b),
    (0x540, 0x400000400, 0x2229f480c522c420),
    (0x4c1, 0x400000400, 0xf94462ab31afa215),
    (0x541, 0x400000400, 0x25f62aac2713f851),
    (0x4c2, 0x400000400, 0x8beb41214f9ee0d6),
    (0x542, 0x400000400, 0x94fd382138403fb1),
    (0x4d0, 0x400000400, 0x1481e46e30d0f3ab),
];

// --- 3-D ---

static CHECKSUMS_3D_INT32: &[(u64, u64, u64)] = &[
    (0x0, 0x300030003, 0xab8e83e9),
    (0xa0, 0x300030003, 0x0da55ac5950c74c2),
    (0x2a0, 0x300030003, 0xb85a3bd936a5c392),
    (0x120, 0x300030003, 0xdbb57cfa),
    (0x320, 0x300030003, 0x205d2fad),
    (0x400, 0x1000100010, 0xad7ade47),
    (0x4b0, 0x1000100010, 0xc92ee0e3f6e6aa91),
    (0x530, 0x1000100010, 0xd2482c01),
    (0x4b1, 0x1000100010, 0x21b0a7777c2c5b2d),
    (0x531, 0x1000100010, 0x9436e0c7),
    (0x4b2, 0x1000100010, 0xfe72d7ca4ce4cd2b),
    (0x532, 0x1000100010, 0xea428b3e),
    (0x4a0, 0x1000100010, 0x32942f0afdb349c2),
    (0x520, 0x1000100010, 0xb3d2ff2c),
    (0x4a1, 0x1000100010, 0x3a036901bbfdee14),
    (0x521, 0x1000100010, 0xb9258768),
    (0x4a2, 0x1000100010, 0x8a8ae9c57224ef8e),
    (0x522, 0x1000100010, 0xea428b3e),
    (0x4d0, 0x1000100010, 0xf0ab4d96d89cc545),
];

static CHECKSUMS_3D_INT64: &[(u64, u64, u64)] = &[
    (0x0, 0x300030003, 0xcc5133515849571c),
    (0xa0, 0x300030003, 0x6c0ff959c2207d41),
    (0x2a0, 0x300030003, 0xd6b771a93e2404f4),
    (0x120, 0x300030003, 0x4b9f52d500000000),
    (0x320, 0x300030003, 0xc78c8cca00000000),
    (0x400, 0x1000100010, 0xee34e487f557278f),
    (0x4b0, 0x1000100010, 0xc92ee0e3f6e6aa91),
    (0x530, 0x1000100010, 0xd2482c0100000000),
    (0x4b1, 0x1000100010, 0x21b0a7777c2c5b2d),
    (0x531, 0x1000100010, 0x9436e0c700000000),
    (0x4b2, 0x1000100010, 0xa8b1239155fdd8ab),
    (0x532, 0x1000100010, 0xc723b42e1e4f2274),
    (0x4a0, 0x1000100010, 0x32942f0afdb349c2),
    (0x520, 0x1000100010, 0xb3d2ff2c00000000),
    (0x4a1, 0x1000100010, 0x84e238f16919a151),
    (0x521, 0x1000100010, 0x879bc89700000000),
    (0x4a2, 0x1000100010, 0x4e6417e960207269),
    (0x522, 0x1000100010, 0xc348c52175d9ec77),
    (0x4d0, 0x1000100010, 0x43c8d544f70dccc5),
];

static CHECKSUMS_3D_FLOAT: &[(u64, u64, u64)] = &[
    (0x0, 0x300030003, 0x54572f34),
    (0xa0, 0x300030003, 0x6ad38b388f18d118),
    (0xd1, 0x300030003, 0x0),
    (0xd2, 0x300030003, 0x04e0632d046a7a0e),
    (0xd3, 0x300030003, 0xf375ed06da7f218c),
    (0xd4, 0x300030003, 0x79aaf683295b4527),
    (0xd5, 0x300030003, 0xc71006bf172ec200),
    (0xd6, 0x300030003, 0x163602c727dbbba2),
    (0xd7, 0x300030003, 0xda199ff1947e73d2),
    (0xd8, 0x300030003, 0x84db4dd6885773b5),
    (0xd9, 0x300030003, 0x68a0b34799c2f1f8),
    (0xda, 0x300030003, 0xd4b6310ae6d2d4de),
    (0x2a0, 0x300030003, 0x256107e3209389ee),
    (0x120, 0x300030003, 0xd822c66d),
    (0x320, 0x300030003, 0xc3e0b4fb),
    (0x400, 0x8000800080, 0xdbe7e231),
    (0x4b0, 0x8000800080, 0xe3b79c04a6174576),
    (0x530, 0x8000800080, 0x6ea0403c),
    (0x4b1, 0x8000800080, 0xb666d473ca7d7e1c),
    (0x531, 0x8000800080, 0xc2408604),
    (0x4b2, 0x8000800080, 0xb0c2a41ec2111183),
    (0x532, 0x8000800080, 0x97a819ae),
    (0x4a0, 0x8000800080, 0x5ec963fda5ed8273),
    (0x520, 0x8000800080, 0xb0695ba7),
    (0x4a1, 0x8000800080, 0xa72b103d6027cbef),
    (0x521, 0x8000800080, 0xcdd7c8b6),
    (0x4a2, 0x8000800080, 0x4e2c7e0bf502c3a1),
    (0x522, 0x8000800080, 0x97a819ae),
    (0x4c0, 0x8000800080, 0xdb9351a2125e34e4),
    (0x540, 0x8000800080, 0x3518c38f),
    (0x4c1, 0x8000800080, 0x02fd5a60cdd2227e),
    (0x541, 0x8000800080, 0x4f0985dd),
    (0x4c2, 0x8000800080, 0x73829fdec12a0374),
    (0x542, 0x8000800080, 0x0cb6afbd),
    (0x4d0, 0x8000800080, 0xb6475a8758f10fe0),
];

static CHECKSUMS_3D_DOUBLE: &[(u64, u64, u64)] = &[
    (0x0, 0x300030003, 0x5f9e82c4fef6f593),
    (0xa0, 0x300030003, 0x20a6c761afd4380b),
    (0xd1, 0x300030003, 0x0),
    (0xd2, 0x300030003, 0x927724ecdcf219fb),
    (0xd3, 0x300030003, 0x393a6de095b240e0),
    (0xd4, 0x300030003, 0x9c9d36f01c36b045),
    (0xd5, 0x300030003, 0x0a71ba2fe0b649fc),
    (0xd6, 0x300030003, 0x7e8c15054d871bd9),
    (0xd7, 0x300030003, 0xe4eab78245c08a26),
    (0xd8, 0x300030003, 0x5ac46921892607c6),
    (0xd9, 0x300030003, 0xbfb026919c6944c4),
    (0xda, 0x300030003, 0xf3420697931ed828),
    (0x2a0, 0x300030003, 0x9a658e0fe05b9657),
    (0x120, 0x300030003, 0x4f653444ff3fdbe4),
    (0x320, 0x300030003, 0x7e8c64faafedcb18),
    (0x400, 0x8000800080, 0xb29ddfb4a7719b6a),
    (0x4b0, 0x8000800080, 0x6b5b0dab297c9d33),
    (0x530, 0x8000800080, 0x1e497b1f00000000),
    (0x4b1, 0x8000800080, 0xe933645e8cf7a7c9),
    (0x531, 0x8000800080, 0xdce089f900000000),
    (0x4b2, 0x8000800080, 0xc3d061d1944a8106),
    (0x532, 0x8000800080, 0x3817c78441377d10),
    (0x4a0, 0x8000800080, 0xd3b75ae8488a556d),
    (0x520, 0x8000800080, 0x0f4bd2afd74af921),
    (0x4a1, 0x8000800080, 0x8d8d80142436d812),
    (0x521, 0x8000800080, 0x9103ee0106602bb1),
    (0x4a2, 0x8000800080, 0x64e50911ed54c0ef),
    (0x522, 0x8000800080, 0xcd7da85356f9db40),
    (0x4c0, 0x8000800080, 0xca9d4c2be9c2a15b),
    (0x540, 0x8000800080, 0x8eaf0fa126b3de89),
    (0x4c1, 0x8000800080, 0x8f79006fd9e45619),
    (0x541, 0x8000800080, 0xb0dd4ed6a7196f47),
    (0x4c2, 0x8000800080, 0x5c056eecba4d5349),
    (0x542, 0x8000800080, 0x3262044561f9cceb),
    (0x4d0, 0x8000800080, 0xaf95ff6301796621),
];

// --- 4-D ---

static CHECKSUMS_4D_INT32: &[(u64, u64, u64)] = &[
    (0x0, 0x3003003003, 0x08b21ff0),
    (0xa0, 0x3003003003, 0xf89b3fdf64ff5b5b),
    (0x2a0, 0x3003003003, 0x8d094f52b8fd6250),
    (0x120, 0x3003003003, 0xcaa8e882),
    (0x320, 0x3003003003, 0x86320cb4),
    (0x400, 0x8008008008, 0x89f6c535),
    (0x4b0, 0x8008008008, 0x38d58bf8bf7f5b07),
    (0x530, 0x8008008008, 0xbd347efd),
    (0x4b1, 0x8008008008, 0xb9f8a476db61b946),
    (0x531, 0x8008008008, 0x6f0e9866),
    (0x4b2, 0x8008008008, 0xb44975c2cdae2907),
    (0x532, 0x8008008008, 0x539b74c9),
    (0x4a0, 0x8008008008, 0xabd0b79d9c135337),
    (0x520, 0x8008008008, 0x5a8a7db4),
    (0x4a1, 0x8008008008, 0xe331fda805ba7319),
    (0x521, 0x8008008008, 0xec560874),
    (0x4a2, 0x8008008008, 0xc934178cb9e06ff5),
    (0x522, 0x8008008008, 0x539b74c9),
    (0x4d0, 0x8008008008, 0x08c888a65b12c884),
];

static CHECKSUMS_4D_INT64: &[(u64, u64, u64)] = &[
    (0x0, 0x3003003003, 0xc9f0cadc2b040375),
    (0xa0, 0x3003003003, 0xbe695ffaef2d6055),
    (0x2a0, 0x3003003003, 0x3bf1627a5fd514a7),
    (0x120, 0x3003003003, 0x83e0508600000000),
    (0x320, 0x3003003003, 0xa194665700000000),
    (0x400, 0x8008008008, 0x3c7a84c24a0d97db),
    (0x4b0, 0x8008008008, 0x38d58bf8bf7f5b07),
    (0x530, 0x8008008008, 0xbd347efd00000000),
    (0x4b1, 0x8008008008, 0xb9f8a476db61b946),
    (0x531, 0x8008008008, 0x6f0e986600000000),
    (0x4b2, 0x8008008008, 0xf1324a2092943e33),
    (0x532, 0x8008008008, 0x6b2d4650a70cb4be),
    (0x4a0, 0x8008008008, 0xabd0b79d9c135337),
    (0x520, 0x8008008008, 0x5a8a7db400000000),
    (0x4a1, 0x8008008008, 0x4269d84b05ba7319),
    (0x521, 0x8008008008, 0x73c78b5d00000000),
    (0x4a2, 0x8008008008, 0x03009aef996d98fa),
    (0x522, 0x8008008008, 0x230d83c5490fa7dd),
    (0x4d0, 0x8008008008, 0x6e014f2638fd24d2),
];

static CHECKSUMS_4D_FLOAT: &[(u64, u64, u64)] = &[
    (0x0, 0x3003003003, 0x8c5867f7),
    (0xa0, 0x3003003003, 0x26580b0af77ece38),
    (0xd1, 0x3003003003, 0x0),
    (0xd2, 0x3003003003, 0xbd6cb9cd6d2735e7),
    (0xd3, 0x3003003003, 0x943b526033810b7b),
    (0xd4, 0x3003003003, 0x4a1da930b29a371d),
    (0xd5, 0x3003003003, 0x276e4805f1ee4de9),
    (0xd6, 0x3003003003, 0xcd3a562a15f5f5e9),
    (0xd7, 0x3003003003, 0x241372c19e9d3507),
    (0xd8, 0x3003003003, 0xbfeec5a9344e5b48),
    (0xd9, 0x3003003003, 0xeac7292d88f982bf),
    (0xda, 0x3003003003, 0x667025a3a09f4198),
    (0x2a0, 0x3003003003, 0xb47c6e115d00b400),
    (0x120, 0x3003003003, 0xda2b72f9),
    (0x320, 0x3003003003, 0xfc12ceb2),
    (0x400, 0x20020020020, 0x725f89ff),
    (0x4b0, 0x20020020020, 0x59e13fe363db5c6f),
    (0x530, 0x20020020020, 0xb444287b),
    (0x4b1, 0x20020020020, 0x23d6299eeaa79a9e),
    (0x531, 0x20020020020, 0x52fe0450),
    (0x4b2, 0x20020020020, 0x69ff59c816afd8bd),
    (0x532, 0x20020020020, 0x0d916b61),
    (0x4a0, 0x20020020020, 0xeaae9de596da1479),
    (0x520, 0x20020020020, 0xf29a4049),
    (0x4a1, 0x20020020020, 0xf0e8cdfbb12d5bdb),
    (0x521, 0x20020020020, 0x8bc47f0d),
    (0x4a2, 0x20020020020, 0x669314f4f9637698),
    (0x522, 0x20020020020, 0x0d916b61),
    (0x4c0, 0x20020020020, 0xc22cce8aa431fffb),
    (0x540, 0x20020020020, 0x0083f41e),
    (0x4c1, 0x20020020020, 0x4b0af02b26351468),
    (0x541, 0x20020020020, 0x425b2a0d),
    (0x4c2, 0x20020020020, 0xedc915189e4764f2),
    (0x542, 0x20020020020, 0x3ca6456a),
    (0x4d0, 0x20020020020, 0xf2a9c72c87868054),
];

static CHECKSUMS_4D_DOUBLE: &[(u64, u64, u64)] = &[
    (0x0, 0x3003003003, 0x061f9b8c3ddcbe9b),
    (0xa0, 0x3003003003, 0x3bd0d8f2da9e9acf),
    (0xd1, 0x3003003003, 0x0),
    (0xd2, 0x3003003003, 0xd7e189562d39c484),
    (0xd3, 0x3003003003, 0x10254b13cbda2b97),
    (0xd4, 0x3003003003, 0x08126589d3735e9d),
    (0xd5, 0x3003003003, 0x81c6ab2ae3cbfbac),
    (0xd6, 0x3003003003, 0xb7521b04e50f0123),
    (0xd7, 0x3003003003, 0x2d382b62747da555),
    (0xd8, 0x3003003003, 0xdff214dccadfe445),
    (0xd9, 0x3003003003, 0xdab8e4ea9761352b),
    (0xda, 0x3003003003, 0x65fa916b3e3e928e),
    (0x2a0, 0x3003003003, 0x1dbd79f0a52ec95b),
    (0x120, 0x3003003003, 0xbf4272427fcd2646),
    (0x320, 0x3003003003, 0xdf41e51abd93ea8a),
    (0x400, 0x20020020020, 0xe1c8a968261e4559),
    (0x4b0, 0x20020020020, 0x7a0d035888f5d7e3),
    (0x530, 0x20020020020, 0x9940d71100000000),
    (0x4b1, 0x20020020020, 0xc0f286466ade809e),
    (0x531, 0x20020020020, 0xaead20c8a88f2622),
    (0x4b2, 0x20020020020, 0xbfd8f5f591cb0f2d),
    (0x532, 0x20020020020, 0x4de9a84f4ab886fa),
    (0x4a0, 0x20020020020, 0x94219b32ec93e2a9),
    (0x520, 0x20020020020, 0xa1ca18c21794908b),
    (0x4a1, 0x20020020020, 0x464485425bf411aa),
    (0x521, 0x20020020020, 0x7abae3fe33d0ce6a),
    (0x4a2, 0x20020020020, 0x8ca875e7386e2cea),
    (0x522, 0x20020020020, 0x6a44d79a5a33d47d),
    (0x4c0, 0x20020020020, 0xc0b867f744b71cc0),
    (0x540, 0x20020020020, 0xb8f1525cd842fbd5),
    (0x4c1, 0x20020020020, 0x6a1908a569eb1a99),
    (0x541, 0x20020020020, 0xb14abd4386fddb81),
    (0x4c2, 0x20020020020, 0xb920196fca3513eb),
    (0x542, 0x20020020020, 0x249da35a6e8ca411),
    (0x4d0, 0x20020020020, 0xd72af5ab206ebe50),
];

// ---------------------------------------------------------------------------
// Lookup
// ---------------------------------------------------------------------------

/// Look up the expected checksum for the given (dims, type, key1, key2).
/// Returns `None` if no matching entry is found.
pub fn get_checksum(dims: u32, zfp_type: ZfpScalarType, key1: u64, key2: u64) -> Option<u64> {
    let table: &[(u64, u64, u64)] = match (dims, zfp_type) {
        (1, ZfpScalarType::Int32) => CHECKSUMS_1D_INT32,
        (1, ZfpScalarType::Int64) => CHECKSUMS_1D_INT64,
        (1, ZfpScalarType::Float) => CHECKSUMS_1D_FLOAT,
        (1, ZfpScalarType::Double) => CHECKSUMS_1D_DOUBLE,
        (2, ZfpScalarType::Int32) => CHECKSUMS_2D_INT32,
        (2, ZfpScalarType::Int64) => CHECKSUMS_2D_INT64,
        (2, ZfpScalarType::Float) => CHECKSUMS_2D_FLOAT,
        (2, ZfpScalarType::Double) => CHECKSUMS_2D_DOUBLE,
        (3, ZfpScalarType::Int32) => CHECKSUMS_3D_INT32,
        (3, ZfpScalarType::Int64) => CHECKSUMS_3D_INT64,
        (3, ZfpScalarType::Float) => CHECKSUMS_3D_FLOAT,
        (3, ZfpScalarType::Double) => CHECKSUMS_3D_DOUBLE,
        (4, ZfpScalarType::Int32) => CHECKSUMS_4D_INT32,
        (4, ZfpScalarType::Int64) => CHECKSUMS_4D_INT64,
        (4, ZfpScalarType::Float) => CHECKSUMS_4D_FLOAT,
        (4, ZfpScalarType::Double) => CHECKSUMS_4D_DOUBLE,
        _ => return None,
    };
    table
        .iter()
        .find(|&&(k1, k2, _)| k1 == key1 && k2 == key2)
        .map(|&(_, _, checksum)| checksum)
}
