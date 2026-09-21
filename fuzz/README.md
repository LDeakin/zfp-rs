# Fuzzing `zfp-rs`

Coverage-guided fuzzing with [cargo-fuzz] / libFuzzer.

The existing test suite is strong on *conformance*: `tests/c/` ports the upstream cmocka suite
with its checksum tables, and `tests/proptest/` compares byte-for-byte against the real C library
through `zfp-sys`. What none of it does is feed the decoder bytes the encoder did not write.
These targets do, plus two things proptest structurally cannot reach: deep randomised operation
*sequences*, and float values the differential tests must exclude (subnormals are filtered out in
four proptest files because C's `fwd_cast` overflow is implementation-defined — here they are
squarely in scope).

## Layout

| Path | What it is |
|---|---|
| `../zfp-fuzz-common/` | A **workspace member** holding the input model and every target body |
| `fuzz_targets/*.rs` | Five-line libFuzzer wrappers |
| `seeds/<target>/` | Committed starting corpus, regenerate with `just fuzz_seeds` |
| `regressions/<target>/` | Committed crash inputs, replayed on **stable** by `cargo test` |
| `corpus/`, `artifacts/` | libFuzzer scratch, gitignored |

Every target body is `pub fn run(data: &[u8])` in `zfp-fuzz-common`, and the harness always takes
`&[u8]` (never `fuzz_target!(|x: MyType|)`). That is the load-bearing decision here: it means a
crash artifact is a raw byte string that `zfp-fuzz-common/tests/regressions.rs` can replay under
plain `cargo test --workspace`, with no nightly toolchain and no cargo-fuzz. It also keeps the
real logic inside the workspace, where `cargo clippy --workspace --all-targets` lints it —
`fuzz/` itself is excluded from the workspace because it needs nightly.

## Running

```sh
cargo install cargo-fuzz          # once
just fuzz_build                   # compile all targets
just fuzz roundtrip 300           # run one target for 300s
just fuzz_smoke                   # 30s of each, mirrors CI
just fuzz_regressions             # replay committed crashes (stable)
just fuzz_clippy                  # `just clippy` cannot see this crate
```

## Targets

| Target | Sanitizer | What it attacks |
|---|---|---|
| `roundtrip` | address | compress → decompress over every (type × rank × mode × execution) combination |
| `decompress_stream` | address | **untrusted bytes** decompressed into a well-formed field |
| `block_codec` | address | strided gather/scatter with permuted, gapped and negative strides |
| `header_decode` | none | `read_header` on arbitrary bytes across all eight mask combinations |
| `config_mode` | none | mode-word round-tripping, `expert()`, and `maximum_size` with unbounded dims |
| `bitstream_ops` | none | deep random sequences of bitstream cursor operations |

Run the ASan ones with the default sanitizer. The bottom three touch no `unsafe`, so `-s none`
roughly triples their throughput:

```sh
cargo +nightly fuzz run -s none header_decode fuzz/corpus/header_decode fuzz/seeds/header_decode
```

`roundtrip` is what justifies having no C oracle. Reversible mode is bit-exact for *every* input
pattern — when the block floating-point path fails its reversibility check the encoder falls back
to raw two's complement (`src/codec/encode/reversible.rs`) — so it is a strict oracle that, unlike
the differential proptests, may be fed NaN, infinities and subnormals.

### Why there is no C-differential target

`tests/proptest/` already sweeps the whole matrix against C. Byte-exactness is a global property
with no narrow input predicate for coverage guidance to search for, so a fuzzer adds little. And
cargo-fuzz's ASan `RUSTFLAGS` never reach `zfp-sys`'s `cc` build, so C-side overreads into Rust
buffers would surface as false positives with Rust-looking stack traces.

## Oracles: what is deliberately *not* asserted

Getting these wrong wedges a fuzzer on false crashes, so each exclusion is load-bearing.

- **`encode(decode(x)) == encode(x)`.** zfp's inverse transform rounds, so lossy modes are not a
  fixed point. Asserting it produces false crashes.
- **`from_mode(c.mode_bits()) == c`.** Parameters outside the short mode forms are legitimately
  reclassified (a `min_exp` below `ZFP_MIN_EXP` becomes reversible). The real invariant, and the
  one asserted, is that the encoding is *idempotent*.
- **zfp's absolute error bound for fixed-accuracy mode.** See the second known-open finding below;
  `check_accuracy` asserts only that a finite input does not decompress to a non-finite value.
- **`read_pos` staying in range.** `bits` is shared between the read and write buffers, so querying
  the read position on a write-only stream wraps. C's `stream_rtell` does the same and the
  differential tests assert against it.
- **Anything about `ZfpConfig::expert` with out-of-range parameters.** It performs no validation,
  so it will build configs with `min_bits = 0` or `max_bits` above `ZFP_MAX_BITS`; those get
  misclassified into a short mode form and the encoding is then not idempotent. `config_mode`
  asserts only that such configs do not *panic*.

## Known-open findings

### 1. Encoder can exceed `maximum_size` when `max_bits` is below the exponent header width.**
A block emits its exponent header (1 + 11 bits for `f64`, 1 + 8 for `f32`) before it can honour a
bit budget, so `ZfpConfig::expert(1, 1, 1, -1074)` makes every block overshoot, and
`maximum_size` — which trusts `max_bits` — under-reports the buffer the encoder needs. For the
`zfp-rs-ffi` drop-in that is a heap overflow in the C caller's buffer.

The C reference behaves the same way, and worse: its `maxbits - bits` subtraction wraps, so it
overshoots further. `ZfpConfig::fixed_rate` clamps to `1 + 11` precisely to avoid this; only the
unvalidated `ZfpConfig::expert` can reach it. `ModeSpec` therefore floors expert `max_bits` at
`MIN_EXPERT_BITS` (`zfp-fuzz-common/src/input.rs`). Removing that floor reproduces the overshoot
immediately. Fixing it properly means validating `expert()`, which is an API decision.

### 2. Fixed-accuracy error bound over a wide dynamic range

A block containing values many binades apart — `7.9e-24` alongside elements near `1e30` — exhausts
the 64-bit precision cap, and the small elements then reconstruct far outside the requested
tolerance (observed: an element `7.9e-24` decompressing to `-9.5e29` with a tolerance of `6.2e26`).

Whether that is a `zfp-rs` divergence or inherent to zfp's block-floating-point representation
needs the C reference to settle, and the C oracle is deliberately kept out of the fuzz targets.
`seeds/roundtrip/known-open-fixed-accuracy-wide-dynamic-range.bin` reproduces it. Until it is
adjudicated, `roundtrip` asserts only that finite inputs stay finite; the exact correctness oracle
is reversible mode, which is unaffected.

Settling it is a good use of the existing differential harness: extend
`tests/proptest/compress_compat.rs` with wide-dynamic-range fixed-accuracy fields and compare
element-by-element against `zfp_decompress`.

## When a fuzzer finds a crash

`artifacts/` is libFuzzer scratch and stays gitignored. To make a finding permanent:

1. Minimize it: `just fuzz_tmin <target> fuzz/artifacts/<target>/crash-<sha1>`
2. Commit it as `fuzz/regressions/<target>/<description>-<sha1[..8]>.bin`
3. Fix the bug. `cargo test --workspace` now covers it forever, on stable.
4. Copy the input into `seeds/<target>/` too, so future sessions start from the interesting shape.

Only commit a regression **once the bug is fixed** — `regressions/` must stay green, since it runs
in the ordinary test job. An input for a known-open finding belongs in `seeds/`, documented above.

One caveat when triaging: the replay harness runs without ASan, so a crash that is purely a heap
overflow into a live neighbouring allocation reproduces there as a silent pass. For those, add an
explicit assertion inside the target body so the case fails on stable too.

## Bounds

`zfp-fuzz-common/src/limits.rs` caps fields at 4096 elements and streams at 1 MiB. Every target
also computes `config.maximum_size(...)` and skips the input if it is zero or over the cap — since
that value is `num_blocks * max_bits`, one check bounds both memory and runtime. Keep libFuzzer's
`-rss_limit_mb` / `-malloc_limit_mb` / `-timeout` flags as well; the cap alone is not enough if a
future target computes its size differently.

## Portability

The bitstream is 64-bit words in **native** byte order, and bit-exact C compatibility requires
little-endian (`src/lib.rs`). Corpora are therefore not portable across endianness — do not share
one with a big-endian job, and only fuzz on little-endian targets.

[cargo-fuzz]: https://rust-fuzz.github.io/book/cargo-fuzz.html
