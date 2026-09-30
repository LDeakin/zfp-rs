# Fuzzing `zfp-rs`

Coverage-guided fuzzing with [cargo-fuzz] / libFuzzer.

The existing test suite is strong on *conformance*: `tests/c/` ports the upstream cmocka suite
with its checksum tables, and `tests/proptest/` compares byte-for-byte against the real C library
through `zfp-sys`. What none of it does is feed the decoder bytes the encoder did not write.
These targets do, plus two things proptest structurally cannot reach: deep randomised operation
*sequences*, and float values the differential tests must exclude (the encoding proptests filter
out blocks below 2^-98 for `f32` and 2^-962 for `f64`, where C's `fwd_cast` overflows and zfp-rs
deliberately differs — here they are squarely in scope).

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
| `config_mode` | none | mode-word round-tripping, `expert()` and `from_raw_params()`, and `maximum_size` with unbounded dims |
| `bitstream_ops` | none | deep random sequences of bitstream cursor operations, with bit counts above 64, offsets near `u64::MAX`, and huge pads and copies |

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
- **zfp's absolute error bound for fixed-accuracy mode.** A block spanning a wide dynamic range
  runs out of precision above its small elements, which can then exceed the tolerance. C does the
  same (zfp FAQ Q17), and `tests/proptest/wide_range_accuracy.rs` pins zfp-rs to C's bytes and
  values there. `roundtrip` makes no accuracy assertion for the lossy modes.
- **Finite input staying finite.** See the known-open finding below.
- **`read_pos` staying in range.** `bits` is shared between the read and write buffers, so querying
  the read position on a write-only stream wraps. C's `stream_rtell` does the same and the
  differential tests assert against it.
- **Anything about `ZfpConfig::from_raw_params` with out-of-range parameters.** Unlike
  `ZfpConfig::expert`, it performs no validation,
  so it will build configs with `min_bits = 0` or `max_bits` above `ZFP_MAX_BITS`; those get
  misclassified into a short mode form and the encoding is then not idempotent. `config_mode`
  asserts only that such configs do not *panic*, and that `expert` rejects exactly the null-mode
  ones.

## Known-open findings

### Finite input can decompress to infinity

Large finite magnitudes can reconstruct to infinity in the lossy modes, so `roundtrip` does not
assert that finite input stays finite. No reproducer is committed, and whether C does the same is
unsettled. `tests/proptest/wide_range_accuracy.rs` is the place to compare such a block against
C; `[MAX, MAX, -MAX, MAX]` at tolerance 1 reconstructs exactly in both.

## Coverage gaps

- `ModeSpec` floors expert `max_bits` at `MIN_EXPERT_BITS` (64), a guard from before
  `maximum_size` counted block headers. Budgets below the header are covered by
  `tests/header_checked.rs`, `tests/rayon_low_budget.rs` and `tests/proptest/differences.rs`
  instead.
- Only `block_codec` varies `ZfpRounding`; `roundtrip` and `decompress_stream` use the default.

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
