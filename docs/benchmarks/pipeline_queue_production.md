# Production bounded-queue evaluation

2026-10-02, Intel Core Ultra 7 268V, Rust 1.98.1. Baseline: the earlier two-buffer implementation, which is not part of this repository's history. Candidate: the production FIFO queue in this working tree. Every speedup below compares these implementations directly; it is not a speedup over serial decoding.

## Behavior and configuration

`ZfpExecution::Rayon` automatically uses the bounded queue for variable-rate decompression. The library-wide default remains `Serial`. There is no new execution variant or queue-depth setting.

One reader fills plane buffers and immediately dispatches reconstruction batches through Rayon FIFO scopes. Workers return buffers to a preallocated FIFO protected by a short-held mutex. A lease returns its buffer on both normal completion and unwinding. A producer waiting for a free buffer releases the lock and helps pending Rayon work, allowing nested decoding calls to progress.

With `chunk_size: 0`, batches target 64 KiB of plane storage and use at most 16 buffers, approximately 1 MiB total. An explicit `chunk_size` remains a count of blocks, now per batch; large batches reduce the number of available reconstruction tasks. All buffers and recycling slots are allocated before any stream read or output write, so setup failure can fall back to serial. Fixed-rate decoding, compression, the compressed format and cursor/error semantics retain their existing behavior.

## Results: all 40 variable-rate cases

Each field has 1,048,576 elements. Means give equal weight to cases. Values below 1 indicate slower decoding.

| Threads | Geometric mean speedup | Median | Range |
|---|---:|---:|---:|
| 2 | 1.163× | 1.089× | 0.918–1.628× |
| 4 | 1.018× | 1.003× | 0.922–1.322× |
| 8 | 1.091× | 1.069× | 0.999–1.606× |

The strongest gains are concentrated in reconstruction-heavy cases at two threads. The queue is not universally faster, and more threads do not guarantee a larger improvement. These are synthetic benchmark fields, not a representative application corpus.

| Case | 2 threads | 4 threads | 8 threads |
|---|---:|---:|---:|
| f32_d1_fixed_accuracy | 0.986× | 0.961× | 1.102× |
| f32_d1_fixed_precision | 0.957× | 1.033× | 0.999× |
| f32_d1_reversible | 0.965× | 0.960× | 1.037× |
| f32_d2_fixed_accuracy | 1.031× | 1.029× | 1.037× |
| f32_d2_fixed_precision | 1.024× | 0.929× | 1.083× |
| f32_d2_reversible | 0.990× | 1.029× | 1.004× |
| f32_d3_fixed_accuracy | 1.197× | 0.998× | 1.097× |
| f32_d3_fixed_precision | 1.202× | 0.989× | 1.084× |
| f32_d3_reversible | 1.050× | 1.070× | 1.056× |
| f32_d4_fixed_accuracy | 1.273× | 1.011× | 1.181× |
| f32_d4_fixed_precision | 1.367× | 1.020× | 1.128× |
| f32_d4_reversible | 1.259× | 1.003× | 1.059× |
| f64_d1_fixed_accuracy | 0.989× | 0.993× | 1.113× |
| f64_d1_fixed_precision | 0.977× | 0.993× | 1.063× |
| f64_d1_reversible | 0.956× | 0.965× | 1.026× |
| f64_d2_fixed_accuracy | 1.015× | 0.922× | 1.030× |
| f64_d2_fixed_precision | 1.022× | 1.017× | 1.145× |
| f64_d2_reversible | 0.999× | 1.026× | 1.024× |
| f64_d3_fixed_accuracy | 1.339× | 1.032× | 1.027× |
| f64_d3_fixed_precision | 1.391× | 1.021× | 1.099× |
| f64_d3_reversible | 1.048× | 0.986× | 1.068× |
| f64_d4_fixed_accuracy | 1.325× | 0.980× | 1.050× |
| f64_d4_fixed_precision | 1.425× | 1.001× | 1.131× |
| f64_d4_reversible | 1.129× | 0.951× | 1.013× |
| i32_d1_fixed_precision | 1.013× | 0.968× | 1.070× |
| i32_d1_reversible | 1.010× | 1.005× | 1.055× |
| i32_d2_fixed_precision | 1.181× | 0.980× | 1.088× |
| i32_d2_reversible | 1.037× | 1.004× | 1.069× |
| i32_d3_fixed_precision | 1.628× | 1.322× | 1.243× |
| i32_d3_reversible | 1.431× | 1.053× | 1.168× |
| i32_d4_fixed_precision | 1.619× | 1.265× | 1.277× |
| i32_d4_reversible | 1.487× | 1.127× | 1.133× |
| i64_d1_fixed_precision | 0.918× | 0.999× | 1.044× |
| i64_d1_reversible | 0.955× | 0.978× | 1.035× |
| i64_d2_fixed_precision | 1.331× | 1.093× | 1.039× |
| i64_d2_reversible | 1.003× | 0.982× | 1.069× |
| i64_d3_fixed_precision | 1.571× | 1.033× | 1.118× |
| i64_d3_reversible | 1.307× | 1.027× | 1.077× |
| i64_d4_fixed_precision | 1.372× | 1.048× | 1.606× |
| i64_d4_reversible | 1.433× | 0.999× | 1.035× |

## Larger fields

Four selected cases use 16,777,216 elements. These are a subset, so their averages should not be compared with the 40-case averages as a general scaling law.

| Threads | Geometric mean speedup | Median | Range |
|---|---:|---:|---:|
| 2 | 1.461× | 1.486× | 1.234–1.695× |
| 4 | 1.103× | 1.068× | 0.955–1.363× |
| 8 | 1.043× | 1.030× | 0.996–1.121× |

| Case | 2 threads | 4 threads | 8 threads |
|---|---:|---:|---:|
| f32_d3_fixed_precision | 1.317× | 0.992× | 1.009× |
| f64_d4_fixed_precision | 1.655× | 1.144× | 1.052× |
| f64_d4_reversible | 1.234× | 0.955× | 0.996× |
| i64_d3_fixed_precision | 1.695× | 1.363× | 1.121× |

## Serial and fixed-rate controls

Controls compare the same two library builds on paths whose algorithms were not changed. Retired instruction counts are the primary regression check; elapsed time alone is sensitive to scheduling and code layout.

| Control | Geometric mean instructions: new / old | Largest absolute deviation |
|---|---:|---:|
| Serial, all 56 cases | 0.999973 | 0.205% |
| Fixed-rate Rayon, 16 cases at four threads | 1.000023 | 0.025% |

The per-case counter and timing results are included in the CSV. No control exceeded the 1% instruction-count investigation threshold.

The initial f64 1-D fixed-rate control showed 0.813× elapsed throughput with a 29.6% candidate sample spread, despite essentially unchanged cycles and instructions. A separate longer interleaved check measured 1.077×, again with unchanged CPU work. Both observations are retained (`rate-controls` and `rate-recheck` in the CSV); the initial result was not replaced. This variability is why elapsed control timings alone are not treated as algorithm regressions.

## Buffer-recycling investigation

The first production attempt used a stack of free buffers, immediately reusing whichever buffer a worker had just returned. It produced a substantial i64 1-D precision regression. The hot plane-reader function had identical normalized instructions to the old build; adding cache-line padding to the recycling metadata did not resolve the problem.

Changing recycling to FIFO improved that case by about 34–39% relative to the first attempt across 2/4/8 threads. The exact hardware cause was not isolated. The final implementation uses FIFO, and a regression test preserves that order. All tables in this report come from a complete new sweep of the final implementation; earlier attempts were not mixed into these results.

## Measurement method

- Both libraries were built with the same standalone harness source, dependency versions and `rayon,ffi` features, using default release optimisation without `target-cpu=native`.
- Cases use the existing api_compare sample generators, precision 16, accuracy 2^-8 for floats, and reversible mode. Shapes are `[1048576]`, `[1024,1024]`, `[128,128,64]` and `[64,64,16,16]`. Large cases multiply the first dimension by 16 and generate all elements with the same sample function. Fixed-rate controls use rate 8.
- The process starts pinned to CPU 3. Reused pool workers are pinned to CPUs 3,1,0,2,4,5,6,7. Two/four threads use P-cores; eight includes four E-cores. Thread counts include the reader. Its assignment within the pool is controlled by Rayon.
- Interleave baseline/candidate, rotating order, for three repetitions. Report each implementation's best elapsed sample. Calibration targets 100 ms, with 10–200 decodes per sample; large cases target 150 ms with 3–200 decodes. Median candidate timing spreads were 1.5% / 1.9% / 3.4% at 2/4/8 threads. Small differences need caution.
- Setup compresses only the selected field, creates a serial reference, constructs the pool and performs one warm-up decode. Those operations are excluded from elapsed throughput. Per-call plane-buffer allocation is included. Output bytes and final cursor are checked against serial after every benchmark process; reversible output is also checked against input.
- Every timed process is paired with a zero-iteration process. `perf stat --no-scale -x,` records user cycles and instructions on both `cpu_core` and `cpu_atom`; subtract setup and divide by iterations. Sum P/E counters and take medians to compare CPU work. Individual PMU deltas can be negative if startup work moves between core types. Work counts are not elapsed latency.

## Verification

The final FIFO implementation passed workspace tests, the Rayon/ffi/internals tests, Clippy with warnings denied and all-feature documentation checks. Builds without Rayon were also checked during implementation. After the session interruption, the final library, pipeline and truncation tests, Clippy and documentation checks were rerun successfully.

Coverage includes all scalar types and dimensions, variable-rate modes, rounding settings, special reversible floats, partial blocks, nonzero stream offsets, negative and aliased strides, truncated streams, many buffer turnovers and nested calls. Dedicated tests exercise a saturated one-buffer pool, FIFO rotation, lease return during unwinding, poisoned-lock recovery and injected allocation failure before stream/output mutation, followed by serial fallback.

The temporary first-sweep files were cleared between sessions. The final measurement and check logs are retained under `target/queue-production/`, which also contains build metadata and SHA-256 hashes of the measured source and binaries.

## Reproduce

The recorded run compared against a build of the earlier two-buffer implementation using a comparison runner, and neither is kept in the repository.
In the raw data, the `double` label identifies the old binary and `queue` the new one.
The `job_kib` and `depth` columns describe the candidate configuration, and the baseline uses its original two 256 KiB batches.
The `scale` column is 16 for the large-field cases, which multiplied the first dimension.
To measure the current implementation against serial decoding, follow the [reproduction steps](pipeline.md#reproduce) for `scripts/bench_pipeline.py`.

## Data

- [Per-case results](pipeline_queue_production.csv): 205 comparisons, including timing spreads and median CPU-work ratios.
- [Raw samples](pipeline_queue_production_raw.csv): 1,230 timed samples with setup-subtracted counters and iteration counts.
- [Original two-buffer evaluation](pipeline.md).
