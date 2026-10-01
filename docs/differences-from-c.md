# Differences from the C implementation

`zfp-rs` produces the same bytes as the C library for every scalar type, dimensionality and compression mode, on little-endian platforms ([`compress_compat`](../tests/proptest/compress_compat.rs), [`decompress_compat`](../tests/proptest/decompress_compat.rs)).
This document lists the exceptions: deliberate differences, and C bugs that zfp-rs does not reproduce.
Source references are to libzfp 1.0.1, the version `zfp-sys` 0.4 builds, vendored in [`zfp/`](../zfp).
Unless stated otherwise, the examples compress the 16 `f64` values `(0.37 * i).sin() * 100.0` for `i` in `0..16`, as a 1-D field, and [`differences.rs`](../tests/proptest/differences.rs) checks them against `zfp-sys`.

## Summary

| Difference | C | zfp-rs | Same bytes? |
| --- | --- | --- | --- |
| [Reading, writing or seeking past the end of a stream](#the-ends-of-a-stream) | Unchecked access past the capacity | Reads yield zeros, and `decompress` reports it; writes are dropped and reported | Yes, for well-formed streams |
| [Bit counts above 64](#bit-counts-above-64) | Undefined | 64 bits | n/a |
| [Bit values above 1](#bit-values-above-1) | Added whole | Low bit written | **No** |
| [Invalid fields and buffers](#field-validation) | Trusted | Rejected with an error | Yes |
| [Rounding mode](#rounding) | Build-time option | Per-call `ZfpRounding` | Yes, with matching settings |
| [Parallel execution](#parallel-execution) | OpenMP compression only | Rayon compression and decompression (variable-rate streams use a pipeline) | Yes |
| [Out-of-range float-to-integer conversion](#platform-independent-output) | Undefined | x86-64 result everywhere | Yes, against x86-64 builds |
| [Fixed rates C cannot convert](#fixed-rates-c-cannot-convert) | Undefined, or wraps to zero | Rejected | n/a |
| [Build configurations](#unsupported-build-configurations) | Big-endian, strided streams, DAZ, CUDA | Default build only | n/a |
| [`max_bits` below the block header](#max_bits-below-the-block-header) | Budget wraps around | Budget is zero | **No** |
| [`maximum_size` of those configurations](#maximum_size-under-reports) | Too small | Fits the output | n/a |
| [Reversible all-zero blocks with `min_bits`](#reversible-all-zero-blocks-are-not-padded) | Not padded; the stream does not decode | Padded | **No** |
| [Blocks of tiny magnitude](#blocks-of-tiny-magnitude-lose-their-values) | Values destroyed | Kept; C decodes them | **No** |
| [Reversible mode with `ROUND_LAST`](#reversible-mode-is-lossy-when-rounding-last) | Lossy | Lossless | Yes; decoded values differ |

## Deliberate differences

### The ends of a stream

C accesses the stream buffer without bounds checks ([`bitstream.inl`](../zfp/include/zfp/bitstream.inl) lines 149 and 161), so decoding a truncated stream, or encoding into an undersized buffer, accesses memory past the capacity given to `stream_open`.
That is undefined behaviour unless the allocation extends further, and even then C's result depends on memory outside the stream.
zfp-rs instead:

- reads zeros past the end ([`when_seek_read_past_end_expect_offset_kept_and_zeros_read`](../src/bitstream.rs)), and `decompress` returns `ZfpDecompressionError::Truncated` when decoding loads a word the buffer does not hold, where C would read past the capacity ([`when_read_loads_word_past_end_expect_overread`](../src/bitstream.rs), [`when_bit_reader_consumes_past_end_expect_overread_but_not_when_peeking`](../src/bitstream.rs)), serially and with Rayon ([`truncated_stream_is_reported_serially_and_in_parallel`](../src/decompress.rs), [`missing_words_are_reported`](../tests/truncated_stream.rs)), never for a whole stream ([`whole_streams_are_not_truncated`](../tests/truncated_stream.rs));
- does not report a stream missing only padding that a block skips to a word boundary, since C reads nothing past the cut either, and returns the whole stream's size, which exceeds the capacity, as C does ([`missing_padding_is_not_truncated`](../tests/truncated_stream.rs), [`zfp_decompress_matches_c_on_a_stream_missing_only_padding`](../zfp-rs-ffi/tests/ffi_compat.rs));
- counts whole words, so a stream cut inside its last word decodes as if missing that word if the buffer excludes it, as `ZfpBitStreamRef::from_bytes` does, and as whole if the buffer zero-pads it, as `ZfpBitStream::from_bytes` does ([`a_cut_inside_the_last_word_is_seen_only_without_padding`](../tests/truncated_stream.rs));
- makes the C ABI's `zfp_decompress` return 0 for a truncated stream, as for any failure ([`zfp_decompress_returns_zero_for_a_truncated_stream`](../zfp-rs-ffi/tests/ffi_compat.rs)), while the C ABI's block decoders, which have no way to report it, read zeros;
- drops writes past the end and sets `overflowed()`, and `compress` and `write_header` return `ZfpCompressionError::BufferTooSmall` ([`given_undersized_stream_when_compress_expect_buffer_too_small_not_panic`](../src/bitstream.rs), [`given_undersized_stream_when_write_header_expect_buffer_too_small_and_nothing_written`](../src/bitstream.rs));
- drops the words that `pad` and `copy_from` would write past the end all at once, leaving the stream as writing them one at a time would ([`when_pad_past_end_expect_same_state_as_writing_word_by_word`](../src/bitstream.rs), [`when_copy_past_end_expect_same_state_as_copying_word_by_word`](../src/bitstream.rs)), so a huge count returns promptly ([`when_pad_huge_expect_prompt_return_and_overflow`](../src/bitstream.rs), [`when_copy_huge_expect_prompt_return_and_overflow`](../src/bitstream.rs)), where C's `stream_pad` and `stream_copy` (lines 381 and 413) write every word;
- keeps a seek offset past the end, as C does ([`seek_past_end_compat`](../tests/proptest/bitstream_compat.rs)), but then reads zeros and drops writes ([`when_seek_read_past_end_expect_offset_kept_and_zeros_read`](../src/bitstream.rs), [`when_seek_write_past_end_expect_offset_kept_and_writes_dropped`](../src/bitstream.rs)), where C's `stream_rseek` and `stream_wseek` (lines 340 and 356) load the out-of-bounds word under an offset that is not word-aligned.

Within the buffer, both give the same bytes and positions ([`bitstream_compat`](../tests/proptest/bitstream_compat.rs)).

### Bit counts above 64

C's `stream_read_bits` and `stream_write_bits` take at most 64 bits ([`bitstream.inl`](../zfp/include/zfp/bitstream.inl) lines 252 and 287), and shift out of range for a larger count, which is undefined behaviour.
zfp-rs reads or writes 64 bits for any larger count ([`when_bit_count_above_64_expect_64_bits`](../src/bitstream.rs)), as does the C ABI ([`stream_bit_counts_above_64_read_and_write_64_bits`](../zfp-rs-ffi/tests/ffi_compat.rs)).

### Bit values above 1

C's `stream_write_bit` documents its argument as a bit that "must be 0 or 1" ([`bitstream.inl`](../zfp/include/zfp/bitstream.inl) line 239), but adds any other value to its buffer whole, setting bits past the cursor that later reads and writes then carry into.
For 0 and 1, the C ABI writes the same bits as C, and for any other value it writes the value's low bit ([`stream_write_bit_matches_c_for_bits_and_writes_the_low_bit_otherwise`](../zfp-rs-ffi/tests/ffi_compat.rs)).
It returns the value, as C does.
The Rust API cannot express such a value, as `write_bit` takes a `bool`.

### Field validation

C trusts a field's pointer, dimensions and strides.
zfp-rs rejects, when a field is constructed, malformed dimensions ([`constructors_reject_malformed_dims`](../src/field.rs)) and buffers too short for the strides ([`strides_that_overrun_the_buffer_are_rejected`](../src/field.rs)) or misaligned for the scalar type ([`constructors_reject_misaligned_data`](../src/field.rs)).
The C ABI in `zfp-rs-ffi` checks the buffer's length and alignment when a field is compressed or decompressed instead ([`codec_rejects_an_unchecked_field_larger_than_its_buffer`](../src/field.rs), [`codec_rejects_an_unchecked_misaligned_field`](../src/field.rs)).

### Rounding

`ZFP_ROUNDING_MODE` and `ZFP_WITH_TIGHT_ERROR` are build options in C, but `ZfpConfig::with_rounding` sets them per call in zfp-rs.
The stream does not record them, so the decoder must use the encoder's setting ([`header_config_needs_the_encoders_rounding`](../tests/proptest/rounding.rs)).
The C ABI fixes them at build time, as C does, through the `round-tight-error` feature.
zfp-rs matches C built with `ZFP_ROUND_FIRST` and `ZFP_WITH_TIGHT_ERROR` ([`c_rounding.rs`](../zfp-round-tests/tests/c_rounding.rs), and [`ffi_compat.rs`](../zfp-rs-ffi/tests/ffi_compat.rs) with `round-tight-error`), and with `ZFP_ROUND_FIRST` alone or `ZFP_ROUND_LAST` with or without `ZFP_WITH_TIGHT_ERROR` ([`*_matches_each_rounding_build`](../zfp-rs-ffi/tests/c_rounding_builds.rs)), except in [reversible mode](#reversible-mode-is-lossy-when-rounding-last).

### Parallel execution

With the `rayon` feature, compression splits blocks into the same chunks as C's OpenMP code, and concatenates them at bit granularity, so it matches serial compression ([`compress_compat`](../tests/proptest/compress_compat.rs)).
Fixed-rate streams also decompress in parallel ([`decompress_compat`](../tests/proptest/decompress_compat.rs), [`given_fixed_rate_stream_when_rayon_decompress_expect_serial_cursor_and_size`](../src/bitstream.rs)), which C does not do: the OpenMP entries of its dispatch table are empty ([`zfp.c`](../zfp/src/zfp.c) line 1182).
Variable-rate streams use a serial plane reader feeding parallel reconstruction workers through a bounded queue of reusable buffers, with no index or format change ([`pipeline_matches_serial_for_types_dimensions_modes_and_rounding`](../tests/pipeline.rs), [`queued_buffers_turn_over_for_all_types_and_dimensions`](../tests/pipeline.rs)).
The C ABI's `zfp_stream_set_omp_threads` and `zfp_stream_set_omp_chunk_size` set the same `threads` and `chunk_size`, so for these streams the chunk size is the blocks per batch, and no setting changes the output ([`zfp_decompress_with_omp_settings_matches_serial_for_variable_rate_streams`](../zfp-rs-ffi/tests/ffi_compat.rs)).
Fields whose strides may alias decompress serially ([`aliasing_strides_decompress_serially`](../src/decompress.rs) for fixed-rate streams, [`pipeline_matches_serial_for_types_dimensions_modes_and_rounding`](../tests/pipeline.rs) for the rest).
So does a variable-rate stream decoded with fewer than two pool threads ([`pipeline_reuses_current_pool_and_nested_calls_complete`](../tests/pipeline.rs)).

### Platform-independent output

C casts `s * x` to an integer ([`encodef.c`](../zfp/src/template/encodef.c) line 57), which is undefined behaviour when the product is out of range, infinite or NaN.
This happens for blocks with infinities or NaNs, which C's lossy modes do not support, and in C for [blocks of tiny magnitude](#blocks-of-tiny-magnitude-lose-their-values).
C's `frexp` also leaves the exponent of infinity unspecified.
zfp-rs always gives the x86-64 glibc results: the minimum integer ([`truncate_f64_gives_i64_min_out_of_range`](../src/codec/encode/core.rs)), and an exponent of zero ([`exponent_block_matches_frexp`](../src/codec/encode/core.rs)).

### Fixed rates C cannot convert

C converts a rate to a block budget with `(uint)floor(n * rate + 0.5)` ([`zfp.c`](../zfp/src/zfp.c) line 806), which is undefined behaviour for a NaN rate, or one that rounds outside `0..=UINT_MAX` bits per block.
It rounds a word-aligned budget up in `uint` arithmetic (line 821), which wraps a budget above `UINT_MAX - 63` around to zero.
For these rates, the C ABI's `zfp_stream_set_rate` returns 0 and leaves the stream unchanged ([`invalid_ffi_rates_return_zero_without_changing_stream`](../zfp-rs-ffi/tests/rate_boundaries.rs)).
For every other rate it sets the same budget as C and returns the same rate, including rates that round to no bits and budgets above `ZFP_MAX_BITS` ([`set_rate_matches_c_where_c_is_defined`](../zfp-rs-ffi/tests/ffi_compat.rs)), which compress as in C ([`set_rate_budgets_outside_fixed_rate_compress_as_c`](../zfp-rs-ffi/tests/ffi_compat.rs)).
`ZfpConfig::fixed_rate` also rejects negative rates, and budgets of zero bits or above `ZFP_MAX_BITS` ([`fixed_rate_rejects_invalid_and_unrepresentable_rates`](../src/config.rs)).
Like C, it raises a zero rate to the block header for float types ([`fixed_rate_raises_a_zero_rate_to_the_float_header`](../src/config.rs)).

### Unsupported build configurations

zfp-rs matches a default build of libzfp, and does not support big-endian platforms, which C supports with `BIT_STREAM_WORD_TYPE=uint8`, strided bit streams (`BIT_STREAM_STRIDED`), `ZFP_WITH_DAZ`, the CUDA backend, or the C++ compressed arrays and their C bindings (cfp).

## C bugs that zfp-rs does not reproduce

### `max_bits` below the block header

A float block starts with a header of `1 + EBITS` bits, 9 for `f32` and 12 for `f64`, and C gives its coefficients a budget of `zfp->maxbits - bits` ([`encodef.c`](../zfp/src/template/encodef.c) line 79, [`decodef.c`](../zfp/src/template/decodef.c) line 20).
These are unsigned, so if `max_bits` is below the header, the budget wraps around to about four billion bits.
Reversible mode subtracts its header of up to `2 + EBITS` bits the same way, and then the `PBITS` bits that record the block's precision, 5 for 32-bit and 6 for 64-bit types ([`revencodef.c`](../zfp/src/template/revencodef.c) line 78, [`revdecodef.c`](../zfp/src/template/revdecodef.c) lines 33 and 43, [`revencode.c`](../zfp/src/template/revencode.c) line 72, [`revdecode.c`](../zfp/src/template/revdecode.c) line 44).
zfp-rs gives such a block no budget, and writes only the header ([`max_bits_below_the_header`](../tests/proptest/differences.rs)):

| Config (`min_bits`, `max_bits`, `max_prec`, `min_exp`) | C | zfp-rs |
| --- | --- | --- |
| `expert(1, 11, 64, -1074)` | 136 bytes | 8 bytes |
| `expert(12, 12, 64, -1074)` | 8 bytes | 8 bytes, identical |
| `expert(1, 18, 64, -1075)`, reversible | 120 bytes | 16 bytes |
| `expert(19, 19, 64, -1075)`, reversible | 16 bytes | 16 bytes, identical |

The streams match once `max_bits` reaches the largest header: 9 bits for `f32` and 12 for `f64`, or in reversible mode 5 for `i32`, 6 for `i64`, 15 for `f32` and 19 for `f64` ([`*_matches_c_from_the_header`](../tests/proptest/differences.rs)).
Only expert configurations go lower: `ZfpConfig::fixed_rate`, like `zfp_stream_set_rate`, raises `max_bits` to the header for float types ([`fixed_rate_min_bits_enforcement`](../src/config.rs), [`compress_compat`](../tests/proptest/compress_compat.rs)).

### `maximum_size` under-reports

C's `zfp_stream_maximum_size` ([`zfp.c`](../zfp/src/zfp.c) line 744) bounds each block by `max_bits`, so for the configurations above its size is too small, as are C's OpenMP chunk buffers, which it sizes ([`parallel.c`](../zfp/src/share/parallel.c) line 44).
zfp-rs never bounds a block below its header, so `ZfpConfig::maximum_size` and each Rayon chunk's buffer hold its output ([`*_fits_maximum_size`](../tests/proptest/budget.rs)), as does the C ABI's `zfp_stream_maximum_size` ([`stream_maximum_size_holds_blocks_longer_than_max_bits`](../zfp-rs-ffi/tests/ffi_compat.rs)).
For `expert(1, 11, 64, -1074)` ([`maximum_size_under_reports`](../tests/proptest/differences.rs)):

| | Maximum size | Output |
| --- | --- | --- |
| C | 24 bytes | 136 bytes |
| zfp-rs | 32 bytes | 8 bytes |

### Reversible all-zero blocks are not padded

C's reversible float encoder codes an all-zero block as one zero bit, without padding it to `min_bits` ([`revencodef.c`](../zfp/src/template/revencodef.c) lines 64–69), but its decoder skips to `min_bits` after that bit ([`revdecodef.c`](../zfp/src/template/revdecodef.c) lines 53–56).
So if `min_bits` is above 1, C cannot decode its own stream.
zfp-rs pads the block, so its streams round-trip ([`*_reversible_min_bits_round_trips`](../tests/proptest/budget.rs)), and decodes C's streams as C does when it is given the zeros C reads past their ends ([`decompress_compat`](../tests/proptest/decompress_compat.rs)).
Without them, decoding C's stream can run off its end, which zfp-rs reports as [truncated](#the-ends-of-a-stream) ([`reversible_zero_blocks_are_padded`](../tests/proptest/differences.rs)).
With the first four values set to zero ([`reversible_zero_blocks_are_padded`](../tests/proptest/differences.rs)):

| Config | C | zfp-rs |
| --- | --- | --- |
| `expert(100, 16658, 64, -1075)` | 96 bytes, which do not decode | 104 bytes, lossless |
| `expert(1, 16658, 64, -1075)` | 96 bytes, lossless | 96 bytes, identical |

### Blocks of tiny magnitude lose their values

To convert a float block to integers, C scales it by `2^(30 - emax)` for `f32`, or `2^(62 - emax)` for `f64`, where `emax` is the exponent of the block's largest magnitude, and computes the scale in the block's own type ([`encodef.c`](../zfp/src/template/encodef.c) lines 44–57).
If the largest magnitude is below 2^-98 for `f32` (about 3.2e-30), or 2^-962 for `f64`, the scale overflows, and the conversion is [undefined](#platform-independent-output), so C loses the block's values ([`tiny_block_thresholds`](../tests/proptest/tiny_blocks.rs), [`c_loses_tiny_blocks`](../tests/proptest/tiny_blocks.rs)).
The C source acknowledges the overflow ([`encodef.c`](../zfp/src/template/encodef.c) line 22, zfp issue #119), but its remedy, `ZFP_WITH_DAZ`, only helps blocks of subnormals, and `f32` values from 2^-126 to 2^-98 are normal.

zfp-rs scales such blocks in two exact steps, so they get the integers C would give them if its scale did not overflow ([`fwd_cast_scales_exactly`](../src/codec/encode/core.rs)).
The stream format is unchanged, so C decodes zfp-rs's streams ([`c_decodes_zfp_rs_tiny_f32_blocks`](../tests/proptest/tiny_blocks.rs), [`c_decodes_zfp_rs_tiny_f64_blocks`](../tests/proptest/tiny_blocks.rs)), unless the largest magnitude is below 2^-120 for `f32` or 2^-1013 for `f64`, where C's decoder's own scale underflows and it decodes zeros ([`tiny_block_thresholds`](../tests/proptest/tiny_blocks.rs), [`underflowing_blocks_decode_to_zeros`](../tests/proptest/tiny_blocks.rs)).
zfp-rs's decoder is unchanged, and decodes every stream as C does ([`decompress_compat`](../tests/proptest/decompress_compat.rs)).
Reversible mode is unaffected: C codes such a block's raw bits, as its conversion does not round-trip, and so does zfp-rs ([`reversible_tiny_blocks_match_c`](../tests/proptest/tiny_blocks.rs)).

Only blocks below the threshold differ.
In fixed-rate mode they keep their size; in other modes their size can change, which moves the blocks after them.
For the `f32` values `(i - 7.5) * 1e-31` for `i` in `0..16`, and the 16 `f64` values times 1e-300, with each stream decoded by C ([`tiny_blocks_lose_their_values`](../tests/proptest/differences.rs)):

| Data, config | C | zfp-rs |
| --- | --- | --- |
| `f32`, `fixed_precision(16)` | 40 bytes, maximum error 2.3e-30 | 32 bytes, maximum error 3.1e-35 |
| `f32`, `fixed_rate(8)` | 16 bytes, maximum error 2.3e-30 | 16 bytes, maximum error 6.1e-33 |
| `f64`, `fixed_precision(16)` | 40 bytes, maximum error 4.8e-298 | 40 bytes, maximum error 1.1e-302 |

### Reversible mode is lossy when rounding last

A C build with `ZFP_ROUNDING_MODE=ZFP_ROUND_LAST` biases the coefficients below the last bit plane it decodes ([`decode.c`](../zfp/src/template/decode.c) lines 124–127, 177–180, 214–217 and 250–253), in the bit-plane decoder that the reversible decoder shares ([`revdecode.c`](../zfp/src/template/revdecode.c) line 44).
A reversible block's lower bit planes are exact, so the bias makes reversible mode lossy: the `i32` values `0, 8, 16, 24, 32, 40, ...` decode as `1, 10, 20, 32, 33, 42, ...` ([`c_reversible_decoding_is_lossy_when_rounding_last`](../zfp-rs-ffi/tests/c_rounding_builds.rs)).
This looks like an oversight, as zfp's documentation describes reversible mode as "a bit-for-bit identical reconstruction".
zfp-rs does not bias reversible decoding, so reversible mode is lossless under every `ZfpRounding` ([`reversible_is_lossless_under_every_rounding`](../tests/proptest/rounding.rs)), and writes the same streams as C ([`c_reversible_decoding_is_lossy_when_rounding_last`](../zfp-rs-ffi/tests/c_rounding_builds.rs)).
