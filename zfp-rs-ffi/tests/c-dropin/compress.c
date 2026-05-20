#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <zfp.h>

#ifndef ZFP_DROPIN_BACKEND_NAME
#define ZFP_DROPIN_BACKEND_NAME "unknown"
#endif

static int fail(const char* message)
{
  fprintf(stderr, "%s\n", message);
  return 1;
}

int main(void)
{
  const size_t n = 33;
  int32_t input[33];
  int32_t output[33] = {0};

  for (size_t i = 0; i < n; i++)
    input[i] = (int32_t)(17 * (int32_t)i * (int32_t)i - 23 * (int32_t)i + 5);

  zfp_field* field = zfp_field_1d(input, zfp_type_int32, n);
  if (!field)
    return fail("zfp_field_1d returned null");

  zfp_stream* zfp = zfp_stream_open(NULL);
  if (!zfp)
    return fail("zfp_stream_open returned null");

  zfp_stream_set_reversible(zfp);

  size_t capacity = zfp_stream_maximum_size(zfp, field);
  if (capacity == 0)
    return fail("zfp_stream_maximum_size returned zero");

  void* buffer = calloc(capacity, 1);
  if (!buffer)
    return fail("calloc failed");

  bitstream* stream = stream_open(buffer, capacity);
  if (!stream)
    return fail("stream_open returned null");

  zfp_stream_set_bit_stream(zfp, stream);

  size_t compressed = zfp_compress(zfp, field);
  if (compressed == 0)
    return fail("zfp_compress returned zero");

  size_t committed = stream_size(stream);
  if (committed == 0 || committed > capacity)
    return fail("stream_size was outside the expected range");

  zfp_field_free(field);
  field = zfp_field_1d(output, zfp_type_int32, n);
  if (!field)
    return fail("zfp_field_1d for output returned null");

  zfp_stream_rewind(zfp);
  size_t decompressed = zfp_decompress(zfp, field);
  if (decompressed == 0)
    return fail("zfp_decompress returned zero");

  for (size_t i = 0; i < n; i++) {
    if (output[i] != input[i]) {
      fprintf(stderr, "round trip mismatch at %zu: got %d, expected %d\n",
              i, output[i], input[i]);
      return 1;
    }
  }

  printf("%s backend compressed %zu int32 values into %zu bytes\n",
         ZFP_DROPIN_BACKEND_NAME, n, committed);

  zfp_field_free(field);
  zfp_stream_close(zfp);
  stream_close(stream);
  free(buffer);

  return 0;
}
