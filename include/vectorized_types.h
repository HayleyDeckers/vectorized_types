#ifndef VECTORIZED_TYPES_HPP
#define VECTORIZED_TYPES_HPP

//Basic definition and fallback type.
#include "vectorized_types/default.h"
#include <immintrin.h>

//pick the largest available vector register as the default.
#ifdef __AVX512F__
#include "vectorized_types/preferences/avx512.h"
#elif __AVX__
#include "vectorized_types/preferences/avx.h"
#elif __SSE__
#include "vectorized_types/preferences/sse.h"
#endif


#endif
