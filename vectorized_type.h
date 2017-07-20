#ifndef VECTORIZED_TYPE_HPP
#define VECTORIZED_TYPE_HPP

#include "vectorized_type/default.h"
#include <immintrin.h>

namespace vec{
#ifdef __AVX512F__
#include "vectorized_type/preferences/avx512.h"
#elif __AVX__
#include "vectorized_type/preferences/avx.h"
#elif __SSE__
#include "vectorized_type/preferences/sse.h"
#endif
}

#endif
