#ifndef VECTORIZED_TYPES_HPP
#define VECTORIZED_TYPES_HPP

//Basic definition and fallback type.
#include "vectorized_types/default.h"
#include <immintrin.h>

//pick the largest available vector register as the default.
#ifdef _MSC_VER
  // MSVC: use wrapper-based headers (GCC vector extensions not available)
  #if defined(__AVX__)
    #include "vectorized_types/preferences/msvc_avx.h"
  #elif defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
    #include "vectorized_types/preferences/msvc_sse.h"
  #endif
#else
  // GCC/Clang: use native vector extension headers
  #ifdef __AVX512F__
    #include "vectorized_types/preferences/avx512.h"
  #elif __AVX__
    #include "vectorized_types/preferences/avx.h"
  #elif __SSE__
    #include "vectorized_types/preferences/sse.h"
  #endif
#endif


#endif
