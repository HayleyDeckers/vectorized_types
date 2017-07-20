#ifndef VECTORIZED_TYPE_HPP
#define VECTORIZED_TYPE_HPP

namespace vec{
#include "vectorized_type/default.h"
template<typename T>
constexpr inline vectorized_type<T> make_vectorized_type(T val){vectorized_type<T> ret(val); return ret;}

#include <immintrin.h>
#include "vectorized_type/default.h"
#ifdef __AVX512F__
#include "vectorized_type/preferences/avx512.h"
#elif __AVX__
#include "vectorized_type/preferences/avx.h"
#elif __SSE__
#include "vectorized_type/preferences/sse.h"
#endif
}

#endif
