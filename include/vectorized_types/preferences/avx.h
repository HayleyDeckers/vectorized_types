#ifndef VECT_PREF_AVX_H
#define VECT_PREF_AVX_H
#include "../default.h"
#include "mathfun/avx_mathfun.h"
#include <cstdint>
/// workaround for g++ lack of intrinsics.
/// source: https://stackoverflow.com/a/32630658/5133184
#define _mm256_set_m128_ps(v0, v1)  _mm256_insertf128_ps(_mm256_castps128_ps256(v1), (v0), 1)

namespace vec{
template<>
struct preffered_vector_type<int8_t>{
  constexpr static int width = 32;
  using type = int8_t  __attribute__ ((vector_size (32)));
};

template<>
struct preffered_vector_type<uint8_t>{
  constexpr static int width = 32;
  using type = uint8_t  __attribute__ ((vector_size (32)));
};

template<>
struct preffered_vector_type<int32_t>{
  constexpr static int width = 8;
  using type = int32_t  __attribute__ ((vector_size (32)));
};

template<>
struct preffered_vector_type<uint32_t>{
  constexpr static int width = 8;
  using type = uint32_t  __attribute__ ((vector_size (32)));
};

template<>
struct preffered_vector_type<float>{
  constexpr static int width = 8;
  using type = float  __attribute__ ((vector_size (32)));
};

template<>
struct preffered_vector_type<int64_t>{
  constexpr static int width = 4;
  using type = int64_t  __attribute__ ((vector_size (32)));
};

template<>
struct preffered_vector_type<uint64_t>{
  constexpr static int width = 4;
  using type = uint64_t  __attribute__ ((vector_size (32)));
};

template<>
struct preffered_vector_type<double>{
  constexpr static int width = 4;
  using type = double  __attribute__ ((vector_size (32)));
};

template<>
inline vectorized_type<float>::vectorized_type(const float* val) : mVal(_mm256_loadu_ps(val)){
}
template<>
inline vectorized_type<double>::vectorized_type(const double* val) : mVal(_mm256_loadu_pd(val)){
}

template<>
inline void vectorized_type<float>::set_1(float val){
  mVal = _mm256_set1_ps(val);
}
template<>
inline void vectorized_type<double>::set_1(double val){
  mVal = _mm256_set1_pd(val);
}

template<>
inline vectorized_type<float> vectorized_type<float>::abs() const{
  static const __m256 SIGNMASK = (__m256)_mm256_set1_epi32(0x80000000);
  return _mm256_andnot_ps(SIGNMASK, mVal); // absval = abs(val)
}
template<>
inline vectorized_type<double> vectorized_type<double>::abs() const{
  static const __m256d SIGNMASK = (__m256d)_mm256_set1_epi64x(1ul<<63);
  return _mm256_andnot_pd(SIGNMASK, mVal); // absval = abs(val)
}

template<>
inline vectorized_type<float> vectorized_type<float>::sqrt() const{
  return _mm256_sqrt_ps(mVal);
}
template<>
inline vectorized_type<double> vectorized_type<double>::sqrt() const{
  return _mm256_sqrt_pd(mVal);
}

template<>
inline vectorized_type<float> vectorized_type<float>::sin() const{
  return internal::avx::sin256_ps(mVal);
}
template<>
inline vectorized_type<float> vectorized_type<float>::cos() const{
  return internal::avx::cos256_ps(mVal);
}
template<>
inline vectorized_type<float> vectorized_type<float>::log() const{
  return internal::avx::log256_ps(mVal);
}
template<>
inline vectorized_type<float> vectorized_type<float>::exp() const{
  return internal::avx::exp256_ps(mVal);
}

#ifdef __AVX2__

template<> template<>
inline vectorized_type<float> vectorized_type<float>::gather<int32_t>(float const* data, const int32_t indices[8]){
  auto vindex = _mm256_loadu_si256((const __m256i*)indices);
  return _mm256_i32gather_ps(data, vindex, 4);
}
template<> template<>
inline vectorized_type<float> vectorized_type<float>::gather_stride<int32_t>(float const* data, const int32_t indices[8], int32_t stride){
  auto vindex = _mm256_loadu_si256((const __m256i*)indices);
  vindex = _mm256_mul_epi32(vindex,_mm256_set1_epi32(stride));
  return _mm256_i32gather_ps(data, vindex, 4);
}

template<> template<>
inline vectorized_type<float>  vectorized_type<float>::gather<int64_t>(float const* data, const int64_t indices[8]){
  auto vindex_1 = _mm256_loadu_si256((const __m256i*)indices);
  auto vindex_2 = _mm256_loadu_si256((const __m256i*)(indices+4));
  auto low = _mm256_i64gather_ps(data, vindex_1, 4);
  auto high = _mm256_i64gather_ps(data, vindex_2, 4);
  return _mm256_set_m128_ps(high, low);
}

template<> template<>
inline vectorized_type<double>  vectorized_type<double>::gather<int32_t>(double const* data, const int32_t indices[4]){
  auto vindex = _mm_loadu_si128((const __m128i*)indices);
  return _mm256_i32gather_pd(data, vindex, 8);
}
template<> template<>
inline vectorized_type<double>  vectorized_type<double>::gather<int64_t>(double const* data, const int64_t indices[4]){
  auto vindex = _mm256_loadu_si256((const __m256i*)indices);
  return _mm256_i64gather_pd(data, vindex, 8);
}
#endif
}

#endif
