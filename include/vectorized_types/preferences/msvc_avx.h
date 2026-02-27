#ifndef VECT_PREF_MSVC_AVX_H
#define VECT_PREF_MSVC_AVX_H
#include "../default.h"
#include "mathfun/avx_mathfun.h"
#include <immintrin.h>

#ifndef _mm256_set_m128_ps
#define _mm256_set_m128_ps(v0, v1) _mm256_insertf128_ps(_mm256_castps128_ps256(v1), (v0), 1)
#endif

namespace vec {
namespace msvc_detail {

struct avx_f32x8 {
  __m256 v;
  avx_f32x8() = default;
  avx_f32x8(__m256 val) : v(val) {}
  operator __m256() const { return v; }
  float operator[](size_t i) const { return v.m256_f32[i]; }
  float& operator[](size_t i) { return v.m256_f32[i]; }
  friend avx_f32x8 operator+(avx_f32x8 a, avx_f32x8 b) { return _mm256_add_ps(a.v, b.v); }
  friend avx_f32x8 operator-(avx_f32x8 a, avx_f32x8 b) { return _mm256_sub_ps(a.v, b.v); }
  friend avx_f32x8 operator*(avx_f32x8 a, avx_f32x8 b) { return _mm256_mul_ps(a.v, b.v); }
  friend avx_f32x8 operator/(avx_f32x8 a, avx_f32x8 b) { return _mm256_div_ps(a.v, b.v); }
};

struct avx_f64x4 {
  __m256d v;
  avx_f64x4() = default;
  avx_f64x4(__m256d val) : v(val) {}
  operator __m256d() const { return v; }
  double operator[](size_t i) const { return v.m256d_f64[i]; }
  double& operator[](size_t i) { return v.m256d_f64[i]; }
  friend avx_f64x4 operator+(avx_f64x4 a, avx_f64x4 b) { return _mm256_add_pd(a.v, b.v); }
  friend avx_f64x4 operator-(avx_f64x4 a, avx_f64x4 b) { return _mm256_sub_pd(a.v, b.v); }
  friend avx_f64x4 operator*(avx_f64x4 a, avx_f64x4 b) { return _mm256_mul_pd(a.v, b.v); }
  friend avx_f64x4 operator/(avx_f64x4 a, avx_f64x4 b) { return _mm256_div_pd(a.v, b.v); }
};

} // namespace msvc_detail

template<>
struct preffered_vector_type<float> {
  constexpr static int width = 8;
  using type = msvc_detail::avx_f32x8;
};

template<>
struct preffered_vector_type<double> {
  constexpr static int width = 4;
  using type = msvc_detail::avx_f64x4;
};

template<>
inline vectorized_type<float>::vectorized_type(const float* val) : mVal(_mm256_loadu_ps(val)) {
}
template<>
inline vectorized_type<double>::vectorized_type(const double* val) : mVal(_mm256_loadu_pd(val)) {
}

template<>
inline void vectorized_type<float>::set_1(float val) {
  mVal = _mm256_set1_ps(val);
}
template<>
inline void vectorized_type<double>::set_1(double val) {
  mVal = _mm256_set1_pd(val);
}

template<>
inline vectorized_type<float> vectorized_type<float>::abs() const {
  static const __m256 SIGNMASK = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
  return _mm256_and_ps(mVal, SIGNMASK);
}
template<>
inline vectorized_type<double> vectorized_type<double>::abs() const {
  static const __m256d SIGNMASK = _mm256_castsi256_pd(_mm256_set1_epi64x(0x7FFFFFFFFFFFFFFFLL));
  return _mm256_and_pd(mVal, SIGNMASK);
}

template<>
inline vectorized_type<float> vectorized_type<float>::sqrt() const {
  return _mm256_sqrt_ps(mVal);
}
template<>
inline vectorized_type<double> vectorized_type<double>::sqrt() const {
  return _mm256_sqrt_pd(mVal);
}

template<>
inline vectorized_type<float> vectorized_type<float>::sin() const {
  return internal::avx::sin256_ps(mVal);
}
template<>
inline vectorized_type<float> vectorized_type<float>::cos() const {
  return internal::avx::cos256_ps(mVal);
}
template<>
inline vectorized_type<float> vectorized_type<float>::log() const {
  return internal::avx::log256_ps(mVal);
}
template<>
inline vectorized_type<float> vectorized_type<float>::exp() const {
  return internal::avx::exp256_ps(mVal);
}

#ifdef __AVX2__
template<> template<>
inline vectorized_type<float> vectorized_type<float>::gather<int32_t>(float const* data, const int32_t indices[8]) {
  auto vindex = _mm256_loadu_si256((const __m256i*)indices);
  return _mm256_i32gather_ps(data, vindex, 4);
}
template<> template<>
inline vectorized_type<float> vectorized_type<float>::gather_stride<int32_t>(float const* data, const int32_t indices[8], int32_t stride) {
  auto vindex = _mm256_loadu_si256((const __m256i*)indices);
  vindex = _mm256_mullo_epi32(vindex, _mm256_set1_epi32(stride));
  return _mm256_i32gather_ps(data, vindex, 4);
}
template<> template<>
inline vectorized_type<float> vectorized_type<float>::gather<int64_t>(float const* data, const int64_t indices[8]) {
  auto vindex_1 = _mm256_loadu_si256((const __m256i*)indices);
  auto vindex_2 = _mm256_loadu_si256((const __m256i*)(indices+4));
  auto low = _mm256_i64gather_ps(data, vindex_1, 4);
  auto high = _mm256_i64gather_ps(data, vindex_2, 4);
  return _mm256_set_m128_ps(high, low);
}
template<> template<>
inline vectorized_type<double> vectorized_type<double>::gather<int32_t>(double const* data, const int32_t indices[4]) {
  auto vindex = _mm_loadu_si128((const __m128i*)indices);
  return _mm256_i32gather_pd(data, vindex, 8);
}
template<> template<>
inline vectorized_type<double> vectorized_type<double>::gather<int64_t>(double const* data, const int64_t indices[4]) {
  auto vindex = _mm256_loadu_si256((const __m256i*)indices);
  return _mm256_i64gather_pd(data, vindex, 8);
}
#endif

} // namespace vec
#endif
