#ifndef VECT_PREF_MSVC_SSE_H
#define VECT_PREF_MSVC_SSE_H
#include "../default.h"
#include "mathfun/sse_mathfun.h"
#include <xmmintrin.h>
#include <emmintrin.h>

namespace vec {
namespace msvc_detail {

struct sse_f32x4 {
  __m128 v;
  sse_f32x4() = default;
  sse_f32x4(__m128 val) : v(val) {}
  operator __m128() const { return v; }
  float operator[](size_t i) const { return v.m128_f32[i]; }
  float& operator[](size_t i) { return v.m128_f32[i]; }
  friend sse_f32x4 operator+(sse_f32x4 a, sse_f32x4 b) { return _mm_add_ps(a.v, b.v); }
  friend sse_f32x4 operator-(sse_f32x4 a, sse_f32x4 b) { return _mm_sub_ps(a.v, b.v); }
  friend sse_f32x4 operator*(sse_f32x4 a, sse_f32x4 b) { return _mm_mul_ps(a.v, b.v); }
  friend sse_f32x4 operator/(sse_f32x4 a, sse_f32x4 b) { return _mm_div_ps(a.v, b.v); }
};

struct sse_f64x2 {
  __m128d v;
  sse_f64x2() = default;
  sse_f64x2(__m128d val) : v(val) {}
  operator __m128d() const { return v; }
  double operator[](size_t i) const { return v.m128d_f64[i]; }
  double& operator[](size_t i) { return v.m128d_f64[i]; }
  friend sse_f64x2 operator+(sse_f64x2 a, sse_f64x2 b) { return _mm_add_pd(a.v, b.v); }
  friend sse_f64x2 operator-(sse_f64x2 a, sse_f64x2 b) { return _mm_sub_pd(a.v, b.v); }
  friend sse_f64x2 operator*(sse_f64x2 a, sse_f64x2 b) { return _mm_mul_pd(a.v, b.v); }
  friend sse_f64x2 operator/(sse_f64x2 a, sse_f64x2 b) { return _mm_div_pd(a.v, b.v); }
};

} // namespace msvc_detail

template<>
struct preffered_vector_type<float> {
  constexpr static int width = 4;
  using type = msvc_detail::sse_f32x4;
};

template<>
struct preffered_vector_type<double> {
  constexpr static int width = 2;
  using type = msvc_detail::sse_f64x2;
};

template<>
inline vectorized_type<float>::vectorized_type(const float* val) {
  mVal = _mm_loadu_ps(val);
}
template<>
inline vectorized_type<double>::vectorized_type(const double* val) {
  mVal = _mm_loadu_pd(val);
}

template<>
inline void vectorized_type<float>::set_1(float val) {
  mVal = _mm_set1_ps(val);
}
template<>
inline void vectorized_type<double>::set_1(double val) {
  mVal = _mm_set1_pd(val);
}

template<>
inline vectorized_type<float> vectorized_type<float>::sqrt() const {
  return _mm_sqrt_ps(mVal);
}
template<>
inline vectorized_type<double> vectorized_type<double>::sqrt() const {
  return _mm_sqrt_pd(mVal);
}

template<>
inline vectorized_type<float> vectorized_type<float>::sin() const {
  return internal::sse::sin_ps(mVal);
}
template<>
inline vectorized_type<float> vectorized_type<float>::cos() const {
  return internal::sse::cos_ps(mVal);
}
template<>
inline vectorized_type<float> vectorized_type<float>::log() const {
  return internal::sse::log_ps(mVal);
}

} // namespace vec
#endif
