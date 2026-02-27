#ifndef VECT_PREF_SSE_H
#define VECT_PREF_SSE_H
#include "../default.h"
#include "mathfun/sse_mathfun.h"
#include <cstdint>
namespace vec{
  template<>
  struct preffered_vector_type<int8_t>{
    constexpr static int width = 16;
    using type = int8_t  __attribute__ ((vector_size (16)));

  };

  template<>
  struct preffered_vector_type<uint8_t>{
    constexpr static int width = 16;
    using type = uint8_t  __attribute__ ((vector_size (16)));

  };

  template<>
  struct preffered_vector_type<int32_t>{
    constexpr static int width = 4;
    using type = int32_t  __attribute__ ((vector_size (16)));

  };

  template<>
  struct preffered_vector_type<uint32_t>{
    constexpr static int width = 4;
    using type = uint32_t  __attribute__ ((vector_size (16)));

  };

  template<>
  struct preffered_vector_type<float>{
    constexpr static int width = 4;
    using type = float  __attribute__ ((vector_size (16)));

  };

  template<>
  struct preffered_vector_type<int64_t>{
    constexpr static int width = 2;
    using type = int64_t  __attribute__ ((vector_size (16)));

  };

  template<>
  struct preffered_vector_type<uint64_t>{
    constexpr static int width = 2;
    using type = uint64_t  __attribute__ ((vector_size (16)));

  };

  template<>
  struct preffered_vector_type<double>{
    constexpr static int width = 2;
    using type = double  __attribute__ ((vector_size (16)));

  };


  template<>
  inline vectorized_type<float>::vectorized_type(const float* val){
    mVal = _mm_loadu_ps(val);
  }
  template<>
  inline vectorized_type<double>::vectorized_type(const double* val){
    mVal = _mm_loadu_pd(val);
  }

  template<>
  inline void vectorized_type<float>::set_1(float val){
    mVal = _mm_set1_ps(val);
  }
  template<>
  inline void vectorized_type<double>::set_1(double val){
    mVal = _mm_set1_pd(val);
  }

  template<>
  inline vectorized_type<float> vectorized_type<float>::sqrt() const{
    return _mm_sqrt_ps(mVal);
  }
  template<>
  inline vectorized_type<double> vectorized_type<double>::sqrt() const{
    return _mm_sqrt_pd(mVal);
  }

  template<>
  inline vectorized_type<float> vectorized_type<float>::sin() const{
    return internal::sse::sin_ps(mVal);
  }
  template<>
  inline vectorized_type<float> vectorized_type<float>::cos() const{
    return internal::sse::cos_ps(mVal);
  }
  template<>
  inline vectorized_type<float> vectorized_type<float>::log() const{
    return internal::sse::log_ps(mVal);
  }
}
#endif
