#ifndef VECT_PREF_SSE_H
#define VECT_PREF_SSE_H
#include "../default.h"
namespace vec{
template<>
struct preffered_vector_type<float>{
  constexpr static int width = 4;
  using type = float  __attribute__ ((vector_size (16)));
};

template<>
struct preffered_vector_type<double>{
  constexpr static int width = 2;
  using type = double  __attribute__ ((vector_size (16)));
};

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
}
#endif
