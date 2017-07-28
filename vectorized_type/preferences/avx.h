#ifndef VECT_PREF_AVX_H
#define VECT_PREF_AVX_H

/// workaround for g++ lack of intrinsics.
/// source: https://stackoverflow.com/a/32630658/5133184
#define _mm256_set_m128_ps(v0, v1)  _mm256_insertf128_ps(_mm256_castps128_ps256(v1), (v0), 1)

template<>
struct preffered_vector_type<float>{
  constexpr static int width = 8;
  using type = float  __attribute__ ((vector_size (32)));
};

template<>
struct preffered_vector_type<double>{
  constexpr static int width = 4;
  using type = double  __attribute__ ((vector_size (32)));
};

template<>
inline void vectorized_type<float>::set_1(float val){
  mVal = _mm256_set1_ps(val);
}
template<>
inline void vectorized_type<double>::set_1(double val){
  mVal = _mm256_set1_pd(val);
}

template<>
inline vectorized_type<float> vectorized_type<float>::sqrt() const{
  return _mm256_sqrt_ps(mVal);
}
template<>
inline vectorized_type<double> vectorized_type<double>::sqrt() const{
  return _mm256_sqrt_pd(mVal);
}

#ifdef __AVX2__

template<> template<>
inline vectorized_type<float>  vectorized_type<float>::Gather<int32_t>(float const* data, int32_t indices[8]){
  auto vindex = _mm256_loadu_si256((const __m256i*)indices);
  return _mm256_i32gather_ps(data, vindex, 4);
}
template<> template<>
inline vectorized_type<float>  vectorized_type<float>::Gather<int64_t>(float const* data, int64_t indices[8]){
  auto vindex_1 = _mm256_loadu_si256((const __m256i*)indices);
  auto vindex_2 = _mm256_loadu_si256((const __m256i*)(indices+4));
  auto low = _mm256_i64gather_ps(data, vindex_1, 4);
  auto high = _mm256_i64gather_ps(data, vindex_2, 4);
  return _mm256_set_m128_ps(high, low);
}

template<> template<>
inline vectorized_type<double>  vectorized_type<double>::Gather<int32_t>(double const* data, int32_t indices[4]){
  auto vindex = _mm_loadu_si128((const __m128i*)indices);
  return _mm256_i32gather_pd(data, vindex, 8);
}
template<> template<>
inline vectorized_type<double>  vectorized_type<double>::Gather<int64_t>(double const* data, int64_t indices[4]){
  auto vindex = _mm256_loadu_si256((const __m256i*)indices);
  return _mm256_i64gather_pd(data, vindex, 8);
}

#endif

#endif
