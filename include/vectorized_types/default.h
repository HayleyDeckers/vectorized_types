#ifndef VECT_PREF_DEFAULT_H
#define VECT_PREF_DEFAULT_H

#include "array_wrapper.hpp"
#include <cmath>
#include <stdlib.h>
#include <new>

namespace vec{
  //By default our 'SIMD' version of T is just T.
template<typename T>
struct preffered_vector_type{
  constexpr static int width = 1;
  using type = array_wrapper<T>; //By wrapping it with array_wrapper it behaves like a normal vector unit.
};

template<typename T>
class vectorized_type{
  static_assert(std::is_arithmetic<T>::value, "Typevalue T should be an ahritmetic type");
  //The vectorized type, internally
  using Simd = typename preffered_vector_type<T>::type;
  Simd mVal;
public:
    //How many elements the SIMD element has
    static constexpr int Width = preffered_vector_type<T>::width;

  constexpr inline Simd inner(){return mVal;}

  inline void set_1(T val){
    for(int i = 0; i < Width; i++){
        mVal[i] = val;
    }
  }
  constexpr vectorized_type() {}
  vectorized_type(T val) {set_1(val);}
  vectorized_type(const T* val) {
    for(int i =0; i < Width; i++){
      mVal[i] = val[i];
    }
  }
  vectorized_type(Simd val) : mVal(val) {}
  template<typename I>
  constexpr static vectorized_type gather(T const* data, const I indices[Width]){
    vectorized_type ret;
    static_assert(std::is_integral<I>::value, "Integral valued indices required");
    for(int i = 0; i < Width; i++){
     ret.set(i, data[indices[i]]);
    }
    return ret;
  }
  template<typename I>
  constexpr static vectorized_type gather_stride(T const* data, const I indices[Width], I stride){
    vectorized_type ret;
    static_assert(std::is_integral<I>::value, "Integral valued indices required");
    for(int i = 0; i < Width; i++){
     ret.set(i, data[indices[i]*stride]);
    }
    return ret;
  }
  constexpr void set(int index, T val){ mVal[index] = val;}
  //array indexing
  constexpr const T operator[](std::size_t i) const {return mVal[i];}
  // T& operator[](std::size_t i){return mVal[i];}
  constexpr static void* operator new(std::size_t sz) {
   void *p = aligned_alloc(alignof(Simd), sz);
   if(!p){
     throw std::bad_alloc();
   }
   return p;
  }
  static void* operator new[](std::size_t sz) {
   return vectorized_type::operator new(sz);
  }
  void operator delete(void *p) {
   free(p);
  }
  void operator delete[](void *p) {
   vectorized_type::operator delete(p);
  }

  //increment
  vectorized_type& operator++(){
    return ((*this) += vectorized_type(1));
  }

  //decrement
  vectorized_type& operator--(){
    return ((*this) -= vectorized_type(1));
  }

  inline float sum() const{
    float sum = 0;
    for(int i = 0; i < Width; i++){
      sum += mVal[i];
    }
    return sum;
  }

  vectorized_type& operator=(const vectorized_type& rhs){
    mVal = rhs.mVal;
    return *this;
  }
  //assign add
  inline vectorized_type& operator+=(const vectorized_type& rhs){
    auto sum = (*this)+rhs;
    return (*this) = sum;
  }
  //assign subtract
  inline vectorized_type& operator-=(const vectorized_type& rhs){
    return (*this) = (*this)-rhs;
  }
  //assign mult
  inline vectorized_type& operator*=(const vectorized_type& rhs){
    return (*this) = (*this)*rhs;
  }
  //assign divide
  inline vectorized_type& operator/=(const vectorized_type& rhs){
    return (*this) = (*this)/rhs;
  }
  //addition
  // friends defined inside class body are inline and are hidden from non-ADL lookup
 inline friend vectorized_type operator+(const vectorized_type lhs,        // passing lhs by value helps optimize chained a+b+c
                    const vectorized_type rhs) // otherwise, both parameters may be const references
 {
  //  std::cout << "vectorized" << std::endl;
   return lhs.mVal + rhs.mVal;
 }

 //subtraction
inline friend vectorized_type operator-(const vectorized_type lhs, const vectorized_type rhs)
{
  return lhs.mVal - rhs.mVal;
}

//multiplication
inline friend vectorized_type operator*(const vectorized_type lhs, const vectorized_type rhs)
{
 return lhs.mVal * rhs.mVal;
}

//division
inline friend vectorized_type operator/(const vectorized_type lhs, const vectorized_type rhs)
{
  return lhs.mVal / rhs.mVal;
}

//some basic mathematical operations
//These are overwritten in the preferences files to use custom SIMD
// algorithms or instructions where applicable.
inline vectorized_type sqrt() const{
  vectorized_type ret;
  for(int i = 0; i < Width; i++){
    ret.set(i, std::sqrt(mVal[i]));
  }
  return ret;
}
inline vectorized_type log() const{
  vectorized_type ret;
  for(int i = 0; i < Width; i++){
    ret.set(i,std::log(mVal[i]));
  }
  return ret;
}
inline vectorized_type cos() const{
  vectorized_type ret;
  for(int i = 0; i < Width; i++){
    ret.set(i,std::cos(mVal[i]));
  }
  return ret;
}
inline vectorized_type sin() const{
  vectorized_type ret;
  for(int i = 0; i < Width; i++){
    ret.set(i,std::sin(mVal[i]));
  }
  return ret;
}
inline vectorized_type tan() const{
  vectorized_type ret;
  for(int i = 0; i < Width; i++){
    ret.set(i, std::tan(mVal[i]));
  }
  return ret;
}
inline vectorized_type exp() const{
  vectorized_type ret;
  for(int i = 0; i < Width; i++){
    ret.set(i, std::exp(mVal[i]));
  }
  return ret;
}
inline vectorized_type pow(const vectorized_type& power) const{
  vectorized_type ret;
  for(int i = 0; i < Width; i++){
    ret.set(i,std::pow(mVal[i], power[i]));
  }
  return ret;
}

inline vectorized_type abs() const{
  vectorized_type ret;
  for(int i = 0; i < Width; i++){
    ret.set(i,std::abs(mVal[i]));
  }
  return ret;
}

};

//basecase mathematical operators simply call std::function.
//these can be specialized to support vectorized calls.
template<typename T>
auto sqrt(vectorized_type<T> val){
  return val.sqrt();
}

template<typename T>
auto sqrt(T val){
  return std::sqrt(val);
}

template<typename T>
 auto log(vectorized_type<T> val){
  return val.log();
}
template<typename T>
 auto log(T val){
  return std::log(val);
}

template<typename T>
 auto sin(vectorized_type<T> val){
  return val.sin();
}
template<typename T>
 auto sin(T val){
  return std::sin(val);
}

template<typename T>
 auto cos(vectorized_type<T> val){
  return val.cos();
}
template<typename T>
 auto cos(T val){
  return std::cos(val);
}

template<typename T>
 auto tan(vectorized_type<T> val){
  return val.tan();
}
template<typename T>
 auto tan(T val){
  return std::tan(val);
}

template<typename T>
 auto exp(vectorized_type<T> val){
  return val.exp();
}
template<typename T>
 auto exp(T val){
  return std::exp(val);
}

template<typename T>
 auto abs(vectorized_type<T> val){
  return val.abs();
}
template<typename T>
 auto abs(T val){
  return std::abs(val);
}

template<typename T>
 auto pow(vectorized_type<T> base, vectorized_type<T> power){
  return base.pow(power);
}
template<typename T>
 auto pow(T val, T power){
  return std::pow(val, power);
}


}

#endif
