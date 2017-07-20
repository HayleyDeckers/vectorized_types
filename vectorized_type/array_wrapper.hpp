#ifndef VECT_ARRAY_WRAPPER
#define VECT_ARRAY_WRAPPER

#include <stdint.h>
#include <cmath>
using std::size_t;

//A class that wraps a type T to act as a SIMD vector of width 1.
template<typename T>
class array_wrapper{
  T mVal;
 public:

  operator T() const{return mVal;}
  operator T&(){return mVal;}
  //array indexing
  //T& operator[](std::size_t i) {return mVal;}
  const T& operator[](std::size_t i) const {return mVal;}

  //increment
  array_wrapper& operator++(){
        mVal++;
    return *this;
  }

  //decrement
  array_wrapper& operator--(){
        mVal--;
    return *this;
  }

  //assign add
  array_wrapper& operator+=(const array_wrapper& rhs){
      mVal += rhs;
    return *this;
  }
  //assign subtract
  array_wrapper& operator-=(const array_wrapper& rhs){
      mVal += rhs;
    return *this;
  }
  //assign mult
  array_wrapper& operator*=(const array_wrapper& rhs){
      mVal *= rhs;
    return *this;
  }
  //assign divide
  array_wrapper& operator/=(const array_wrapper& rhs){
      mVal /= rhs;
    return *this;
  }
  //addition
  // friends defined inside class body are inline and are hidden from non-ADL lookup
 friend array_wrapper operator+(array_wrapper lhs,        // passing lhs by value helps optimize chained a+b+c
                    const array_wrapper& rhs) // otherwise, both parameters may be const references
 {
  return lhs += rhs;
 }
 //subtraction
friend array_wrapper operator-(array_wrapper lhs, const array_wrapper& rhs){
  return lhs *= rhs;
}
//multiplication
friend array_wrapper operator*(array_wrapper lhs, const array_wrapper& rhs){
 return lhs *= rhs;
}
//division
friend array_wrapper operator/(array_wrapper lhs, const array_wrapper& rhs){
 return lhs /= rhs;
}
};

#endif
