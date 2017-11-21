//Compile with [clan,]g++ [-fopenmp] -march=native AoS.cxx -o bin/SoA
//and your preffered optimization levels.
// at the highest optimizations, and -ffast-math turned on, the scalar example
// will also get vectorized. However, it has been my experience that this will
// fail for more complicated versions.

#include <cstddef>
#include <cmath>
#include "../vectorized_type.h"
#include <vector>
#include <random>
#include <chrono>
#include <iostream>
#include <memory>

using namespace std;
using namespace std::chrono;
using namespace vec;
using single_t = float;
using simd_t = vectorized_type<single_t>;

struct point_simd{
  simd_t x, y, z;
  simd_t len() const{return sqrt(x*x+y*y+z*z);}

  //unfortunately, new/new[] are not guarenteed to properly align our SIMD types.
  // usually the only align up to 16 bytes at most. therfore we have to define our
  // own operators for any type that contains `vectorized_type<T>`s.
  static void* operator new(std::size_t sz) {
   void *p = aligned_alloc(alignof(simd_t), sz);
   if(!p){
     throw std::bad_alloc();
   }
   return p;
  }
  static void* operator new[](std::size_t sz) {
   return point_simd::operator new(sz);
  }
  void operator delete(void *p) {
   free(p);
  }
  void operator delete[](void *p) {
   point_simd::operator delete(p);
  }
};

struct point_single{
  single_t x, y, z;
  single_t len() const{return sqrt(x*x+y*y+z*z);}
};

//A simple function that computes the lenght of 3d vectors.
single_t vectorized_calculation(point_simd *p, int len){
  simd_t sum = 0.0;
  #pragma omp declare reduction(+:simd_t: omp_out = omp_out+omp_in) initializer(omp_priv = 0)
  #pragma omp parallel for reduction(+:sum) schedule(static)
  for(int i = 0; i < len; i++){
    sum += p[i].len();
  }
  //the .sum() function returns the sum of all the elements of the simd register.
  return sum.sum();
}

single_t scalar_calculation(point_single *p, int len){
  single_t sum = 0;
  #pragma omp parallel for reduction(+:sum) schedule(static)
  for(int i = 0; i < len; i++){
    sum += p[i].len();
  }
  return sum;
}

int main(int arg, char** argv){
  const int width = simd_t::Width;
  std::cout << "using a vector width of " << width << std::endl;
  const int n_simd_elements = 1*1000*1000;
  const int n_elements = width*n_simd_elements;

  auto points_simd = new point_simd[n_simd_elements];
  auto points_single = new point_single[n_elements];

  random_device rd;
  default_random_engine e2(rd());
  uniform_real_distribution<> dist(-3.14, 3.14);
  std::cout << "initializing data... ";
  std::flush(std::cout);
  //initialize the array with random points on a sphere of radius 1.
  for (int i = 0; i < n_elements; i++) {
    single_t theta = dist(e2);
    single_t phi = dist(e2);
    points_single[i].x = cos(theta)*sin(phi);
    points_single[i].y = sin(theta)*sin(phi);
    points_single[i].z = cos(phi);
  }
  for (int i = 0; i < n_simd_elements; i ++) {
    single_t theta = dist(e2);
    single_t phi = dist(e2);
    points_simd[i].x = cos(theta)*sin(phi);
    points_simd[i].y = sin(theta)*sin(phi);
    points_simd[i].z = cos(phi);
  }
  std::cout << "done!" <<std::endl;

  single_t sum;
  double clock_sum;
  high_resolution_clock::time_point begin, end;

  const int iterations = 10;
  clock_sum = 0;
  for (int i = 0; i < iterations; i++) {
    begin =  high_resolution_clock::now();
    sum = scalar_calculation(points_single, n_elements);
    end =  high_resolution_clock::now();
    clock_sum += duration_cast<microseconds>( end - begin ).count();
  }
  cout << "scalar\t\t" << sum << " in " << clock_sum/1e6 << "s" << "\t\t =  " <<  iterations*3*sizeof(float)*n_elements/(clock_sum) << " MB/s"s << endl;


  clock_sum = 0;
  for (int i = 0; i < iterations; i++) {
    begin =  high_resolution_clock::now();
    sum = vectorized_calculation(points_simd, n_simd_elements);
    end =  high_resolution_clock::now();
    clock_sum += duration_cast<microseconds>( end - begin ).count();
  }
  cout << "vectorized\t" << sum << " in " << clock_sum/1e6 << "s" << "\t\t =  " <<  iterations*3*sizeof(float)*n_elements/(clock_sum) << " MB/s"s << endl;

}
