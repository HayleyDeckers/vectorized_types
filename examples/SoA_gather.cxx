//Compile with [clan,]g++ [-fopenmp] -march=native SoA.cxx -o bin/SoA
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
#include <algorithm>
#include <memory>

using namespace std;
using namespace std::chrono;
using namespace vec;
using single_t = double;
using simd_t = vectorized_type<single_t>;
using index_t = int32_t;

//A simple function that computes the lenght of 3d vectors.
single_t vectorized_calculation(single_t *__restrict__ x, single_t *__restrict__ y, single_t *__restrict__ z, index_t *__restrict__ indices, int len){
  simd_t sum = 0;
  #pragma omp declare reduction(+:simd_t: omp_out = omp_out+omp_in) initializer(omp_priv = 0)
  #pragma omp parallel for reduction(+:sum) schedule(static)
  for(int i = 0; i < len; i+= simd_t::Width){
    simd_t X = simd_t::Gather(x, indices+i);
    simd_t Y = simd_t::Gather(y, indices+i);
    simd_t Z = simd_t::Gather(z, indices+i);
    sum += sqrt(X*X + Y*Y + Z*Z);
    //without -ffast-math we should write
    // sum += sqrt((x[i]*x[i])+y[i]*y[i])+z[i]*z[i]));
    //in which case the compiler can produce fma instructions instead.
  }
  //the .sum() function returns the sum of all the elements of the simd register.
  return sum.sum();
}

//A simple function that computes the lenght of 3d vectors.
single_t scalar_calculation(single_t *__restrict__ x, single_t *__restrict__ y, single_t *__restrict__ z, index_t *__restrict__ indices, int len){
  float sum = 0;
  #pragma omp parallel for reduction(+:sum) schedule(static)
  for(int i = 0; i < len; i++){
    auto X = x[indices[i]];
    auto Y = y[indices[i]];
    auto Z = z[indices[i]];
    sum += sqrt(X*X + Y*Y + Z*Z);
  }
  return sum;
}


int main(int arg, char** argv){
  const int width = simd_t::Width;
  std::cout << "using a vector width of " << width << std::endl;
  const int n_simd_elements = 10*1000*1000;
  const int n_elements = width*n_simd_elements;

  simd_t *x_simd = new simd_t[n_simd_elements];
  simd_t *y_simd = new simd_t[n_simd_elements];
  simd_t *z_simd = new simd_t[n_simd_elements];

  //simd_t is a simple type and can be directly cast to an array of its base type.
  single_t *x = (single_t*)x_simd;
  single_t *y = (single_t*)y_simd;
  single_t *z = (single_t*)z_simd;

  index_t *indices = new index_t[n_elements];

  random_device rd;
  default_random_engine e2(rd());
  uniform_real_distribution<> dist(-3.14, 3.14);
  uniform_int_distribution<int32_t> index_dist(0,128);
  //initialize the array with random points on a sphere of radius 1.
  for (int i = 0; i < n_elements; i ++) {
    single_t theta = dist(e2);
    single_t phi = dist(e2);
    x[i] = cos(theta)*sin(phi);
    y[i] = sin(theta)*sin(phi);
    z[i] = cos(phi);
    indices[i] = index_dist(e2);
  }
  // shuffle(indices, indices+n_elements, e2);
  single_t sum;
  double clock_sum;
  high_resolution_clock::time_point begin, end;

  const int iterations = 10;

    clock_sum = 0;
    for (int i = 0; i < iterations; i++) {
      begin =  high_resolution_clock::now();
      sum = vectorized_calculation(x, y, z, indices, n_elements);
      end =  high_resolution_clock::now();
      clock_sum += duration_cast<microseconds>( end - begin ).count();
    }
    cout << "vectorized\t" << sum << " in " << clock_sum/1e6 << "s" << "\t\t =  " <<  iterations*3*sizeof(float)*n_elements/(clock_sum) << " MB/s"s << endl;

  clock_sum = 0;
  for (int i = 0; i < iterations; i++) {
    begin =  high_resolution_clock::now();
    sum = scalar_calculation(x, y, z, indices, n_elements);
    end =  high_resolution_clock::now();
    clock_sum += duration_cast<microseconds>( end - begin ).count();
  }
  cout << "scalar\t\t" << sum << " in " << clock_sum/1e6 << "s" << "\t\t =  " <<  iterations*3*sizeof(float)*n_elements/(clock_sum) << " MB/s"s << endl;

}
