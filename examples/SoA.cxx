//Compile with [clan,]g++ [-fopenmp] -march=native SoA.cxx -o bin/SoA
//and your preffered optimization levels.
// at the highest optimizations, and -ffast-math turned on, the scalar example
// will also get vectorized. However, it has been my experience that this will
// fail for more complicated versions.

#include <cstddef>
#include <cmath>
#include <vectorized_types.h>
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

//A simple function that computes the lenght of 3d vectors.
single_t vectorized_calculation(simd_t *__restrict__ x, simd_t *__restrict__ y, simd_t *__restrict__ z, int len){
  simd_t sum = 0.0;
  #pragma omp declare reduction(+:simd_t: omp_out = omp_out+omp_in) initializer(omp_priv = 0.0)
  #pragma omp parallel for reduction(+:sum) schedule(static)
  for(int i = 0; i < len; i++){
    sum += sqrt(x[i]*x[i]+y[i]*y[i]+z[i]*z[i]);
    //without -ffast-math we should write
    // sum += sqrt((x[i]*x[i])+y[i]*y[i])+z[i]*z[i]));
    //in which case the compiler can produce fma instructions instead.
  }
  //the .sum() function returns the sum of all the elements of the simd register.
  return sum.sum();
}

single_t scalar_calculation(single_t *__restrict__ x, single_t *__restrict__ y, single_t *__restrict__ z, int len){
  single_t sum = 0;
  #pragma omp parallel for reduction(+:sum) schedule(static)
  for(int i = 0; i < len; i++){
    //At the time of writting,
    // this function is actually simple enough that clang vectorizes it at -O2 -ffast-math.
    //g++ vectorizes it at -O3 -ffast-math
    sum += sqrt(x[i]*x[i]+y[i]*y[i]+z[i]*z[i]);
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

  random_device rd;
  default_random_engine e2(rd());
  uniform_real_distribution<> dist(-3.14, 3.14);
  //initialize the array with random points on a sphere of radius 1.
  for (int i = 0; i < n_elements; i ++) {
    single_t theta = dist(e2);
    single_t phi = dist(e2);
    x[i] = cos(theta)*sin(phi);
    y[i] = sin(theta)*sin(phi);
    z[i] = cos(phi);
  }

  single_t sum;
  double clock_sum;
  high_resolution_clock::time_point begin, end;

  const int iterations = 10;
  clock_sum = 0;
  for (int i = 0; i < iterations; i++) {
    begin =  high_resolution_clock::now();
    sum = scalar_calculation(x, y, z, n_elements);
    end =  high_resolution_clock::now();
    clock_sum += duration_cast<microseconds>( end - begin ).count();
  }
  cout << "scalar\t\t" << sum << " in " << clock_sum/1e6 << "s" << "\t\t =  " <<  iterations*3*sizeof(float)*n_elements/(clock_sum) << " MB/s"s << endl;


  clock_sum = 0;
  for (int i = 0; i < iterations; i++) {
    begin =  high_resolution_clock::now();
    sum = vectorized_calculation(x_simd, y_simd, z_simd, n_simd_elements);
    end =  high_resolution_clock::now();
    clock_sum += duration_cast<microseconds>( end - begin ).count();
  }
  cout << "vectorized\t" << sum << " in " << clock_sum/1e6 << "s" << "\t\t =  " <<  iterations*3*sizeof(float)*n_elements/(clock_sum) << " MB/s"s << endl;

}
