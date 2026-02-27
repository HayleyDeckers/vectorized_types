//Compile with [clan,]g++ [-fopenmp] -march=native SoA.cxx -o bin/SoA
//and your preffered optimization levels.
// at the highest optimizations, and -ffast-math turned on, the scalar example
// will also get vectorized. However, it has been my experience that this will
// fail for more complicated versions.


//What to test:
// different layouts:
//  -AoS, SoA, interleaved (various sizes: simd size, cacheline, pagesize?)
// different access patterns:
//  -read all, Pruning (skip some member variables), skimming (skip some indices), pruning + skimming
//    For skimming, we should use a "1 in X change of being read for realistic results"
//     i.e. choice of being selected is independent of position/neighbours.
//  also try different, but realistic, sizes for each.
//
// // c.f. critical stride, might (dramatically) effect certain tests.
//
// Conclusion: cacheline sized or pagesized data is best for reading from memory. Pruning is harder though because data is heavilly interleaved.
// Generally speaking, interleaved by page-sizes gives the best results. Should also mmap/copy fast when pruned.
//  Now, the problem with this route is:
//     - hard to specify for each type.
//     - requires *some* serialization effect when copying.
//     - indices are complicated, a bit. If they are generated based on non-interleaved data (i.e., logically specified)
//       then they require complex computation to match the 'real' index for the interleaved data. This cost can be partially migated
//       by forcing powers-of-two interleaving but still require at least 1 integer multiplication per index calculation. This is an acceptable
//       cost if followed by some kind of computation.
// So now also consider reading from a file (or machine) using mmap while pruning and compression.
// i.e. create a large file on disk


#include <thread>
#include <cassert>
#include <cstddef>
#include <cmath>
#include <vectorized_types.h>
#include <vector>
#include <random>
#include <chrono>
#include <iostream>
#include <algorithm>
#include <memory>

using namespace std;
using namespace std::chrono;
using namespace vec;
using single_t = float;
using simd_t = vectorized_type<single_t>;
using index_t = int32_t;

struct point_t{
  single_t x,y,z;
};

point_t point_gen(){
 static random_device rd;
 static default_random_engine e2(rd());
 static uniform_real_distribution<single_t> dist(-3.14, 3.14);
 point_t ret;
 single_t theta = dist(e2);
 single_t phi = dist(e2);
 ret.x = cos(theta)*sin(phi);
 ret.y = sin(theta)*sin(phi);
 ret.z = cos(phi);
 return ret;
}

template<typename T>
 struct myVec{
   std::unique_ptr<T> mData;
   int mSize;
   myVec(int size) : mData((T*)aligned_alloc(4096, size*sizeof(T))), mSize(size){
    //  std::cout << "created container of size " << mSize << " @ " << mData.get() << std::endl;
   }
   ~myVec(){
    //  std::cout << "freeing data @ " << mData.get() << std::endl;
   }
   int size() const{return mSize;}
   T& operator[](int idx){return mData.get()[idx];}
   const T& operator[](int idx) const{return mData.get()[idx];}

 };
template<bool Store_SIMD, unsigned Extra_Vars, unsigned Mul_factor = 1>
class AoS{
public:
  using Type = typename std::conditional<Store_SIMD, simd_t, single_t>::type;
  class Data{
    Type data[(3+Extra_Vars)*Mul_factor];
    public:
    const Type* x() const{return &data[0];}
    const Type* y() const{return &data[Mul_factor];}
    const Type* z() const{return &data[2*Mul_factor];}
    single_t* xp(){return (single_t*)(data);}
    single_t* yp(){return (single_t*)(data+Mul_factor);}
    single_t* zp(){return (single_t*)(data+2*Mul_factor);}
    Type len(int idx) const{
      simd_t X = _mm256_stream_load_si256((const __m256i*)(x()+idx));
      simd_t Y = _mm256_stream_load_si256((const __m256i*)(y()+idx));
      simd_t Z = _mm256_stream_load_si256((const __m256i*)(z()+idx));
      return X*X + Y*Y + Z*Z;
    }
  };
private:
  std::vector<index_t> mIndices;
  myVec<Data> mData;
  constexpr int width() const{
    if constexpr(Store_SIMD){
      return simd_t::Width*Mul_factor;
    }else{
      return Mul_factor;
    }
  }
public:
  AoS(int size) : mData(size / width()){/*assert(size % width() == 0);*/}
  const Data& operator[](int idx) const {return mData[idx];}
  void fill(){
    for(int i = 0; i < mData.size();i++){
      for(int j = 0; j < width(); j++){
        point_t p = point_gen();
        mData[i].xp()[j] = p.x;
        mData[i].yp()[j] = p.y;
        mData[i].zp()[j] = p.z;
      }
    }
  }
  int bytes() const {
    return mData.size()*sizeof(mData[0]);
  }
  void generate_indices(double chance){
    random_device rd;
    default_random_engine e2(rd());
    geometric_distribution<index_t> index_dist(chance);
    int counter = 0;
    mIndices.clear();
    do{
      if constexpr (Store_SIMD){
        //divide by width to get block logical index. Multiply by amount of variables to get block stride, multiply by width to get element index of start block, add modulo to get index into block.
        //mask lower bits, multiply by constant, and with inverse mask lower bits.
        int real_idx = (counter/width())*(3+Extra_Vars)*width() + (counter%width());
        mIndices.push_back(real_idx);
        counter += 1;
      }else{
        mIndices.push_back(counter++);
      }
      counter += index_dist(e2);
    }while(counter < mData.size()*width());
    mIndices.resize(mIndices.size() - (mIndices.size() % width()));
    // std::cout << "generated " << mIndices.size() << " indexes = " << ((double)mData.size()*width())/mIndices.size() << std::endl;
  }
  size_t indexed_size() const {return mIndices.size();}
  single_t indexed_len() {
    const single_t *X = (const single_t*)((const simd_t*)mData.data()+0);
    const single_t *Y = (const single_t*)((const simd_t*)mData.data()+1*Mul_factor);
    const single_t *Z = (const single_t*)((const simd_t*)mData.data()+2*Mul_factor);

    Type sum = 0.0;
      if constexpr (Store_SIMD){
        #pragma omp declare reduction(+:Type: omp_out = omp_out+omp_in) initializer(omp_priv = 0.0)
        #pragma omp parallel for reduction(+:sum) schedule(static)
        for(int i = 0; i < mIndices.size(); i+= simd_t::Width){
          simd_t x = simd_t::gather(X, mIndices.data()+i);
          simd_t y = simd_t::gather(Y, mIndices.data()+i);
          simd_t z = simd_t::gather(Z, mIndices.data()+i);
          sum += cos(x*x + y*y + z*z);
        }
        return sum.sum();
      }else{
        #pragma omp declare reduction(+:Type: omp_out = omp_out+omp_in) initializer(omp_priv = 0.0)
        #pragma omp parallel for reduction(+:sum) schedule(static)
        for(int i = 0; i < mIndices.size();i++){
          for(int j = 0; j < Mul_factor; j++){
            sum += mData[mIndices[i]].len(j);
          }
        }
        return sum;
    }
  }
  single_t total_len() {
    Type sum = 0.0;
    #pragma omp declare reduction(+:Type: omp_out = omp_out+omp_in) initializer(omp_priv = 0.0)
    #pragma omp parallel for reduction(+:sum) schedule(static,1)
    for(int i = 0; i < mData.size(); i++){
      for(int j = 0; j < Mul_factor; j++){
        sum += mData[i].len(j);
      }
    }
    if constexpr (Store_SIMD){
      return sum.sum();
    }else{
      return sum;
    }
  }
};

template<bool SIMD>
class SoA{
private:
  std::vector<index_t> mIndices;
  std::vector<simd_t> mData[3];
  constexpr int width() const {return simd_t::Width;}
public:
  using Type = typename std::conditional<SIMD, simd_t, single_t>::type;
  SoA(int size){
    assert(size % width() == 0);
    for(int i = 0; i < 3; i++){
      mData[i] = std::vector<simd_t>(size/width());
    }
  }
  void generate_indices(double chance){
    random_device rd;
    default_random_engine e2(rd());
    geometric_distribution<index_t> index_dist(chance);
    int counter = 0;
    mIndices.clear();
    do{
      mIndices.push_back(counter++);
      counter += index_dist(e2);
    }while(counter < mData[0].size()*width());
    mIndices.resize(mIndices.size() - (mIndices.size() % width()));
    // std::cout << "generated " << mIndices.size() << " indexes = " << ((double)mData[0].size()*width())/mIndices.size() << std::endl;
  }
  // const Data& operator[](int idx) const {return mData[idx];}
  void fill(){
    for(int i = 0; i < mData[0].size();i++){
      for(int j = 0; j < width(); j++){
        point_t p = point_gen();
        mData[0][i].set(j,p.x);
        mData[1][i].set(j,p.y);
        mData[2][i].set(j,p.z);
      }
    }
  }
  int indexed_size() const{
    return mIndices.size();
  }
  single_t indexed_len(){
    Type sum = 0.0;
    const single_t * X = (const single_t*)mData[0].data();
    const single_t * Y = (const single_t*)mData[1].data();
    const single_t * Z = (const single_t*)mData[2].data();
    if constexpr(SIMD) {
      #pragma omp declare reduction(+:Type: omp_out = omp_out+omp_in) initializer(omp_priv = 0.0)
      #pragma omp parallel for reduction(+:sum) schedule(static)
      for(int i = 0; i < mIndices.size(); i+= simd_t::Width){
        simd_t x = simd_t::gather(X, mIndices.data()+i);
        simd_t y = simd_t::gather(Y, mIndices.data()+i);
        simd_t z = simd_t::gather(Z, mIndices.data()+i);
        sum += cos(x*x + y*y + z*z);
      }
      return sum.sum();
    }else{
      #pragma omp declare reduction(+:Type: omp_out = omp_out+omp_in) initializer(omp_priv = 0.0)
      #pragma omp parallel for reduction(+:sum) schedule(static)
      for(int i = 0; i < mIndices.size(); i++){
        single_t x = X[mIndices[i]];
        single_t y = Y[mIndices[i]];
        single_t z = Z[mIndices[i]];
        sum += cos(x*x + y*y + z*z);
      }
      return sum;
    }
  }
  single_t total_len(){
    Type sum = 0.0;
    #pragma omp declare reduction(+:Type: omp_out = omp_out+omp_in) initializer(omp_priv = 0.0)
    #pragma omp parallel for reduction(+:sum) schedule(static)
    for(int i = 0; i < mData[0].size(); i++){
      if constexpr(SIMD) {
        simd_t x = mData[0][i];
        simd_t y = mData[1][i];
        simd_t z = mData[2][i];
        sum += cos(x*x + y*y + z*z);
      }else{
        for(int j = 0; j < width(); j++){
          single_t x = mData[0][i][j];
          single_t y = mData[1][i][j];
          single_t z = mData[2][i][j];
          sum += cos(x*x + y*y + z*z);
        }
      }
    }
    if constexpr (SIMD){
      return sum.sum();
    }else{
      return sum;
    }
  }
};


struct my_clock_t{
  double min;
  double max;
};

template<typename Lambda>
my_clock_t time_invocation(Lambda&& object, int iterations){
  high_resolution_clock::time_point begin, end;
  single_t sum;
  long clock_max = INT64_MIN;
  long clock_min = INT64_MAX;
  for (int i = 0; i < iterations; i++) {
    std::this_thread::yield();
    begin =  high_resolution_clock::now();
    sum = object();
    end =  high_resolution_clock::now();
    // std::cout << "finished in: " << duration_cast<microseconds>( end - begin ).count() << " ticks" << std::endl;
    clock_min = std::min(clock_min, duration_cast<microseconds>( end - begin ).count());
    clock_max = std::max(clock_max, duration_cast<microseconds>( end - begin ).count());
  }
  my_clock_t ret = {.min = (double)clock_min, .max = (double)clock_max};
  return ret;
}

template<bool is_AoS, bool SIMD, unsigned Extra = 0, unsigned Pack = 1>
void benchmark(){
  const int n_elements = 1024*1024*64;
  using bench_t = typename std::conditional<is_AoS, AoS<SIMD, Extra, Pack>, SoA<SIMD>>::type;
  bench_t bench(n_elements);
  if constexpr(is_AoS){
    std::cout << "AoS: " << (SIMD ? "SIMD, " : "Scalar, ") << Extra << ", " << Pack << std::endl;
  }else{
    std::cout << "SoA: " << (SIMD ? "SIMD " : "Scalar ")  << std::endl;
  }
  bench.fill();
  // std::cout << "  filled, sum = " << bench.total_len() << std::endl;
  my_clock_t timing = time_invocation([&bench]{return bench.total_len();}, 20);
  std::cout << "  bandwidth (GB/s) = " << (bench.bytes()) / (1000*timing.min) << " / " << (bench.bytes()) / (1000*timing.max) << std::endl;
    // for(int i = 2; i <= 64; i *=2){
    //   bench.generate_indices(1.0/i);
    //   // std::cout << "toucing " << (4*bench.indexed_size()*sizeof(single_t))/(1<<20) << "MiB"<<std::endl;
    //   // std::cout << "  indexed sum = " << bench.indexed_len() << "/" << bench.indexed_size()std::endl;
    //   timing = time_invocation([bench]{return bench.indexed_len();}, 20);
    //   std::cout << "  bandwidth (GB/s) = " << (3e-3*bench.indexed_size()*sizeof(single_t)) /timing.min << " / " << (3e-3*bench.indexed_size()*sizeof(single_t)) /timing.max << std::endl;
    // }
  //
}

int main(){
  benchmark<true,true,  0, (32)/sizeof(simd_t)>();
  // benchmark<true,true,  0, (32)/sizeof(simd_t)+(32/2)/sizeof(simd_t)>();
  benchmark<true,true,  0, (64)/sizeof(simd_t)>();
  // benchmark<true,true,  0, (64)/sizeof(simd_t)+(64/2)/sizeof(simd_t)>();
  // benchmark<true,true,  0, (128)/sizeof(simd_t)>();
  // benchmark<true,true,  0, (128)/sizeof(simd_t)+(128/2)/sizeof(simd_t)>();
  // benchmark<true,true,  0, (256)/sizeof(simd_t)>();
  // benchmark<true,true,  0, (256)/sizeof(simd_t)+(256/2)/sizeof(simd_t)>();
  // benchmark<true,true,  0, (512)/sizeof(simd_t)>();
  // benchmark<true,true,  0, (512)/sizeof(simd_t)+(512/2)/sizeof(simd_t)>();
  // benchmark<true,true,  0, (1024)/sizeof(simd_t)>();
  // benchmark<true,true,  0, (1024)/sizeof(simd_t)+(1024/2)/sizeof(simd_t)>();
  // benchmark<true,true,  0, (2048)/sizeof(simd_t)>();
  // benchmark<true,true,  0, (2048)/sizeof(simd_t)+(2048/2)/sizeof(simd_t)>();
  benchmark<true,true,  0, (4096)/sizeof(simd_t)>();
  benchmark<true,true,  0, (4096)/sizeof(simd_t)+(4096/2)/sizeof(simd_t)>();
  // benchmark<true,true,  0, (8192)/sizeof(simd_t)>();
  // benchmark<true,true,  0, (8192)/sizeof(simd_t)+(8192/2)/sizeof(simd_t)>();
  benchmark<true,true,  0, (16384)/sizeof(simd_t)>();
  benchmark<true,true,  0, (16384)/sizeof(simd_t)+(16384/2)/sizeof(simd_t)>();
  // benchmark<true,true,  0, (32)/sizeof(simd_t)>();
  // benchmark<true,true,  0, (32)/sizeof(simd_t)+(32/2)/sizeof(simd_t)>();
  // benchmark<true,true,  0, (64)/sizeof(simd_t)>();
  // benchmark<true,true,  0, (64)/sizeof(simd_t)+(64/2)/sizeof(simd_t)>();
  // benchmark<true,true,  0, (128)/sizeof(simd_t)>();
  // benchmark<true,true,  0, (128)/sizeof(simd_t)+(128/2)/sizeof(simd_t)>();
  // benchmark<true,true,  0, (256)/sizeof(simd_t)>();
  // benchmark<true,true,  0, (256)/sizeof(simd_t)+(256/2)/sizeof(simd_t)>();
  // benchmark<true,true,  0, (512)/sizeof(simd_t)>();
  // benchmark<true,true,  0, (512)/sizeof(simd_t)+(512/2)/sizeof(simd_t)>();
  // benchmark<true,true,  0, (1024)/sizeof(simd_t)>();
  // benchmark<true,true,  0, (1024)/sizeof(simd_t)+(1024/2)/sizeof(simd_t)>();
  // benchmark<true,true,  0, (2048)/sizeof(simd_t)>();
  // benchmark<true,true,  0, (2048)/sizeof(simd_t)+(2048/2)/sizeof(simd_t)>();
  // benchmark<true,true,  0, (4096)/sizeof(simd_t)>();
  // benchmark<true,true,  0, (4096)/sizeof(simd_t)+(4096/2)/sizeof(simd_t)>();
  // benchmark<true,true,  0, (8192)/sizeof(simd_t)>();
  // benchmark<true,true,  0, (8192)/sizeof(simd_t)+(8192/2)/sizeof(simd_t)>();
  // benchmark<true,true,  0, (16384)/sizeof(simd_t)>();
  // benchmark<true,true,  0, (16384)/sizeof(simd_t)+(16384/2)/sizeof(simd_t)>();
  // benchmark<true,true,  0, (1<<21)/sizeof(simd_t)>();
  // benchmark<false,true>();
  // benchmark<false,false>();
  // std::cout << "================" << std::endl;
  // benchmark<true,true,  0, (2048)/sizeof(simd_t)>();
  // benchmark<true,true,  0, (4096)/sizeof(simd_t)>();
  // benchmark<true,true,  0, (4000)/sizeof(simd_t)>();
  //
  // std::cout << "================" << std::endl;
  // // benchmark<true,true, 0, 1>();
  // benchmark<true,true,  0, 2>();
  // // benchmark<true,false, 0, 1>();
  // std::cout << "================" << std::endl;
  // // benchmark<true,true,  1, 1>();
  // // benchmark<true,true,  1, 4096/sizeof(simd_t)>();
  // // benchmark<true,true,  1, 2>();
  // std::cout << "================" << std::endl;
  // // benchmark<true,true,  4, 1>();
  // // benchmark<true,true,  4, (2048+4096)/sizeof(simd_t)>();
  // // benchmark<true,true,  4, (1<<21)/sizeof(simd_t)>();
  // // benchmark<true,true,  4, 2>();
  // std::cout << "================" << std::endl;
  // benchmark<false,true>();
  // benchmark<true,true,  5, 1>();
  // benchmark<true,true,  5, 2>();
  // benchmark<true,false, 4, 1>();

  return 0;
}
