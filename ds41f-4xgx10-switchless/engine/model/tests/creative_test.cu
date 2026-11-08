#include "model/creative.h"
#include "common/check.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

int main() {
  constexpr int D = 5120;
  cudaStream_t st; CUDA_CHECK(cudaStreamCreate(&st));
  std::mt19937 gen(42); std::normal_distribution<float> normal;
  std::vector<float> r(D); double norm=0;
  for (auto& v:r) { v=normal(gen); norm+=double(v)*v; }
  for (auto& v:r) v/=std::sqrt(norm);
  float *rd,*ld; __nv_bfloat16* yd; unsigned long long* counts;
  CUDA_CHECK(cudaMalloc(&rd,D*sizeof(float))); CUDA_CHECK(cudaMalloc(&ld,sizeof(float)));
  CUDA_CHECK(cudaMalloc(&yd,128*D*sizeof(__nv_bfloat16))); CUDA_CHECK(cudaMalloc(&counts,sizeof(*counts)));
  CUDA_CHECK(cudaMemcpy(rd,r.data(),D*sizeof(float),cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(counts,0,sizeof(*counts)));
  for (int rows:{1,6,16,128}) {
    std::vector<__nv_bfloat16> in(rows*D),out(in.size()),ref(in.size());
    for (auto& y:in) y=__float2bfloat16_rn(normal(gen));
    float lam=0; CUDA_CHECK(cudaMemcpy(ld,&lam,sizeof lam,cudaMemcpyHostToDevice));
    cudaGraph_t g; cudaGraphExec_t e;
    CUDA_CHECK(cudaStreamBeginCapture(st,cudaStreamCaptureModeThreadLocal));
    model::creative::project(yd,rows,rd,ld,counts,st);
    CUDA_CHECK(cudaStreamEndCapture(st,&g)); CUDA_CHECK(cudaGraphInstantiate(&e,g,0));
    for (int mode:{0,1,0}) {
      lam=mode?1.5f:0.f;
      CUDA_CHECK(cudaMemcpy(ld,&lam,sizeof lam,cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(yd,in.data(),in.size()*2,cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaGraphLaunch(e,st)); CUDA_CHECK(cudaStreamSynchronize(st));
      CUDA_CHECK(cudaMemcpy(out.data(),yd,out.size()*2,cudaMemcpyDeviceToHost));
      if (!mode) {
        if (std::memcmp(in.data(),out.data(),in.size()*2)) return 2;
      } else {
        float worst=0; size_t changed=0;
        for(int row=0;row<rows;++row) {
          double dot=0;
          for(int d=0;d<D;++d) dot+=double(__bfloat162float(in[row*D+d]))*r[d];
          for(int d=0;d<D;++d) {
            const int i=row*D+d;
            const float want=__bfloat162float(__float2bfloat16_rn(float(__bfloat162float(in[i])-1.5*dot*r[d])));
            worst=std::max(worst,std::abs(__bfloat162float(out[i])-want));
            changed+=std::memcmp(&in[i],&out[i],2)!=0;
          }
        }
        if(worst>0.015625f || !changed) return 3;
        // Same input: eager and a graph captured with lambda=0 must agree bit for bit at lambda=1.5.
        CUDA_CHECK(cudaMemcpy(yd,in.data(),in.size()*2,cudaMemcpyHostToDevice));
        model::creative::project(yd,rows,rd,ld,counts,st); CUDA_CHECK(cudaStreamSynchronize(st));
        CUDA_CHECK(cudaMemcpy(ref.data(),yd,ref.size()*2,cudaMemcpyDeviceToHost));
        if(std::memcmp(ref.data(),out.data(),ref.size()*2)) return 4;
        std::printf("rows %d: changed %zu; max absolute error vs FP64 reference %.9g; graph=eager\n",rows,changed,worst);
      }
    }
    CUDA_CHECK(cudaGraphExecDestroy(e)); CUDA_CHECK(cudaGraphDestroy(g));
  }
  unsigned long long n=0; CUDA_CHECK(cudaMemcpy(&n,counts,sizeof n,cudaMemcpyDeviceToHost));
  if(n!=2*(1+6+16+128)) return 5;
  CUDA_CHECK(cudaFree(rd)); CUDA_CHECK(cudaFree(ld)); CUDA_CHECK(cudaFree(yd)); CUDA_CHECK(cudaFree(counts));
  CUDA_CHECK(cudaStreamDestroy(st));
  std::puts("PASS: Normal bitwise identity, N/C/N replay, FP64 oracle, eager/graph, row counts");
}
