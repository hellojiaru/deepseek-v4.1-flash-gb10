#include "model/creative.h"
#include "common/check.h"

namespace model::creative {
namespace {
constexpr int D = 5120, T = 256;
__global__ void projection(__nv_bfloat16* y, const float* r, const float* lambda,
                           unsigned long long* counts) {
  const float lam = *lambda;
  // Normal does not read, round, or write any activation.
  if (lam == 0.f) return;
  __shared__ float sums[T];
  const int t = threadIdx.x;
  y += (size_t)blockIdx.x * D;
  float dot = 0.f;
  for (int d = t; d < D; d += T) dot = __fadd_rn(dot, __fmul_rn(__bfloat162float(y[d]), r[d]));
  sums[t] = dot;
  __syncthreads();
  for (int stride = T / 2; stride; stride /= 2) {
    if (t < stride) sums[t] = __fadd_rn(sums[t], sums[t + stride]);
    __syncthreads();
  }
  const float scale = __fmul_rn(lam, sums[0]);
  for (int d = t; d < D; d += T)
    y[d] = __float2bfloat16_rn(__fsub_rn(__bfloat162float(y[d]), __fmul_rn(scale, r[d])));
  if (t == 0 && counts) atomicAdd(counts, 1ULL);
}
}
void project(__nv_bfloat16* y, int rows, const float* direction, const float* lambda,
             unsigned long long* counts, cudaStream_t stream) {
  if (!rows) return;
  projection<<<rows, T, 0, stream>>>(y, direction, lambda, counts);
  CUDA_CHECK(cudaGetLastError());
}
}
