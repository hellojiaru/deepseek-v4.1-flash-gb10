#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>
namespace model::creative {
// Each row contains the complete hidden dimension, even when it is a TP partial.
// lambda is device resident: CUDA Graph replay observes mode changes at command boundaries.
void project(__nv_bfloat16* y, int rows, const float* direction, const float* lambda,
             unsigned long long* counts, cudaStream_t stream);
}
