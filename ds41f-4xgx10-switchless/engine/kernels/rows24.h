// Explicit wide entry points. The released <=16-row ABI/dispatch stays intact.
#pragma once
#include "kernels/dsk.h"
namespace dsk {
void norm24(const __nv_bfloat16*,const __nv_bfloat16*,float,int,int,__nv_bfloat16*,uint8_t*,uint8_t*,int,cudaStream_t);
void gemv_fp8_multi24(const uint8_t*,const uint8_t*,int,const Fp8Seg*,int,int,GemvCfg,cudaStream_t);
void gemv_woa24(const __nv_bfloat16*,int,const uint8_t*,const uint8_t*,int,int,int,__nv_bfloat16*,GemvCfg,cudaStream_t,uint8_t*,uint8_t*);
void gemv_bf16_norm24(const __nv_bfloat16*,const __nv_bfloat16*,float,int,const uint8_t*,int,int,float*,__nv_bfloat16*,GemvCfg,cudaStream_t,uint8_t*,uint8_t*,bool);
size_t moe24_workspace_bytes();
void moe_routed24(const uint8_t*,const uint8_t*,int,const float*,const float*,int,float,const MoeWeights&,void*,float*,MoeCfg,cudaStream_t,int*,float*);
}
