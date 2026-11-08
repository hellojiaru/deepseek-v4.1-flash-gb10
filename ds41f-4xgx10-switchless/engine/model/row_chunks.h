// Preserve the released <=16 path; wide weight projections/MoE use native24 entry points.
// The each() helper remains for narrow attention contracts. Wide FP8 norm uses owned
// scratch once before a weight-reusing projection, to stay within SM shared memory.
#pragma once
#include <algorithm>
#include "kernels/dsk.h"
#include "kernels/rows24.h"
#include "model/config.h"

namespace model::row_chunks {
struct Scratch { __nv_bfloat16* y=nullptr; uint8_t* q=nullptr; uint8_t* s=nullptr; };
template <class T> inline T* offset(T* p, size_t n) { return p ? p+n : nullptr; }
template <class Fn> inline void each(int rows, Fn fn) {
  for (int r=0; r<rows; r+=dsk::kMaxRows) fn(r, std::min(dsk::kMaxRows, rows-r));
}
inline void gemv_fp8(const uint8_t* xq,const uint8_t* xs,int M,const uint8_t* W,const uint8_t* S,
                     int N,int K,__nv_bfloat16* y,dsk::GemvCfg cfg,cudaStream_t st) {
  if(M>16){const dsk::Fp8Seg seg{W,S,y,N};dsk::gemv_fp8_multi24(xq,xs,M,&seg,1,K,cfg,st);return;}
  each(M,[&](int r,int n){dsk::gemv_fp8(xq+size_t(r)*K,xs+size_t(r)*(K/32),n,W,S,N,K,y+size_t(r)*N,cfg,st);});
}
inline void gemv_fp8_norm(const __nv_bfloat16* x,const __nv_bfloat16* norm,float eps,int M,
                          const dsk::Fp8Seg* seg,int nseg,int K,dsk::GemvCfg cfg,cudaStream_t st,
                          __nv_bfloat16* y=nullptr,uint8_t* q=nullptr,uint8_t* qs=nullptr,Scratch scratch={}) {
  if(M>16){
    dsk::norm24(x,norm,eps,M,K,y?y:scratch.y,q?q:scratch.q,qs?qs:scratch.s,32*cfg.wn*cfg.wk,st);
    dsk::gemv_fp8_multi24(q?q:scratch.q,qs?qs:scratch.s,M,seg,nseg,K,cfg,st);return;
  }
  each(M,[&](int r,int n){
    dsk::Fp8Seg part[dsk::kMaxSegs];
    for(int j=0;j<nseg;++j){part[j]=seg[j];part[j].y+=size_t(r)*seg[j].N;}
    dsk::gemv_fp8_norm(x+size_t(r)*K,norm,eps,n,part,nseg,K,cfg,st,
                       offset(y,size_t(r)*K),offset(q,size_t(r)*K),offset(qs,size_t(r)*(K/32)));
  });
}
inline void gemv_woa(const __nv_bfloat16* x,int M,const uint8_t* W,const uint8_t* S,int N,int K,
                     int groups,__nv_bfloat16* y,dsk::GemvCfg cfg,bool cc,cudaStream_t st,
                     uint8_t* q=nullptr,uint8_t* qs=nullptr) {
  if(M>16){if(cc)abort();dsk::gemv_woa24(x,M,W,S,N,K,groups,y,cfg,st,q,qs);return;}
  each(M,[&](int r,int n){dsk::gemv_woa(x+size_t(r)*(N/groups)*K,n,W,S,N,K,groups,
    offset(y,size_t(r)*N),cfg,cc,st,offset(q,size_t(r)*N),offset(qs,size_t(r)*(N/32)));});
}
inline void gemv_bf16_norm(const __nv_bfloat16* x,const __nv_bfloat16* norm,float eps,int M,
                           const uint8_t* W,int N,int K,float* yf,__nv_bfloat16* yb,dsk::GemvCfg cfg,
                           cudaStream_t st,uint8_t* q=nullptr,uint8_t* qs=nullptr) {
  if(M>16){dsk::gemv_bf16_norm24(x,norm,eps,M,W,N,K,yf,yb,cfg,st,q,qs,false);return;}
  each(M,[&](int r,int n){dsk::gemv_bf16_norm(x+size_t(r)*K,norm,eps,n,W,N,K,
    offset(yf,size_t(r)*N),offset(yb,size_t(r)*N),cfg,st,offset(q,size_t(r)*K),offset(qs,size_t(r)*(K/32)));});
}
inline void gemv_bf16x_norm(const __nv_bfloat16* x,const __nv_bfloat16* norm,float eps,int M,
                            const uint8_t* W,int N,int K,float* yf,__nv_bfloat16* yb,dsk::GemvCfg cfg,
                            cudaStream_t st) {
  if(M>16){dsk::gemv_bf16_norm24(x,norm,eps,M,W,N,K,yf,yb,cfg,st,nullptr,nullptr,true);return;}
  each(M,[&](int r,int n){dsk::gemv_bf16x_norm(x+size_t(r)*K,norm,eps,n,W,N,K,
    offset(yf,size_t(r)*N),offset(yb,size_t(r)*N),cfg,st);});
}
inline void moe_routed(const uint8_t* xq,const uint8_t* xs,int M,const float* logits,const float* bias,
                       int E,float scale,const dsk::MoeWeights& w,void* ws,float* y,dsk::MoeCfg cfg,
                       cudaStream_t st,int* idx=nullptr,float* weights=nullptr) {
  if(M>16){dsk::moe_routed24(xq,xs,M,logits,bias,E,scale,w,ws,y,cfg,st,idx,weights);return;}
  each(M,[&](int r,int n){dsk::moe_routed(xq+size_t(r)*kDim,xs+size_t(r)*(kDim/32),n,
    logits+size_t(r)*E,bias,E,scale,w,ws,y+size_t(r)*kDim,cfg,st,
    offset(idx,size_t(r)*dsk::kTopK),offset(weights,size_t(r)*dsk::kTopK));});
}
} // namespace model::row_chunks
