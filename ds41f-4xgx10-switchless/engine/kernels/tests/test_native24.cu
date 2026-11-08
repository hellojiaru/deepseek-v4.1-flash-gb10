// Wide kernels against the released <=16-row calls, including disjoint MoE union144.
#include "kernels/rows24.h"
#include "common/check.h"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <vector>
using bf=__nv_bfloat16;
int checks=0;
template<class T>T* alloc(size_t n){T* p;CUDA_CHECK(cudaMalloc(&p,n*sizeof(T)));CUDA_CHECK(cudaMemset(p,0,n*sizeof(T)));return p;}
__global__ void fill_bf(bf* p,size_t n,int seed){for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=size_t(gridDim.x)*blockDim.x)p[i]=__float2bfloat16(float(int((i*1664525+seed*1013904223u)%127)-63)/128);}
__global__ void fill_bytes(uint8_t* p,size_t n,int seed,bool scales=false){for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=size_t(gridDim.x)*blockDim.x)p[i]=scales?127+((i+seed)%2):uint8_t((i*13+seed*31)%256);}
__global__ void fill_fp8(uint8_t* p,size_t n,int seed){for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=size_t(gridDim.x)*blockDim.x)p[i]=uint8_t(((i+seed)%2)*128+((i*13+seed*31)%40));}
template<class T>void equal(T* a,T* b,size_t n,const char* label){std::vector<T>x(n),y(n);CUDA_CHECK(cudaMemcpy(x.data(),a,n*sizeof(T),cudaMemcpyDeviceToHost));CUDA_CHECK(cudaMemcpy(y.data(),b,n*sizeof(T),cudaMemcpyDeviceToHost));size_t bad=0;for(size_t i=0;i<n;++i)bad+=memcmp(&x[i],&y[i],sizeof(T))!=0;printf("%s: %zu different / %zu\n",label,bad,n);if(bad)exit(3);++checks;}
template<class F>void chunks(int M,F f){for(int r=0;r<M;r+=16)f(r,std::min(16,M-r));}
void gemvs(){
 const int K=5120,N=256;
 bf* x=alloc<bf>(24*K),*norm=alloc<bf>(K),*yn=alloc<bf>(24*K),*yr=alloc<bf>(24*K);
 fill_bf<<<64,256>>>(x,24*K,1);fill_bf<<<64,256>>>(norm,K,2);
 uint8_t* q=alloc<uint8_t>(24*K),*sc=alloc<uint8_t>(24*K/32),*qr=alloc<uint8_t>(24*K),*sr=alloc<uint8_t>(24*K/32);
 uint8_t* w=alloc<uint8_t>(N*K),*wt=alloc<uint8_t>(N*K),*ws=alloc<uint8_t>(N/32*K/32);
 fill_fp8<<<64,256>>>(w,N*K,3);fill_bytes<<<64,256>>>(ws,N/32*K/32,4,true);dsk::tile_fp8(w,wt,N,K,0);
 bf* an=alloc<bf>(24*N),*ar=alloc<bf>(24*N);
 float* fn=alloc<float>(24*N),*fr=alloc<float>(24*N);
 bf* wb=alloc<bf>(N*K);uint8_t* wbt=alloc<uint8_t>(size_t(N)*K*2);
 fill_bf<<<64,256>>>(wb,size_t(N)*K,5);dsk::tile_fp8((uint8_t*)wb,wbt,N,2*K,0);
 size_t coded_bytes;int raw;uint8_t* coded=dsk::bf16x_encode(wbt,N,K,&coded_bytes,&raw,0);
 for(int M:{17,18,23,24})for(bool pdl:{false,true}){
  dsk::GemvCfg c{1,1,8};c.tiled=true;c.pdl=pdl;
  dsk::act_quant(x,M,K,q,sc,0);
  dsk::Fp8Seg seg{wt,ws,an,N};dsk::gemv_fp8_multi24(q,sc,M,&seg,1,K,c,0);
  chunks(M,[&](int r,int n){dsk::gemv_fp8(q+r*K,sc+r*K/32,n,wt,ws,N,K,ar+r*N,c,0);});equal(an,ar,M*N,"fp8");
  dsk::norm24(x,norm,1e-6f,M,K,yn,q,sc,256,0);dsk::gemv_fp8_multi24(q,sc,M,&seg,1,K,c,0);
  chunks(M,[&](int r,int n){dsk::Fp8Seg s{wt,ws,ar+r*N,N};dsk::gemv_fp8_norm(x+r*K,norm,1e-6f,n,&s,1,K,c,0,yr+r*K,qr+r*K,sr+r*K/32);});
  equal(yn,yr,M*K,"norm-y");equal(q,qr,M*K,"norm-q");equal(sc,sr,M*K/32,"norm-scale");equal(an,ar,M*N,"fp8-norm");
  for(bool code:{false,true}){
   auto* weight=code?coded:wbt;
   dsk::gemv_bf16_norm24(x,norm,1e-6f,M,weight,N,K,fn,nullptr,c,0,q,sc,code);
   chunks(M,[&](int r,int n){if(code)dsk::gemv_bf16x_norm(x+r*K,norm,1e-6f,n,weight,N,K,fr+r*N,nullptr,c,0);else dsk::gemv_bf16_norm(x+r*K,norm,1e-6f,n,weight,N,K,fr+r*N,nullptr,c,0,qr+r*K,sr+r*K/32);});
   equal(fn,fr,M*N,code?"coded-head":"gate");
   if(!code){equal(q,qr,M*K,"gate-q");equal(sc,sr,M*K/32,"gate-scale");}
  }
  dsk::GemvCfg cw{1,4,4};cw.pdl=pdl;
  dsk::gemv_woa24(x,M,wt,ws,N,K,N,an,cw,0,q,sc);
  chunks(M,[&](int r,int n){dsk::gemv_woa(x+r*K,n,wt,ws,N,K,N,ar+r*N,cw,false,0,qr+r*N,sr+r*N/32);});
  equal(an,ar,M*N,"woa");equal(q,qr,M*N,"woa-q");equal(sc,sr,M*N/32,"woa-scale");
 }
 CUDA_CHECK(cudaDeviceSynchronize());
 for(void* p:std::initializer_list<void*>{x,norm,yn,yr,q,sc,qr,sr,w,wt,ws,an,ar,fn,fr,wb,wbt,coded})CUDA_CHECK(cudaFree(p));
}
void moe(){
 const int E=160,D=5120,I=576;dsk::MoeWeights w{};
 const size_t w13=size_t(E)*I*D/2,w2=size_t(E)*D*I/2,s13=size_t(E)*I*D/32,s2=size_t(E)*D*I/32;
 std::vector<void*> owned;
 auto bytes=[&](size_t n,int seed,bool sc=false){auto*p=alloc<uint8_t>(n);fill_bytes<<<256,256>>>(p,n,seed,sc);owned.push_back(p);return p;};
 auto fp8=[&](size_t n,int seed){auto*p=alloc<uint8_t>(n);fill_fp8<<<64,256>>>(p,n,seed);owned.push_back(p);return p;};
 w.w1=bytes(w13,1);w.w3=bytes(w13,2);w.w2=bytes(w2,3);w.s1=bytes(s13,4,true);w.s3=bytes(s13,5,true);w.s2=bytes(s2,6,true);
 w.sh_w1=fp8(size_t(I)*D,7);w.sh_w3=fp8(size_t(I)*D,8);w.sh_w2=fp8(size_t(D)*I,9);
 w.sh_s1=bytes(size_t(I/32)*D/32,1,true);w.sh_s3=bytes(size_t(I/32)*D/32,2,true);w.sh_s2=bytes(size_t(D/32)*I/32,3,true);
 dsk::MoeWeights coded=w;
 auto code=[&](const uint8_t* scales,int N,int KB,dsk::ScaleCode& out){auto*p=alloc<uint32_t>(dsk::scale_code_words(E,N,KB));owned.push_back(p);int base;dsk::scale_code(scales,E,N,KB,p,&base,0);out={p,uint32_t(base)*0x01010101u};};
 code(w.s1,I,D/32,coded.c1);code(w.s3,I,D/32,coded.c3);code(w.s2,D,I/32,coded.c2);
 bf* x=alloc<bf>(24*D);fill_bf<<<64,256>>>(x,24*D,11);auto*q=alloc<uint8_t>(24*D),*sc=alloc<uint8_t>(24*D/32);dsk::act_quant(x,24,D,q,sc,0);
 float* logits=alloc<float>(24*E),*bias=alloc<float>(E),*yn=alloc<float>(24*D),*yr=alloc<float>(24*D),*wn=alloc<float>(24*6),*wr=alloc<float>(24*6);
 int* in=alloc<int>(24*6),*ir=alloc<int>(24*6);
 auto* ws=alloc<uint8_t>(dsk::moe24_workspace_bytes()+64);auto* ref=alloc<uint8_t>(dsk::moe_workspace_bytes()+64);
 for(int pattern:{0,1,2}){
  std::vector<float> hl(24*E,-1000.f);
  for(int r=0;r<24;++r)for(int k=0;k<6;++k){int e=pattern==0?k:pattern==1?r*6+k:(r*7+k*13)%E;hl[r*E+e]=float(12-k);}
  CUDA_CHECK(cudaMemcpy(logits,hl.data(),hl.size()*4,cudaMemcpyHostToDevice));
  for(int M:{17,24})for(bool cs:{false,true})for(bool pdl:{false,true}){
   dsk::MoeCfg c{};c.up16=4;c.down16=2;c.pdl=pdl;
   auto weights=cs?coded:w;
   dsk::moe_routed24(q,sc,M,logits,bias,E,1.f,weights,ws,yn,c,0,in,wn);
   chunks(M,[&](int r,int n){dsk::moe_routed(q+r*D,sc+r*D/32,n,logits+r*E,bias,E,1.f,weights,ref,yr+r*D,c,0,ir+r*6,wr+r*6);});
   equal(yn,yr,M*D,"moe-y");equal(in,ir,M*6,"moe-ids");equal(wn,wr,M*6,"moe-weights");
  }
 }
 CUDA_CHECK(cudaDeviceSynchronize());
 for(void*p:owned)CUDA_CHECK(cudaFree(p));
 for(void*p:std::initializer_list<void*>{x,q,sc,logits,bias,yn,yr,wn,wr,in,ir,ws,ref})CUDA_CHECK(cudaFree(p));
}
int main(){cudaDeviceProp prop{};CUDA_CHECK(cudaGetDeviceProperties(&prop,0));printf("device %s cc%d%d\n",prop.name,prop.major,prop.minor);gemvs();moe();printf("native24: %d exact checks, 0 failed\n",checks);}
