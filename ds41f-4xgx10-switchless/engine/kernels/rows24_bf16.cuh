template <int FMT, int MT, bool CC, bool NORM, int RB>
__device__ __forceinline__ void gemv16_body24(const __nv_bfloat16* __restrict__ x, int M, const uint8_t* __restrict__ W,
                                            const uint8_t* __restrict__ S, int N, int K, int group_rows,
                                            float* __restrict__ yf, __nv_bfloat16* __restrict__ yb,
                                            uint8_t* __restrict__ q, uint8_t* __restrict__ qs, int wn, int wk,
                                            NormIn nin) {
  static_assert(!NORM || (FMT != kWOA && !CC), "the norm prologue is for the bf16 tensor-core path");
  static_assert(RB == kLowRows || (RB == 24 && !CC), "9..16 rows: tensor cores");
  constexpr bool kHi = RB > kLowRows;     // rows 8..15 in the upper half
  constexpr bool kSx = NORM && !kHi;      // the normed rows in shared memory (otherwise normed as read)
  constexpr bool kB16 = FMT != kWOA;      // bf16 values (kBF16X: once decoded)
  constexpr int KU = kB16 ? 4 : 2;  // steps per batch
  constexpr int VPS = kB16 ? 32 : 64;  // k values per 64-byte step
  extern __shared__ float red[];  // [wk][RB][rows_cta] (NORM: >= 256 + RB), up to 8 rows NORM: then x [M][K + 32] bf16
  const int rows_cta = wn * 8 * MT;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int wi_n = warp % wn, wi_k = warp / wn;
  const int n_cta = blockIdx.x * rows_cta, n0 = n_cta + wi_n * 8 * MT;
  const int rowb = kB16 ? 2 * K : K;  // weight bytes per row (kBF16X: before the coding)
  const int P = rowb / 64;
  const int p0 = (P * wi_k) / wk, p1 = (P * (wi_k + 1)) / wk;
  const int ldx = FMT == kWOA ? K * (N / group_rows) : K;
  const int xoff = FMT == kWOA ? (n_cta / group_rows) * K : 0;  // a CTA's rows share a group
  const bool live = g < M, live2 = kHi && g + 8 < M;

  const uint8_t* wrow[MT];   // kBF16X: the sign-mantissa bytes
  const uint8_t* srow[MT];   // kWOA: the block scales; kBF16X: the exponent offsets
  const uint32_t* hrow[MT];  // kBF16X: the headers, hrow[i][p] tile (i, p)'s (every lane reads it: L1)
  const size_t T = (size_t)(N / 8) * P;  // kBF16X's tiles
#pragma unroll
  for (int i = 0; i < MT; ++i) {
    const int n = n0 + 8 * i;
    if constexpr (FMT == kBF16X) {
      wrow[i] = W + (size_t)(n / 8) * P * 256 + 8 * lane;
      srow[i] = W + T * 256 + (size_t)(n / 8) * P * 128 + 4 * lane;
      hrow[i] = reinterpret_cast<const uint32_t*>(W + T * 384) + (size_t)(n / 8) * P;
    } else {
      wrow[i] = W + (size_t)(n / 8) * 8 * rowb + g * 64 + 16 * t;
      srow[i] = FMT == kWOA ? S + (size_t)(n / 32) * (K / 32) + (t >> 1) : nullptr;
    }
  }
  // kBF16X: the lane's bytes of raw tile 0, less 2^40: raw tile r (header h = 2^31 + r) at rawq + 512 h
  const uint64_t rawq = FMT == kBF16X ? reinterpret_cast<uint64_t>(W + bf16x_raw_off(T) + 16 * lane) - (1ull << 40) : 0;
  // MMA: this lane's activation row (kHi: and row g + 8); CC: all rows, read per step
  __nv_bfloat16* sx = reinterpret_cast<__nv_bfloat16*>(red + max(wk * RB * rows_cta, 256 + RB));
  const __nv_bfloat16* xrow = kSx ? sx + (live ? g : 0) * (K + 32) + 8 * t
                                  : (NORM ? nin.x : x) + (size_t)(live ? g : 0) * ldx + xoff + (kB16 ? 8 : 16) * t;
  const __nv_bfloat16* xrow2 = (NORM ? nin.x : x) + (size_t)(live2 ? g + 8 : 0) * ldx + xoff + (kB16 ? 8 : 16) * t;
  const __nv_bfloat16* nwrow = NORM ? nin.w + 8 * t : nullptr;  // kHi NORM: the norm weight, and the rows' 1/rms
  float rr = 0.f, rr2 = 0.f;

  float c[2][MT][4] = {};
  float rr24[2]={}, rr224[2]={};
  float acc[MT][RB];
#pragma unroll
  for (int i = 0; i < MT; ++i) {
    c[0][i][0] = c[0][i][1] = c[0][i][2] = c[0][i][3] = 0.f;
#pragma unroll
    for (int m = 0; m < RB; ++m) acc[i][m] = 0.f;
  }

  uint4 ba[KU][MT], bb[KU][MT];
  uint32_t sa[KU][MT], sb[KU][MT];
  auto load = [&](uint4 (&b)[KU][MT], uint32_t (&s)[KU][MT], int p) {
#pragma unroll
    for (int u = 0; u < KU; ++u)
      if (p + u < p1)
#pragma unroll
        for (int i = 0; i < MT; ++i) {
          if constexpr (FMT == kBF16X) {
            const uint32_t h = __ldg(hrow[i] + p + u);
            b[u][i] = ldg_bf16x(h, reinterpret_cast<const uint8_t*>(rawq + (uint64_t)h * 512),
                                wrow[i] + (size_t)(p + u) * 256, srow[i] + (size_t)(p + u) * 128);
          } else {
            b[u][i] = ldg_stream128(wrow[i] + (p + u) * 512);
            if (FMT == kWOA) s[u][i] = __ldg(srow[i] + 2 * (p + u));
          }
        }
  };
  // a lane's 16 bytes of bf16 of the warp's tile (i, p), as kBF16 loads them: kBF16X decodes a coded tile
  // in place (the header read again rather than kept: 16 registers fewer)
  auto w16 = [&](uint4& b, int i, int p) -> const uint4& {
    if constexpr (FMT == kBF16X) {
      const uint32_t h = __ldg(hrow[i] + p);
      if ((int)h >= 0) b = bf16x_dec(b.x, b.y, b.z, h);
    }
    return b;
  };
  auto comp = [&](uint4 (&b)[KU][MT], const uint32_t (&s)[KU][MT], int p) {
#pragma unroll
    for (int u = 0; u < KU; ++u) {
      if (p + u >= p1) break;
      const int k0 = (p + u) * VPS;
      uint4 decoded[MT];
#pragma unroll
      for(int i=0;i<MT;++i)decoded[i]=w16(b[u][i],i,p+u);
#pragma unroll
      for(int z=0;z<2;++z){
        const bool live=g+16*z<M,live2=g+16*z+8<M;
        const __nv_bfloat16* xrow=(NORM?nin.x:x)+size_t(live?g+16*z:0)*ldx+xoff+(kB16?8:16)*t;
        const __nv_bfloat16* xrow2=(NORM?nin.x:x)+size_t(live2?g+16*z+8:0)*ldx+xoff+(kB16?8:16)*t;
        const float rr=rr24[z],rr2=rr224[z];
      if constexpr (!CC && !kHi) {
        if constexpr (kB16) {
          uint4 a = make_uint4(0, 0, 0, 0);
          if (live) a = NORM ? *reinterpret_cast<const uint4*>(xrow + k0) : __ldg(reinterpret_cast<const uint4*>(xrow + k0));
#pragma unroll
          for (int i = 0; i < MT; ++i) {
            const uint4& w = decoded[i];
            mma_bf16(c[z][i], a.x, 0u, a.y, 0u, w.x, w.y);
            mma_bf16(c[z][i], a.z, 0u, a.w, 0u, w.z, w.w);
          }
        } else {
          uint4 a0 = make_uint4(0, 0, 0, 0), a1 = a0;
          if (live) a0 = __ldg(reinterpret_cast<const uint4*>(xrow + k0)), a1 = __ldg(reinterpret_cast<const uint4*>(xrow + k0 + 8));
#pragma unroll
          for (int i = 0; i < MT; ++i) {
            uint32_t wv[8];
            widen16(b[u][i], s[u][i], wv);
            mma_bf16(c[z][i], a0.x, 0u, a0.y, 0u, wv[0], wv[1]);
            mma_bf16(c[z][i], a0.z, 0u, a0.w, 0u, wv[2], wv[3]);
            mma_bf16(c[z][i], a1.x, 0u, a1.y, 0u, wv[4], wv[5]);
            mma_bf16(c[z][i], a1.z, 0u, a1.w, 0u, wv[6], wv[7]);
          }
        }
      } else if constexpr (!CC) {  // 9..16 rows: rows g and g + 8
        if constexpr (kB16) {
          uint4 a = make_uint4(0, 0, 0, 0), a2 = a;
          if constexpr (NORM) {
            const uint4 wn8 = __ldg(reinterpret_cast<const uint4*>(nwrow + k0));
            if (live) a = rmsnorm8(__ldg(reinterpret_cast<const uint4*>(xrow + k0)), wn8, rr);
            if (live2) a2 = rmsnorm8(__ldg(reinterpret_cast<const uint4*>(xrow2 + k0)), wn8, rr2);
          } else {
            if (live) a = __ldg(reinterpret_cast<const uint4*>(xrow + k0));
            if (live2) a2 = __ldg(reinterpret_cast<const uint4*>(xrow2 + k0));
          }
#pragma unroll
          for (int i = 0; i < MT; ++i) {
            const uint4& w = decoded[i];
            mma_bf16(c[z][i], a.x, a2.x, a.y, a2.y, w.x, w.y);
            mma_bf16(c[z][i], a.z, a2.z, a.w, a2.w, w.z, w.w);
          }
        } else {
          uint4 a0 = make_uint4(0, 0, 0, 0), a1 = a0, a2 = a0, a3 = a0;  // row g: a0, a1; row g + 8: a2, a3
          if (live) a0 = __ldg(reinterpret_cast<const uint4*>(xrow + k0)), a1 = __ldg(reinterpret_cast<const uint4*>(xrow + k0 + 8));
          if (live2)
            a2 = __ldg(reinterpret_cast<const uint4*>(xrow2 + k0)), a3 = __ldg(reinterpret_cast<const uint4*>(xrow2 + k0 + 8));
#pragma unroll
          for (int i = 0; i < MT; ++i) {
            uint32_t wv[8];
            widen16(b[u][i], s[u][i], wv);
            mma_bf16(c[z][i], a0.x, a2.x, a0.y, a2.y, wv[0], wv[1]);
            mma_bf16(c[z][i], a0.z, a2.z, a0.w, a2.w, wv[2], wv[3]);
            mma_bf16(c[z][i], a1.x, a3.x, a1.y, a3.y, wv[4], wv[5]);
            mma_bf16(c[z][i], a1.z, a3.z, a1.w, a3.w, wv[6], wv[7]);
          }
        }
      } else {
        constexpr int NV = kB16 ? 8 : 16;  // weight values per lane and step
        float wf[MT][NV];
#pragma unroll
        for (int i = 0; i < MT; ++i) {
          if constexpr (kB16) {
            const uint4& w = decoded[i];
            const uint32_t* v = &w.x;
#pragma unroll
            for (int j = 0; j < 4; ++j) wf[i][2 * j] = bf16_lo(v[j]), wf[i][2 * j + 1] = bf16_hi(v[j]);
          } else {
            uint32_t wv[8];
            widen16(b[u][i], s[u][i], wv);
#pragma unroll
            for (int j = 0; j < 8; ++j) wf[i][2 * j] = bf16_lo(wv[j]), wf[i][2 * j + 1] = bf16_hi(wv[j]);
          }
        }
#pragma unroll
        for (int m = 0; m < RB; ++m) {
          if (m >= M) break;
          const __nv_bfloat16* xr = x + (size_t)m * ldx + xoff + NV * t + k0;
          float xv[NV];
#pragma unroll
          for (int h = 0; h < NV / 8; ++h) {
            const uint4 a = __ldg(reinterpret_cast<const uint4*>(xr + 8 * h));
            const uint32_t* av = &a.x;
#pragma unroll
            for (int j = 0; j < 4; ++j) xv[8 * h + 2 * j] = bf16_lo(av[j]), xv[8 * h + 2 * j + 1] = bf16_hi(av[j]);
          }
#pragma unroll
          for (int i = 0; i < MT; ++i)
#pragma unroll
            for (int j = 0; j < NV; ++j) acc[i][m] = fmaf(xv[j], wf[i][j], acc[i][m]);
        }
      }
      }
    }
  };

  load(ba, sa, p0);
  pdl_launch_dependents();
  pdl_wait();
  if constexpr (kSx) {
    const bool first = blockIdx.x == 0;
    rmsnorm_rows(nin.x, K, nin.w, M, K, nin.eps, red, sx, K + 32, first ? nin.q : nullptr, K, first ? nin.qs : nullptr,
                 nullptr, nullptr);
    __syncthreads();
  } else if constexpr (NORM) {  // the rows' 1/rms (CTA 0: and the act_quant of the normed rows)
    if (blockIdx.x == 0 && nin.q)
      rmsnorm_rows<RB>(nin.x, K, nin.w, M, K, nin.eps, red, nullptr, 0, nin.q, K, nin.qs, nullptr, nullptr);
    else
      rmsnorm_scale<RB>(nin.x, K, M, K, nin.eps, red);
#pragma unroll
    for(int z=0;z<2;++z){rr24[z]=red[384+(g+16*z<M?g+16*z:0)];rr224[z]=red[384+(g+16*z+8<M?g+16*z+8:0)];}
    __syncthreads();  // read before the partial sums reuse red
  }
  for (int p = p0; p < p1; p += 2 * KU) {
    if (p + KU < p1) load(bb, sb, p + KU);
    comp(ba, sa, p);
    if (p + KU >= p1) break;
    if (p + 2 * KU < p1) load(ba, sa, p + 2 * KU);
    comp(bb, sb, p + KU);
  }

#pragma unroll
  for(int z=0;z<2;++z){
  if constexpr (!CC) {
    if (g+16*z<M)
#pragma unroll
      for (int i = 0; i < MT; ++i) {
        const int r = wi_n * 8 * MT + 8 * i + 2 * t;
        red[(wi_k * RB + g + 16*z) * rows_cta + r] = c[z][i][0];
        red[(wi_k * RB + g + 16*z) * rows_cta + r + 1] = c[z][i][1];
      }
    if constexpr (kHi)
      if (g+16*z+8<M)
#pragma unroll
        for (int i = 0; i < MT; ++i) {
          const int r = wi_n * 8 * MT + 8 * i + 2 * t;
          red[(wi_k * RB + g + 16*z + 8) * rows_cta + r] = c[z][i][2];
          red[(wi_k * RB + g + 16*z + 8) * rows_cta + r + 1] = c[z][i][3];
        }
  }
  }
  __syncthreads();
  if (q) {  // bf16 out and its act_quant: one warp per (row, 32 outputs)
    const int nblk = rows_cta / 32;
    for (int pr = warp; pr < M * nblk; pr += blockDim.x >> 5) {
      const int m = pr / nblk, r = 32 * (pr % nblk) + lane;
      float v = 0.f;
      for (int j = 0; j < wk; ++j) v += red[(j * RB + m) * rows_cta + r];
      const __nv_bfloat16 vb = __float2bfloat16_rn(v);
      if (yb) yb[(size_t)m * N + n_cta + r] = vb;
      const float vf = __bfloat162float(vb);
      const uint32_t e = act_scale_byte(warp_max(fabsf(vf)));
      q[(size_t)m * N + n_cta + r] = (uint8_t)quant_e4m3(vf, e);
      if (lane == 0) qs[(size_t)m * (N / 32) + (n_cta + r) / 32] = (uint8_t)e;
    }
    return;
  }
  for (int idx = threadIdx.x; idx < M * rows_cta; idx += blockDim.x) {
    const int m = idx / rows_cta, r = idx - m * rows_cta;
    float v = 0.f;
    for (int j = 0; j < wk; ++j) v += red[(j * RB + m) * rows_cta + r];
    if (yf) yf[(size_t)m * N + n_cta + r] = v;
    else yb[(size_t)m * N + n_cta + r] = __float2bfloat16_rn(v);
  }
}
template<int FMT,int MT,bool NORM>
__global__ void __launch_bounds__(512,1) k_bf24(const __nv_bfloat16* x,int M,const uint8_t* W,const uint8_t* S,int N,int K,int groups,
float* yf,__nv_bfloat16* yb,uint8_t* q,uint8_t* qs,int wn,int wk,NormIn nin){
 gemv16_body24<FMT,MT,false,NORM,24>(x,M,W,S,N,K,groups,yf,yb,q,qs,wn,wk,nin);
}
template<int FMT,int MT,bool NORM>
void launch_bf24(const __nv_bfloat16* x,int M,const uint8_t* W,const uint8_t* S,int N,int K,int groups,float* yf,__nv_bfloat16* yb,
uint8_t* q,uint8_t* qs,GemvCfg cfg,cudaStream_t st,NormIn nin={}){
 const int rc=cfg.wn*8*MT;
 if(M<17||M>24||N%rc||cfg.wn*cfg.wk>16||(q&&rc%32)||(FMT==kWOA&&groups%rc))die("bf24: shape");
 size_t smem=size_t(std::max(cfg.wk*24*rc,408))*4;
 if(smem>48*1024)die("bf24: smem");
 launch_pdl(k_bf24<FMT,MT,NORM>,dim3(N/rc),dim3(32*cfg.wn*cfg.wk),smem,st,cfg.pdl,x,M,W,S,N,K,groups,yf,yb,q,qs,cfg.wn,cfg.wk,nin);
}
