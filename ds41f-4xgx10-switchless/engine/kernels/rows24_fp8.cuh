template <int MT, bool TILED, bool NORM, int RB>
__device__ __forceinline__ void gemv_fp8_body24(const uint8_t* __restrict__ xq, const uint8_t* __restrict__ xs, int M,
                                              const Segs& segs, int K, int wn, int wk, NormIn nin) {
  extern __shared__ float red[];  // [wk][RB][rows_cta] (>= 256 + RB); NORM: then q [M][K + 32], s [M][K/32]
  constexpr bool kHi = RB > kLowRows;
  const int rows_cta = wn * 8 * MT;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int wi_n = warp % wn, wi_k = warp / wn;
  const int r_cta = blockIdx.x * rows_cta;
  int si = 0;
  while (si + 1 < segs.n && r_cta >= segs.row0[si + 1]) ++si;
  const uint8_t* __restrict__ W = segs.s[si].W;
  const uint8_t* __restrict__ S = segs.s[si].S;
  const int N = segs.s[si].N;
  const int n_cta = r_cta - segs.row0[si];
  const int n0 = n_cta + wi_n * 8 * MT;
  const int KB = K / 32;

  // 64-byte steps (two k-blocks): lane (g, t) loads 16 bytes of its row at 16t, then trades half
  // with lane t^2 so that it holds 8 bytes of each block at offset 16(t&1) + 8(t>>1).
  const int P = K / 64;
  const int p0 = (P * wi_k) / wk, p1 = (P * (wi_k + 1)) / wk;
  float c[2][MT][4] = {};
  const uint8_t* wrow[MT];
  const uint8_t* srow[MT];
#pragma unroll
  for (int i = 0; i < MT; ++i) {
    c[0][i][0] = c[0][i][1] = c[0][i][2] = c[0][i][3] = 0.f;
    int n = n0 + 8 * i;
    n = n < N ? n : 0;
    // natural: row-major [N][K]; tiled: [N/8][K/64][8 rows][64 bytes], so a warp's 16-byte loads
    // for one 64-byte step of 8 rows are 512 contiguous bytes
    wrow[i] = TILED ? W + (size_t)(n / 8) * 8 * K + g * 64 + 16 * t : W + (size_t)(n + g) * K + 16 * t;
    srow[i] = S + (size_t)(n / 32) * KB;
  }
  const int off = 16 * (t & 1) + 8 * (t >> 1);
  const bool arow_live = g < M, arow_live2 = kHi && g + 8 < M;
  const int red_n = max(wk * RB * rows_cta, 256 + RB), ldq = NORM ? K + 32 : K;  // +32: no bank conflicts
  uint8_t* sq = reinterpret_cast<uint8_t*>(red + red_n);
  uint8_t* ss = sq + (size_t)M * ldq;
  const uint8_t* arow = (NORM ? sq : xq) + (size_t)(arow_live ? g : 0) * ldq + off;
  const uint8_t* srowa = (NORM ? ss : xs) + (size_t)(arow_live ? g : 0) * KB;
  const bool odd = lane & 1, lo = t < 2;
  // kHi: row g + 8, and the scale row this lane gives the MMA (row g + 8 for odd lanes)
  const uint8_t* arow2 = (NORM ? sq : xq) + (size_t)(arow_live2 ? g + 8 : 0) * ldq + off;
  const uint8_t* srowm = odd ? (NORM ? ss : xs) + (size_t)(arow_live2 ? g + 8 : 0) * KB : srowa;
  const bool slive = odd ? arow_live2 : arow_live;

  // Software pipeline: the weights of the next batch are in flight while this one computes. The
  // first batch is requested before waiting on the previous kernel (PDL), since weights never
  // depend on it.
  // Two register buffers used in turn (the loop is unrolled by two so that both stay in registers).
  uint4 ba[kU][MT], bb[kU][MT];
  auto load_w = [&](uint4 (&b)[kU][MT], int p) {
#pragma unroll
    for (int u = 0; u < kU; ++u)
      if (p + u < p1)
#pragma unroll
        for (int i = 0; i < MT; ++i) b[u][i] = ldg_stream128(wrow[i] + (p + u) * (TILED ? 512 : 64));
  };
  auto comp8 = [&](const uint4 (&b)[kU][MT], int p) {  // up to 8 rows
    uint2 a[kU][2];
    uint32_t sfa[kU][2], sfb[kU][2][MT];
#pragma unroll
    for (int u = 0; u < kU; ++u) {
      if (p + u >= p1) break;
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int kb = 2 * (p + u) + h;
        if constexpr (NORM) {
          a[u][h] = arow_live ? *reinterpret_cast<const uint2*>(arow + kb * 32) : make_uint2(0, 0);
          sfa[u][h] = (odd || !arow_live) ? 127u : srowa[kb];
        } else {
          a[u][h] = arow_live ? __ldg(reinterpret_cast<const uint2*>(arow + kb * 32)) : make_uint2(0, 0);
          sfa[u][h] = (odd || !arow_live) ? 127u : __ldg(srowa + kb);
        }
#pragma unroll
        for (int i = 0; i < MT; ++i) sfb[u][h][i] = __ldg(srow[i] + kb);
      }
    }
#pragma unroll
    for (int u = 0; u < kU; ++u) {
      if (p + u >= p1) break;
#pragma unroll
      for (int i = 0; i < MT; ++i) {
        const uint4 w = b[u][i];
        uint32_t s0 = lo ? w.z : w.x, s1 = lo ? w.w : w.y;
        uint32_t r0 = __shfl_xor_sync(0xffffffffu, s0, 2), r1 = __shfl_xor_sync(0xffffffffu, s1, 2);
        mma_e4m3_e4m3(c[0][i], a[u][0].x, 0u, a[u][0].y, 0u, lo ? w.x : r0, lo ? w.y : r1, sfa[u][0], sfb[u][0][i]);
        mma_e4m3_e4m3(c[0][i], a[u][1].x, 0u, a[u][1].y, 0u, lo ? r0 : w.z, lo ? r1 : w.w, sfa[u][1], sfb[u][1][i]);
      }
    }
  };
  // 9..16 rows: rows g and g + 8 (the odd lanes' scales are row g + 8's), the activations of half the
  // batch's steps at a time
  auto comp16 = [&](const uint4 (&b)[kU][MT], int p) {
#pragma unroll
    for(int z=0;z<2;++z){
      const bool arow_live=g+16*z<M, arow_live2=g+16*z+8<M;
      const uint8_t* arow=xq+size_t(arow_live?g+16*z:0)*ldq+off;
      const uint8_t* arow2=xq+size_t(arow_live2?g+16*z+8:0)*ldq+off;
      const uint8_t* srowm=xs+size_t(odd?(arow_live2?g+16*z+8:0):(arow_live?g+16*z:0))*KB;
      const bool slive=odd?arow_live2:arow_live;

#pragma unroll
    for (int u0 = 0; u0 < kU; u0 += kU / 2) {
      uint2 a[kU / 2][2], a2[kU / 2][2];
      uint32_t sfa[kU / 2][2], sfb[kU / 2][2][MT];
#pragma unroll
      for (int du = 0; du < kU / 2; ++du) {
        const int u = u0 + du;
        if (p + u >= p1) break;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int kb = 2 * (p + u) + h;
          a[du][h] = a2[du][h] = make_uint2(0, 0);
          if constexpr (NORM) {
            if (arow_live) a[du][h] = *reinterpret_cast<const uint2*>(arow + kb * 32);
            if (arow_live2) a2[du][h] = *reinterpret_cast<const uint2*>(arow2 + kb * 32);
            sfa[du][h] = slive ? srowm[kb] : 127u;
          } else {
            if (arow_live) a[du][h] = __ldg(reinterpret_cast<const uint2*>(arow + kb * 32));
            if (arow_live2) a2[du][h] = __ldg(reinterpret_cast<const uint2*>(arow2 + kb * 32));
            sfa[du][h] = slive ? __ldg(srowm + kb) : 127u;
          }
#pragma unroll
          for (int i = 0; i < MT; ++i) sfb[du][h][i] = __ldg(srow[i] + kb);
        }
      }
#pragma unroll
      for (int du = 0; du < kU / 2; ++du) {
        const int u = u0 + du;
        if (p + u >= p1) break;
#pragma unroll
        for (int i = 0; i < MT; ++i) {
          const uint4 w = b[u][i];
          uint32_t s0 = lo ? w.z : w.x, s1 = lo ? w.w : w.y;
          uint32_t r0 = __shfl_xor_sync(0xffffffffu, s0, 2), r1 = __shfl_xor_sync(0xffffffffu, s1, 2);
          mma_e4m3_e4m3(c[z][i], a[du][0].x, a2[du][0].x, a[du][0].y, a2[du][0].y, lo ? w.x : r0, lo ? w.y : r1,
                        sfa[du][0], sfb[du][0][i]);
          mma_e4m3_e4m3(c[z][i], a[du][1].x, a2[du][1].x, a[du][1].y, a2[du][1].y, lo ? r0 : w.z, lo ? r1 : w.w,
                        sfa[du][1], sfb[du][1][i]);
        }
      }
    }
  
    }
};
  auto comp = [&](const uint4 (&b)[kU][MT], int p) {
    if constexpr (kHi) comp16(b, p);
    else comp8(b, p);
  };
  load_w(ba, p0);
  pdl_launch_dependents();
  pdl_wait();
  if constexpr (NORM) {
    const bool first = blockIdx.x == 0;
    rmsnorm_rows<RB>(nin.x, K, nin.w, M, K, nin.eps, red, first ? nin.y : nullptr, K, sq, ldq, ss,
                     first ? nin.q : nullptr, first ? nin.qs : nullptr);
    __syncthreads();
  }
  for (int p = p0; p < p1; p += 2 * kU) {
    if (p + kU < p1) load_w(bb, p + kU);
    comp(ba, p);
    if (p + kU >= p1) break;
    if (p + 2 * kU < p1) load_w(ba, p + 2 * kU);
    comp(bb, p + kU);
  }

#pragma unroll
  for(int z=0;z<2;++z){
  // Partial sums per k-slice -> smem [wk][m][row], then a fixed-order sum over the slices.
#pragma unroll
  for (int i = 0; i < MT; ++i) {
    int r = wi_n * 8 * MT + 8 * i + 2 * t;
    if(g+16*z<M) red[(wi_k * RB + g + 16*z) * rows_cta + r] = c[z][i][0];
    if(g+16*z<M) red[(wi_k * RB + g + 16*z) * rows_cta + r + 1] = c[z][i][1];
    if(g+16*z+8<M) {
      if(g+16*z<M) red[(wi_k * RB + g + 16*z + 8) * rows_cta + r] = c[z][i][2];
      if(g+16*z<M) red[(wi_k * RB + g + 16*z + 8) * rows_cta + r + 1] = c[z][i][3];
    }
  }
  }
  __syncthreads();
  __nv_bfloat16* __restrict__ y = segs.s[si].y;
  for (int idx = threadIdx.x; idx < M * rows_cta; idx += blockDim.x) {
    int m = idx / rows_cta, r = idx - m * rows_cta;
    if (n_cta + r >= N) continue;
    float v = 0.f;
    for (int j = 0; j < wk; ++j) v += red[(j * RB + m) * rows_cta + r];
    y[(size_t)m * N + n_cta + r] = __float2bfloat16_rn(v);
  }
}
template<bool TILED>
__global__ void __launch_bounds__(512,1) k_fp8_24(const uint8_t* q,const uint8_t* sc,int M,Segs seg,int K,int wn,int wk){
  gemv_fp8_body24<1,TILED,false,24>(q,sc,M,seg,K,wn,wk,NormIn{});
}
__global__ void __launch_bounds__(512) k_norm24(const __nv_bfloat16* x,const __nv_bfloat16* w,float eps,int M,int K,
                                              __nv_bfloat16* y,uint8_t* q,uint8_t* sc){
  __shared__ float red[408];
  pdl_launch_dependents();pdl_wait();
  rmsnorm_rows<24>(x,K,w,M,K,eps,red,y,K,q,K,sc,nullptr,nullptr);
}
