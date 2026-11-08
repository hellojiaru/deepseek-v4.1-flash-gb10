template <int NT, int WK, int KU, bool CS = false>
__global__ void __launch_bounds__(32 * (4 / NT) * WK, 1) k_moe_up_r24(Params a) {
  static_assert(!CS || (kP1 % WK == 0 && kP1 / WK <= 32 && 2 * KU < 32), "no scale-code build");
  constexpr int RB = 24, WN = 4 / NT, KH = KU / 2 > 0 ? KU / 2 : 1;
  __shared__ Union<RB> U;
  __shared__ float red[WK][RB][64];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int wi_n = warp % WN, wi_k = warp / WN;
  const int p0 = kP1 * wi_k / WK, p1 = kP1 * (wi_k + 1) / WK;
  const int M = a.M;
  const bool live = g < M, live2 = g + 8 < M, odd = lane & 1, lo = t < 2, slive = odd ? live2 : live;
  const size_t r1 = live ? g : 0, r2 = live2 ? g + 8 : 0;
  const uint8_t* arow4 = a.xq + r1 * kDim + 8 * t;
  const uint8_t* arow4_2 = a.xq + r2 * kDim + 8 * t;
  const uint8_t* arow8 = a.xq + r1 * kDim + 16 * (t & 1) + 8 * (t >> 1);
  const uint8_t* arow8_2 = a.xq + r2 * kDim + 16 * (t & 1) + 8 * (t >> 1);
  const uint8_t* srowm = a.xs + (odd ? r2 : r1) * kKB1;

  // shared expert (FP8): this warp's tiles of task jb
  uint4 c8a[KH][NT][2], c8b[KH][NT][2];
  uint32_t s8a[KH][NT], s8b[KH][NT];
  const uint8_t* w8[NT][2];
  const uint8_t* sr8[NT];
  auto setup8 = [&](int jb) {
#pragma unroll
    for (int i = 0; i < NT; ++i) {
      const int ti = wi_n * NT + i, tile = 2 * jb + (ti & 1);
      const uint8_t* W = ti < 2 ? a.w.sh_w1 : a.w.sh_w3;
      w8[i][0] = W + (size_t)(2 * tile) * 8 * kDim + g * 64 + 16 * t;
      w8[i][1] = w8[i][0] + 8 * kDim;
      sr8[i] = (ti < 2 ? a.w.sh_s1 : a.w.sh_s3) + (size_t)jb * kKB1;
    }
  };
  auto load8 = [&](uint4 (&b)[KH][NT][2], uint32_t (&s)[KH][NT], int p) {
#pragma unroll
    for (int u = 0; u < KH; ++u)
      if (p + u < p1)
#pragma unroll
        for (int i = 0; i < NT; ++i) {
          b[u][i][0] = ldg_stream128(w8[i][0] + (p + u) * 512);
          b[u][i][1] = ldg_stream128(w8[i][1] + (p + u) * 512);
          s[u][i] = __ldg(reinterpret_cast<const uint16_t*>(sr8[i] + 2 * (p + u)));
        }
  };
  float c[2][NT][2][4];
  auto comp8 = [&](const uint4 (&b)[KH][NT][2], const uint32_t (&s)[KH][NT], int p) {
#pragma unroll
    for(int z=0;z<2;++z){
      const bool live=g+16*z<M,live2=g+16*z+8<M,slive=odd?live2:live;
      const size_t r1=live?g+16*z:0,r2=live2?g+16*z+8:0;
      const uint8_t* arow4=a.xq+r1*kDim+8*t;
      const uint8_t* arow4_2=a.xq+r2*kDim+8*t;
      const uint8_t* arow8=a.xq+r1*kDim+16*(t&1)+8*(t>>1);
      const uint8_t* arow8_2=a.xq+r2*kDim+16*(t&1)+8*(t>>1);
      const uint8_t* srowm=a.xs+(odd?r2:r1)*kKB1;

#pragma unroll
    for (int u = 0; u < KH; ++u)
      if (p + u < p1) {
        uint2 av[2], av2[2];
        uint32_t sfa[2];
        act_frag2(av, av2, sfa, arow8, arow8_2, srowm, p + u, live, live2, slive);
#pragma unroll
        for (int i = 0; i < NT; ++i) {
          mma8h(c[z][i][0], av, av2, sfa, b[u][i][0], s[u][i], lo);
          mma8h(c[z][i][1], av, av2, sfa, b[u][i][1], s[u][i], lo);
        }
      }
  
    }
};

  int task = blockIdx.x;
  bool pre = task < kJB;  // the first task is the shared expert's: its weights need nothing upstream
  if (pre) {
    setup8(task);
    load8(c8a, s8a, p0);
  }
  pdl_wait();
  build_union(U, a, M);
  if (a.logits && a.idx_out && blockIdx.x == 0 && threadIdx.x < M * kTopK)
    a.idx_out[threadIdx.x] = U.sel_id[threadIdx.x], a.wts_out[threadIdx.x] = U.sel_w[threadIdx.x];
  pdl_launch_dependents();
  const int ntask = kJB * (1 + U.n);

  __shared__ int s_next[2];  // k_moe_up's schedule
#pragma unroll 1
  for (int it = 0; task < ntask; ++it) {
    if (a.ctr && threadIdx.x == 0) s_next[it & 1] = (int)gridDim.x + atomicAdd(a.ctr, 1);
#pragma unroll
    for(int z=0;z<2;++z)
#pragma unroll
    for (int i = 0; i < NT; ++i)
#pragma unroll
      for (int h = 0; h < 2; ++h) c[z][i][h][0] = c[z][i][h][1] = c[z][i][h][2] = c[z][i][h][3] = 0.f;
    const bool sh = task < kJB;
    const int u = sh ? 0 : task / kJB - 1, jb = task % kJB;
    if (sh) {
      if (!pre) {
        setup8(jb);
        load8(c8a, s8a, p0);
      }
      for (int p = p0; p < p1; p += 2 * KH) {
        if (p + KH < p1) load8(c8b, s8b, p + KH);
        comp8(c8a, s8a, p);
        if (p + KH >= p1) break;
        if (p + 2 * KH < p1) load8(c8a, s8a, p + 2 * KH);
        comp8(c8b, s8b, p + KH);
      }
    } else {
      const int e = U.id[u];
      const uint8_t* w4[NT];
      const uint8_t* sr4[NT];
#pragma unroll
      for (int i = 0; i < NT; ++i) {
        const int ti = wi_n * NT + i, tile = 2 * jb + (ti & 1);
        w4[i] = (ti < 2 ? a.w.w1 : a.w.w3) + (size_t)e * kW13 + (size_t)tile * (kP1 * 512) + lane * 16;
        sr4[i] = (ti < 2 ? a.w.s1 : a.w.s3) + (size_t)e * kS13 + (size_t)tile * (kP1 * 32) + g * 4;
      }
      uint4 ba[KU][NT], bb[KU][NT];
      uint32_t sa[KU][NT], sb[KU][NT];
      auto load4 = [&](uint4 (&b)[KU][NT], uint32_t (&s)[KU][NT], int p) {
#pragma unroll
        for (int q = 0; q < KU; ++q)
          if (p + q < p1)
#pragma unroll
            for (int i = 0; i < NT; ++i) {
              b[q][i] = ldg_stream128(w4[i] + (p + q) * 512);
              s[q][i] = ldg_stream32(sr4[i] + (p + q) * 32);
            }
      };
      auto comp4 = [&](const uint4 (&b)[KU][NT], const uint32_t (&s)[KU][NT], int p) {
#pragma unroll
    for(int z=0;z<2;++z){
      const bool live=g+16*z<M,live2=g+16*z+8<M,slive=odd?live2:live;
      const size_t r1=live?g+16*z:0,r2=live2?g+16*z+8:0;
      const uint8_t* arow4=a.xq+r1*kDim+8*t;
      const uint8_t* arow4_2=a.xq+r2*kDim+8*t;
      const uint8_t* arow8=a.xq+r1*kDim+16*(t&1)+8*(t>>1);
      const uint8_t* arow8_2=a.xq+r2*kDim+16*(t&1)+8*(t>>1);
      const uint8_t* srowm=a.xs+(odd?r2:r1)*kKB1;

#pragma unroll
        for (int q = 0; q < KU; ++q)
          if (p + q < p1) {
            uint2 av[2], av2[2];
            uint32_t sfa[2];
            act_frag2(av, av2, sfa, arow4, arow4_2, srowm, p + q, live, live2, slive);
#pragma unroll
            for (int i = 0; i < NT; ++i) {
              const uint4 w = b[q][i];
              const uint32_t sc = s[q][i];
              mma4h(c[z][i][0], av[0], av2[0], w.x, sfa[0], sc & 0xFFu);
              mma4h(c[z][i][0], av[1], av2[1], w.y, sfa[1], (sc >> 8) & 0xFFu);
              mma4h(c[z][i][1], av[0], av2[0], w.z, sfa[0], (sc >> 16) & 0xFFu);
              mma4h(c[z][i][1], av[1], av2[1], w.w, sfa[1], sc >> 24);
            }
          }
      
    }
};
      if constexpr (CS) {  // k_moe_up's
        const uint32_t* cw[NT];
        uint32_t fl[NT], b4[NT];
#pragma unroll
        for (int i = 0; i < NT; ++i) {
          const int ti = wi_n * NT + i, tile = 2 * jb + (ti & 1);
          const ScaleCode& k = ti < 2 ? a.w.c1 : a.w.c3;
          cw[i] = k.code + ((size_t)e * kT13 + tile) * kB13;
          b4[i] = k.base4;
          fl[i] = code_flags_up(cw[i], p0);
        }
        auto load4c = [&](uint4 (&b)[KU][NT], uint32_t (&s)[KU][NT], int p, const uint32_t (&f)[NT]) {
#pragma unroll
          for (int q = 0; q < KU; ++q)
            if (p + q < p1)
#pragma unroll
              for (int i = 0; i < NT; ++i) b[q][i] = ldg_stream128(w4[i] + (p + q) * 512);
#pragma unroll
          for (int q = 0; q < KU; ++q)
            if (p + q < p1)
#pragma unroll
              for (int i = 0; i < NT; ++i)
                s[q][i] = ldg_stream32_sel(f[i] >> (p - p0), 1u << q, sr4[i] + (p + q) * 32, cw[i] + p + q);
        };
        auto comp4c = [&](const uint4 (&b)[KU][NT], const uint32_t (&s)[KU][NT], int p) {
#pragma unroll
    for(int z=0;z<2;++z){
      const bool live=g+16*z<M,live2=g+16*z+8<M,slive=odd?live2:live;
      const size_t r1=live?g+16*z:0,r2=live2?g+16*z+8:0;
      const uint8_t* arow4=a.xq+r1*kDim+8*t;
      const uint8_t* arow4_2=a.xq+r2*kDim+8*t;
      const uint8_t* arow8=a.xq+r1*kDim+16*(t&1)+8*(t>>1);
      const uint8_t* arow8_2=a.xq+r2*kDim+16*(t&1)+8*(t>>1);
      const uint8_t* srowm=a.xs+(odd?r2:r1)*kKB1;

#pragma unroll
          for (int q = 0; q < KU; ++q)
            if (p + q < p1) {
              uint2 av[2], av2[2];
              uint32_t sfa[2];
              act_frag2(av, av2, sfa, arow4, arow4_2, srowm, p + q, live, live2, slive);
#pragma unroll
              for (int i = 0; i < NT; ++i) {
                const uint4 w = b[q][i];
                const uint32_t sc = fl[i] >> (p + q - p0) & 1u ? s[q][i] : ((s[q][i] >> g) & 0x01010101u) + b4[i];
                mma4h(c[z][i][0], av[0], av2[0], w.x, sfa[0], sc & 0xFFu);
                mma4h(c[z][i][0], av[1], av2[1], w.y, sfa[1], (sc >> 8) & 0xFFu);
                mma4h(c[z][i][1], av[0], av2[0], w.z, sfa[0], (sc >> 16) & 0xFFu);
                mma4h(c[z][i][1], av[1], av2[1], w.w, sfa[1], sc >> 24);
              }
            }
        
    }
};
        const uint32_t none[NT] = {};
        load4c(ba, sa, p0, none);
        if (p0 + KU < p1) load4c(bb, sb, p0 + KU, none);
#pragma unroll
        for (int i = 0; i < NT; ++i)
          if (fl[i] & ((1u << 2 * KU) - 1u))
#pragma unroll
            for (int q = 0; q < 2 * KU; ++q)
              if (p0 + q < p1 && (fl[i] >> q & 1u)) {
                const uint32_t v = ldg_stream32(sr4[i] + (p0 + q) * 32);
                if (q < KU) sa[q][i] = v;
                else sb[q - KU][i] = v;
              }
        for (int p = p0; p < p1; p += 2 * KU) {
          comp4c(ba, sa, p);
          if (p + KU >= p1) break;
          if (p + 2 * KU < p1) load4c(ba, sa, p + 2 * KU, fl);
          comp4c(bb, sb, p + KU);
          if (p + 3 * KU < p1) load4c(bb, sb, p + 3 * KU, fl);
        }
      } else {
        load4(ba, sa, p0);
        for (int p = p0; p < p1; p += 2 * KU) {
          if (p + KU < p1) load4(bb, sb, p + KU);
          comp4(ba, sa, p);
          if (p + KU >= p1) break;
          if (p + 2 * KU < p1) load4(ba, sa, p + 2 * KU);
          comp4(bb, sb, p + KU);
        }
      }
    }
    pre = false;

#pragma unroll
    for(int z=0;z<2;++z){
    // partials -> smem (rows g and g + 8); rows 0..31 of the task are w1 (gate), 32..63 w3 (up)
#pragma unroll
    for (int i = 0; i < NT; ++i)
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int r = 16 * (wi_n * NT + i) + 8 * h + 2 * t;
        if(g+16*z<M) red[wi_k][g+16*z][r] = c[z][i][h][0], red[wi_k][g+16*z][r + 1] = c[z][i][h][1];
        if(g+16*z+8<M) red[wi_k][g+16*z+8][r] = c[z][i][h][2], red[wi_k][g+16*z+8][r + 1] = c[z][i][h][3];
      }
    }
    __syncthreads();
    for (int m = warp; m < M; m += WN * WK) {
      if (!sh && !(U.rows[u] >> m & 1)) continue;
      float gs = 0.f, us = 0.f;
#pragma unroll
      for (int k = 0; k < WK; ++k) gs += red[k][m][lane], us += red[k][m][32 + lane];
      float gt = __bfloat162float(__float2bfloat16_rn(gs)), up = __bfloat162float(__float2bfloat16_rn(us));
      up = fminf(fmaxf(up, -10.f), 10.f);
      gt = fminf(gt, 10.f);
      float h = gt / (1.0f + expf(-gt)) * up;
      if (!sh) h = U.rw[u][m] * h;
      h = __bfloat162float(__float2bfloat16_rn(h));
      const uint32_t e = act_scale_byte(warp_max(fabsf(h)));
      const int slot = sh ? RB * kTopK : u;
      a.hq[((size_t)slot * RB + m) * kInter + 32 * jb + lane] = (uint8_t)quant_e4m3(h, e);
      if (lane == 0) a.hs[((size_t)slot * RB + m) * kJB + jb] = (uint8_t)e;
    }
    __syncthreads();
    task = a.ctr ? s_next[it & 1] : task + (int)gridDim.x;
  }
  if (a.ctr && threadIdx.x == 0) {
    __threadfence();
    if (atomicAdd(a.ctr + 1, 1) == (int)gridDim.x - 1) atomicExch(a.ctr, 0), atomicExch(a.ctr + 1, 0);
  }
}
template <int W, int R, bool CS = false>
__global__ void __launch_bounds__(32 * W, 1) k_moe_down_r24(Params a) {
  constexpr int RB = 24, kOut = (RB * 16 * R + 32 * W - 1) / (32 * W);  // outputs per thread
  static_assert(W % R == 0, "a round holds whole experts");
  __shared__ __nv_bfloat16 outs[2][W][RB][16];  // a round's outputs: [its units][row][the unit's tile's columns]
  __shared__ Union<RB> U;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int M = a.M;
  const bool live = g < M, live2 = g + 8 < M, odd = lane & 1, lo = t < 2, slive = odd ? live2 : live;
  const int tile0 = blockIdx.x * R;
  build_union(U, a, M);  // the router (gate) finished before moe_up let this grid start
  const int nr = U.n * R, nunits = nr + 2 * R, nround = (nunits + W - 1) / W;
  __shared__ uint32_t sfl[CS ? RB * kTopK * R : 1];  // CS: k_moe_down's flags
  if constexpr (CS) {
#pragma unroll 1
    for (int q = warp + W * lane; q < nr; q += 32 * W)
      sfl[q] = __ldg(a.w.c2.code + ((size_t)U.id[q / R] * kT2 + tile0 + q % R) * kB2 + kP2);
    __syncwarp();
  }

  auto load = [&](uint4 (&b)[kP2], uint32_t (&s)[kP2], int q) {
    if (q < nr) {
      const int u = q / R, tile = tile0 + q % R;
      const size_t e = U.id[u];
      const uint8_t* w = a.w.w2 + e * kW2 + (size_t)tile * (kP2 * 512) + lane * 16;
      const uint8_t* sr = a.w.s2 + e * kS2 + (size_t)tile * (kP2 * 32) + g * 4;
#pragma unroll
      for (int p = 0; p < kP2; ++p) b[p] = ldg_stream128(w + p * 512), s[p] = ldg_stream32(sr + p * 32);
    } else {
      const int qq = q - nr, tile = tile0 + qq / 2, nt = 2 * tile + (qq & 1);
      const uint8_t* w = a.w.sh_w2 + (size_t)nt * 8 * kInter + g * 64 + 16 * t;
      const uint8_t* sr = a.w.sh_s2 + (size_t)(tile / 2) * kKB2;
#pragma unroll
      for (int p = 0; p < kP2; ++p)
        b[p] = ldg_stream128(w + p * 512), s[p] = __ldg(reinterpret_cast<const uint16_t*>(sr + 2 * p));
    }
  };
  // unit q's outputs (bf16) into o [row][column of its tile]
  auto comp = [&](const uint4 (&b)[kP2], const uint32_t (&s)[kP2], int q, __nv_bfloat16 (&o)[RB][16]) {
#pragma unroll
    for(int z=0;z<2;++z){
      const bool live=g+16*z<M,live2=g+16*z+8<M,slive=odd?live2:live;

    float c[2][4] = {};
    const bool sh = q >= nr;
    const int slot = sh ? RB * kTopK : q / R;
    const uint8_t* base = a.hq + (size_t)slot * RB * kInter + (sh ? 16 * (t & 1) + 8 * (t >> 1) : 8 * t);
    const uint8_t* arow = base + (size_t)(live ? g+16*z : 0) * kInter;
    const uint8_t* arow2 = base + (size_t)(live2 ? g+16*z+8 : 0) * kInter;
    const uint8_t* srowm = a.hs + ((size_t)slot * RB + (odd ? (live2 ? g+16*z+8 : 0) : (live ? g+16*z : 0))) * kJB;
    if (!sh) {
#pragma unroll
      for (int p = 0; p < kP2; ++p) {
        uint2 av[2], av2[2];
        uint32_t sfa[2];
        act_frag2(av, av2, sfa, arow, arow2, srowm, p, live, live2, slive);
        mma4h(c[0], av[0], av2[0], b[p].x, sfa[0], s[p] & 0xFFu);
        mma4h(c[0], av[1], av2[1], b[p].y, sfa[1], (s[p] >> 8) & 0xFFu);
        mma4h(c[1], av[0], av2[0], b[p].z, sfa[0], (s[p] >> 16) & 0xFFu);
        mma4h(c[1], av[1], av2[1], b[p].w, sfa[1], s[p] >> 24);
      }
    } else {
#pragma unroll
      for (int p = 0; p < kP2; ++p) {
        uint2 av[2], av2[2];
        uint32_t sfa[2];
        act_frag2(av, av2, sfa, arow, arow2, srowm, p, live, live2, slive);
        mma8h(c[0], av, av2, sfa, b[p], s[p], lo);
      }
    }
    const int col = sh ? 8 * ((q - nr) & 1) + 2 * t : 2 * t;  // the shared expert: one 8-row half per unit
    if (live) {
      o[g+16*z][col] = __float2bfloat16_rn(c[0][0]), o[g+16*z][col + 1] = __float2bfloat16_rn(c[0][1]);
      if (!sh) o[g+16*z][col + 8] = __float2bfloat16_rn(c[1][0]), o[g+16*z][col + 9] = __float2bfloat16_rn(c[1][1]);
    }
    if (live2) {
      o[g+16*z+8][col] = __float2bfloat16_rn(c[0][2]), o[g+16*z+8][col + 1] = __float2bfloat16_rn(c[0][3]);
      if (!sh) o[g+16*z+8][col + 8] = __float2bfloat16_rn(c[1][2]), o[g+16*z+8][col + 9] = __float2bfloat16_rn(c[1][3]);
    }
  
    }
};
  auto load_cs = [&](uint4 (&b)[kP2], uint32_t& x, uint32_t& y, int q) {
    if (q < nr) {
      const int u = q / R, tile = tile0 + q % R;
      const size_t e = U.id[u];
      const uint8_t* w = a.w.w2 + e * kW2 + (size_t)tile * (kP2 * 512) + lane * 16;
#pragma unroll
      for (int p = 0; p < kP2; ++p) b[p] = ldg_stream128(w + p * 512);
      code_load_down(a.w.c2.code + (e * kT2 + tile) * kB2, a.w.s2 + e * kS2 + (size_t)tile * (kP2 * 32), sfl[q], lane,
                     x, y);
    } else {
      const int qq = q - nr, tile = tile0 + qq / 2, nt = 2 * tile + (qq & 1);
      const uint8_t* w = a.w.sh_w2 + (size_t)nt * 8 * kInter + g * 64 + 16 * t;
#pragma unroll
      for (int p = 0; p < kP2; ++p) b[p] = ldg_stream128(w + p * 512);
      x = __ldg(reinterpret_cast<const uint16_t*>(a.w.sh_s2 + (size_t)(tile / 2) * kKB2) + min(lane, kP2 - 1));
    }
  };
  auto comp_cs = [&](const uint4 (&b)[kP2], uint32_t x, uint32_t y, int q, __nv_bfloat16 (&o)[RB][16]) {
#pragma unroll
    for(int z=0;z<2;++z){
      const bool live=g+16*z<M,live2=g+16*z+8<M,slive=odd?live2:live;

    float c[2][4] = {};
    const bool sh = q >= nr;
    const int slot = sh ? RB * kTopK : q / R;
    const uint8_t* base = a.hq + (size_t)slot * RB * kInter + (sh ? 16 * (t & 1) + 8 * (t >> 1) : 8 * t);
    const uint8_t* arow = base + (size_t)(live ? g+16*z : 0) * kInter;
    const uint8_t* arow2 = base + (size_t)(live2 ? g+16*z+8 : 0) * kInter;
    const uint8_t* srowm = a.hs + ((size_t)slot * RB + (odd ? (live2 ? g+16*z+8 : 0) : (live ? g+16*z : 0))) * kJB;
    if (!sh) {
      code_steps_down(sfl[q], x, y, g, a.w.c2.base4,
                      a.w.s2 + (size_t)U.id[q / R] * kS2 + (size_t)(tile0 + q % R) * (kP2 * 32), [&](int p, uint32_t sc) {
                        uint2 av[2], av2[2];
                        uint32_t sfa[2];
                        act_frag2(av, av2, sfa, arow, arow2, srowm, p, live, live2, slive);
                        mma4h(c[0], av[0], av2[0], b[p].x, sfa[0], sc & 0xFFu);
                        mma4h(c[0], av[1], av2[1], b[p].y, sfa[1], (sc >> 8) & 0xFFu);
                        mma4h(c[1], av[0], av2[0], b[p].z, sfa[0], (sc >> 16) & 0xFFu);
                        mma4h(c[1], av[1], av2[1], b[p].w, sfa[1], sc >> 24);
                      });
    } else {
#pragma unroll
      for (int p = 0; p < kP2; ++p) {
        uint2 av[2], av2[2];
        uint32_t sfa[2];
        act_frag2(av, av2, sfa, arow, arow2, srowm, p, live, live2, slive);
        mma8h(c[0], av, av2, sfa, b[p], __shfl_sync(~0u, x, p), lo);
      }
    }
    const int col = sh ? 8 * ((q - nr) & 1) + 2 * t : 2 * t;
    if (live) {
      o[g+16*z][col] = __float2bfloat16_rn(c[0][0]), o[g+16*z][col + 1] = __float2bfloat16_rn(c[0][1]);
      if (!sh) o[g+16*z][col + 8] = __float2bfloat16_rn(c[1][0]), o[g+16*z][col + 9] = __float2bfloat16_rn(c[1][1]);
    }
    if (live2) {
      o[g+16*z+8][col] = __float2bfloat16_rn(c[0][2]), o[g+16*z+8][col + 1] = __float2bfloat16_rn(c[0][3]);
      if (!sh) o[g+16*z+8][col + 8] = __float2bfloat16_rn(c[1][2]), o[g+16*z+8][col + 9] = __float2bfloat16_rn(c[1][3]);
    }
  
    }
};
  // round k's units folded into this thread's sums: per output, its tile's routed units of the round in
  // order (rows that picked the expert), then the shared expert's unit if it is in the round
  float v[kOut];
#pragma unroll
  for (int j = 0; j < kOut; ++j) v[j] = 0.f;
  auto fold = [&](int k, const __nv_bfloat16 (&ob)[W][RB][16]) {
#pragma unroll
    for (int j = 0; j < kOut; ++j) {
      const int o = threadIdx.x + 32 * W * j;
      if (o >= M * 16 * R) break;
      const int m = o / (16 * R), col = o % (16 * R), tile = col / 16, c16 = col % 16;
      float x = v[j];
      for (int jj = tile; jj < W; jj += R) {
        const int q = k * W + jj;
        if (q >= nr) break;
        if (U.rows[q / R] >> m & 1) x += __bfloat162float(ob[jj][m][c16]);
      }
      const int js = nr + 2 * tile + (c16 >> 3) - k * W;
      if (js >= 0 && js < W) x += __bfloat162float(ob[js][m][c16]);
      v[j] = x;
    }
  };

  uint4 ba[kP2], bb[kP2];
  int q = warp;
  if constexpr (CS) {
    uint32_t xa = 0, ya = 0, xb = 0, yb = 0;
    if (q < nunits) load_cs(ba, xa, ya, q);
    pdl_wait();
    pdl_launch_dependents();
    for (int k = 0; k < nround; k += 2, q += 2 * W) {
      if (q + W < nunits) load_cs(bb, xb, yb, q + W);
      if (q < nunits) comp_cs(ba, xa, ya, q, outs[0][warp]);
      __syncthreads();
      fold(k, outs[0]);
      if (k + 1 >= nround) break;
      if (q + 2 * W < nunits) load_cs(ba, xa, ya, q + 2 * W);
      if (q + W < nunits) comp_cs(bb, xb, yb, q + W, outs[1][warp]);
      __syncthreads();
      fold(k + 1, outs[1]);
    }
  } else {
    uint32_t sa[kP2], sb[kP2];
    if (q < nunits) load(ba, sa, q);
    pdl_wait();
    pdl_launch_dependents();
    for (int k = 0; k < nround; k += 2, q += 2 * W) {
      if (q + W < nunits) load(bb, sb, q + W);
      if (q < nunits) comp(ba, sa, q, outs[0][warp]);
      __syncthreads();
      fold(k, outs[0]);
      if (k + 1 >= nround) break;
      if (q + 2 * W < nunits) load(ba, sa, q + 2 * W);
      if (q + W < nunits) comp(bb, sb, q + W, outs[1][warp]);
      __syncthreads();  // also: every thread's fold of round k is done before round k + 2 rewrites outs[0]
      fold(k + 1, outs[1]);
    }
  }
#pragma unroll
  for (int j = 0; j < kOut; ++j) {
    const int o = threadIdx.x + 32 * W * j;
    if (o >= M * 16 * R) break;
    a.y[(size_t)(o / (16 * R)) * kDim + 16 * tile0 + o % (16 * R)] = v[j];
  }
}