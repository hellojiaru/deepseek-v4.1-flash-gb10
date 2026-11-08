#include "server/vision_features.h"
#include "server/vision_prefix.h"
#include <cassert>
#include <iostream>
int main() {
  fe::ImageSpan a{200000,2,{1}},b{202000,2,{2}},replacement=a;replacement.digest[0]=3;
  const size_t size=2*5120*2;std::vector<char> zero(size,0),other(size,0);other[0]=1;
  srv::VisionFeatures cache(size*2);
  assert(!cache.get(a));
  auto pinned=cache.put(a,zero.data(),size);assert(pinned);
  assert(cache.get(a)==pinned);
  assert(!cache.put(a,other.data(),size)); // cannot rewrite an immutable feature identity
  assert(!cache.put(a,zero.data(),size-2));
  auto bad=zero;bad[0]=char(0x80);bad[1]=char(0x7f);
  assert(!cache.put(b,bad.data(),size)); // inf/NaN never enters the cache
  assert(cache.put(b,zero.data(),size));
  assert(cache.put(replacement,other.data(),size)); // oldest index evicted; pinned request remains valid
  assert(!cache.get(a));assert(pinned->size()==size/2 && (*pinned)[0]==0);
  assert(cache.bytes()==size*2);
  auto wrong=b;wrong.rows=1;assert(!cache.get(wrong));
  srv::VisionFeatures reconnect;assert(!reconnect.get(b)); // connection generation is isolated
  using V=std::vector<fe::ImageSpan>;
  assert(srv::vision_prefix_limit(V{},V{a},true)==200000);
  assert(srv::vision_prefix_limit(V{a},V{},true)==200000);
  assert(srv::vision_prefix_limit(V{},V{a},false)==0); // CED protection remains
  assert(srv::vision_prefix_limit(V{a},V{a,b},true)==202000);
  assert(srv::vision_prefix_limit(V{a,b},V{replacement,b},true)==200000);
  for(int mode=0;mode<2;++mode) for(int n=1;n<64;++n) {
    V old;
    for(int i=0;i<n;++i)old.push_back({uint32_t(200000+64*i),32,{uint8_t(i+1)}});
    auto next=old;next.push_back({uint32_t(200000+64*n),32,{uint8_t(n+1)}});
    assert(srv::vision_prefix_limit(old,next,true)==int(next.back().start));
    for(int j=0;j<n;++j){auto changed=next;changed[j].digest[31]^=1;assert(srv::vision_prefix_limit(old,changed,true)==int(old[j].start));}
  }
  std::cout<<"vision_features_test: references, immutable identity, missing recovery, budget, pinning, reconnect, exact/legacy prefix boundaries passed\n";
}
