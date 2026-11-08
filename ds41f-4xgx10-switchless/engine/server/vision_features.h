#pragma once
#include "common/frontend_protocol.h"
#include <algorithm>
#include <cstring>
#include <map>
#include <memory>
#include <vector>

namespace srv {
// Connection-local, immutable CPU features. Requests pin shared objects even if the
// LRU index evicts them; a reference miss occurs BEFORE a request is admitted.
class VisionFeatures {
 public:
  using Data = std::shared_ptr<const std::vector<uint16_t>>;
  explicit VisionFeatures(size_t limit = size_t(1) << 30) : limit_(limit) {}
  Data get(const fe::ImageSpan& image) {
    auto it = entries_.find(image.digest);
    if (it == entries_.end() || it->second.rows != image.rows) return {};
    it->second.used = ++clock_;
    return it->second.data;
  }
  Data put(const fe::ImageSpan& image, const char* bytes, size_t size) {
    if (!image.rows || image.rows > 1024 || size != size_t(image.rows)*5120*2 || size > limit_) return {};
    auto data = std::make_shared<std::vector<uint16_t>>(size/2);
    std::memcpy(data->data(), bytes, size);
    for (auto value : *data) if ((value & 0x7f80) == 0x7f80) return {};
    auto old = entries_.find(image.digest);
    if (old != entries_.end()) {
      if (old->second.rows != image.rows || *old->second.data != *data) return {};
      old->second.used = ++clock_;
      return old->second.data;
    }
    while (used_ + size > limit_ && !entries_.empty()) {
      auto victim = std::min_element(entries_.begin(), entries_.end(), [](const auto& a,const auto& b){return a.second.used<b.second.used;});
      used_ -= victim->second.data->size()*2;
      entries_.erase(victim);
    }
    entries_.emplace(image.digest, Entry{image.rows, ++clock_, data}); used_ += size;
    return data;
  }
  size_t bytes() const { return used_; }
 private:
  struct Entry { uint32_t rows; uint64_t used; Data data; };
  std::map<std::array<uint8_t,32>,Entry> entries_;
  size_t limit_, used_ = 0;
  uint64_t clock_ = 0;
};
}
