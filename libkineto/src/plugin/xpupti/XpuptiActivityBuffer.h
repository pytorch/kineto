#pragma once

#include "XpuptiProfilerMacros.h"

#include <assert.h>
#include <stdlib.h>
#include <map>
#include <memory>
#include <vector>

namespace KINETO_NAMESPACE {

class XpuptiActivityBuffer {
 public:
  explicit XpuptiActivityBuffer(size_t size) : size_(size) {
    buf_.reserve(size);
  }
  XpuptiActivityBuffer() = delete;
  XpuptiActivityBuffer& operator=(const XpuptiActivityBuffer&) = delete;
  XpuptiActivityBuffer(XpuptiActivityBuffer&&) = default;
  XpuptiActivityBuffer& operator=(XpuptiActivityBuffer&&) = default;

  size_t size() const {
    return size_;
  }

  void setSize(size_t size) {
    assert(size <= buf_.capacity());
    size_ = size;
  }

  uint8_t* data() {
    return buf_.data();
  }

 private:
  std::vector<uint8_t> buf_;
  size_t size_;
};

using XpuptiActivityBufferMap =
    std::map<uint8_t*, std::unique_ptr<XpuptiActivityBuffer>>;

} // namespace KINETO_NAMESPACE
