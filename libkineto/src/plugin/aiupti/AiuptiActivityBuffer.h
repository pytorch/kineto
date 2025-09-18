#pragma once

#include "AiuptiProfilerMacros.h"

#include <assert.h>
#include <stdlib.h>
#include <deque>
#include <map>
#include <memory>
#include <vector>

namespace KINETO_NAMESPACE {

class AiuptiActivityBuffer {
 public:
  explicit AiuptiActivityBuffer(size_t size) : size_(size) {
    buf_.reserve(size);
  }
  AiuptiActivityBuffer() = delete;
  AiuptiActivityBuffer& operator=(const AiuptiActivityBuffer&) = delete;
  AiuptiActivityBuffer(AiuptiActivityBuffer&&) = default;
  AiuptiActivityBuffer& operator=(AiuptiActivityBuffer&&) = default;

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

using AiuptiActivityBufferDeque =
    std::deque<std::pair<uint8_t*, std::unique_ptr<AiuptiActivityBuffer>>>;
} // namespace KINETO_NAMESPACE
