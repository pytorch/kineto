// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <stdlib.h>
#include <assert.h>
#include <cstdint>
#include <map>
#include <memory>
#include <sys/types.h>
#include <vector>

#include "ITraceActivity.h"

namespace KINETO_NAMESPACE {

class CuptiActivityBuffer {
 public:
  explicit CuptiActivityBuffer(size_t size) : size_(size) {
    buf_.reserve(size);
  }
  CuptiActivityBuffer() = delete;
  CuptiActivityBuffer& operator=(const CuptiActivityBuffer&) = delete;
  CuptiActivityBuffer(CuptiActivityBuffer&&) = default;
  CuptiActivityBuffer& operator=(CuptiActivityBuffer&&) = default;

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

  std::vector<std::unique_ptr<const ITraceActivity>> wrappers_;
};

using CuptiActivityBufferMap =
    std::map<uint8_t*, std::unique_ptr<CuptiActivityBuffer>>;

} // namespace KINETO_NAMESPACE
