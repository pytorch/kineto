// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once


#include <list>
#include <memory>

#include "libkineto.h"
#include "CuptiActivityBuffer.h"

namespace KINETO_NAMESPACE {

struct ActivityBuffers {
  std::list<std::unique_ptr<libkineto::CpuTraceBuffer>> cpu;
  std::unique_ptr<CuptiActivityBufferMap> gpu;

  // Add a wrapper object to the underlying struct stored in the buffer
  template<class T>
  const ITraceActivity& addActivityWrapper(const T& act) {
    wrappers_.push_back(std::make_unique<T>(act));
    return *wrappers_.back().get();
  }

 private:
  std::vector<std::unique_ptr<const ITraceActivity>> wrappers_;
};

} // namespace KINETO_NAMESPACE
