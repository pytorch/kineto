/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <atomic>
#include <deque>
#include <functional>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <unordered_map>
#include <vector>

#include <hip/hip_runtime_api.h>


namespace libkineto {
class RoctracerActivityApi;
}

typedef uint64_t timestamp_t;

static timestamp_t timespec_to_ns(const timespec& time) {
  return ((timestamp_t)time.tv_sec * 1000000000) + time.tv_nsec;
}

namespace RocLogger {
enum CorrelationDomain {
  begin,
  Default = begin,
  Domain0 = begin,
  Domain1,
  end,
  size = end
};
} // namespace RocLogger

class ApiIdList {
 public:
  ApiIdList();
  bool invertMode() {
    return invert_;
  }
  void setInvertMode(bool invert) {
    invert_ = invert;
  }
  void add(const std::string& apiName);
  void remove(const std::string& apiName);
  bool loadUserPrefs();

  // Map api string to cnid enum
  virtual uint32_t mapName(const std::string &apiName) = 0;

  bool contains(uint32_t apiId);
  const std::unordered_map<uint32_t, uint32_t>& filterList() {
    return filter_;
  }

 private:
  std::unordered_map<uint32_t, uint32_t> filter_;
  bool invert_;
};

typedef enum {
  ROCTRACER_ACTIVITY_DEFAULT = 0,
  ROCTRACER_ACTIVITY_KERNEL,
  ROCTRACER_ACTIVITY_COPY,
  ROCTRACER_ACTIVITY_MALLOC,
  ROCTRACER_ACTIVITY_ASYNC,
  ROCTRACER_ACTIVITY_NONE
} roctracer_activity_types;

struct roctracerBase {
  roctracerBase(
      uint64_t id,
      uint32_t domain,
      uint64_t begin,
      uint64_t end,
      roctracer_activity_types type = ROCTRACER_ACTIVITY_NONE)
      : id(id), begin(begin), end(end), domain(domain), type(type) {}
  uint64_t id; // correlation_id
  uint64_t begin;
  uint64_t end;
  uint32_t domain;
  roctracer_activity_types type;
};

struct roctracerRow : public roctracerBase {
  roctracerRow(
      uint64_t id,
      uint32_t domain,
      uint32_t cid,
      uint32_t pid,
      uint32_t tid,
      uint64_t begin,
      uint64_t end,
      roctracer_activity_types type = ROCTRACER_ACTIVITY_DEFAULT)
      : roctracerBase(id, domain, begin, end, type),
        cid(cid),
        pid(pid),
        tid(tid) {}
  uint32_t cid;
  uint32_t pid;
  uint32_t tid;
};

struct roctracerKernelRow : public roctracerRow {
  roctracerKernelRow(
      uint64_t id,
      uint32_t domain,
      uint32_t cid,
      uint32_t pid,
      uint32_t tid,
      uint64_t begin,
      uint64_t end,
      const void* faddr,
      hipFunction_t function,
      unsigned int gx,
      unsigned int gy,
      unsigned int gz,
      unsigned int wx,
      unsigned int wy,
      unsigned int wz,
      size_t gss,
      hipStream_t stream,
      roctracer_activity_types type = ROCTRACER_ACTIVITY_KERNEL)
      : roctracerRow(id, domain, cid, pid, tid, begin, end, type),
        functionAddr(faddr),
        function(function),
        gridX(gx),
        gridY(gy),
        gridZ(gz),
        workgroupX(wx),
        workgroupY(wy),
        workgroupZ(wz),
        groupSegmentSize(gss),
        stream(stream) {}
  const void* functionAddr;
  hipFunction_t function;
  unsigned int gridX;
  unsigned int gridY;
  unsigned int gridZ;
  unsigned int workgroupX;
  unsigned int workgroupY;
  unsigned int workgroupZ;
  size_t groupSegmentSize;
  hipStream_t stream;
};

struct roctracerCopyRow : public roctracerRow {
  roctracerCopyRow(
      uint64_t id,
      uint32_t domain,
      uint32_t cid,
      uint32_t pid,
      uint32_t tid,
      uint64_t begin,
      uint64_t end,
      const void* src,
      const void* dst,
      size_t size,
      hipMemcpyKind kind,
      hipStream_t stream,
      roctracer_activity_types type = ROCTRACER_ACTIVITY_COPY)
      : roctracerRow(id, domain, cid, pid, tid, begin, end, type),
        src(src),
        dst(dst),
        size(size),
        kind(kind),
        stream(stream) {}
  const void* src;
  const void* dst;
  size_t size;
  hipMemcpyKind kind;
  hipStream_t stream;
};

struct roctracerMallocRow : public roctracerRow {
  roctracerMallocRow(
      uint64_t id,
      uint32_t domain,
      uint32_t cid,
      uint32_t pid,
      uint32_t tid,
      uint64_t begin,
      uint64_t end,
      const void* ptr,
      size_t size,
      roctracer_activity_types type = ROCTRACER_ACTIVITY_MALLOC)
      : roctracerRow(id, domain, cid, pid, tid, begin, end, type),
        ptr(ptr),
        size(size) {}
  const void* ptr;
  size_t size;
};

struct roctracerAsyncRow : public roctracerBase {
  roctracerAsyncRow(
      uint64_t id,
      uint32_t domain,
      uint32_t kind,
      uint32_t op,
      int device,
      uint64_t queue,
      uint64_t begin,
      uint64_t end,
      const std::string& kernelName,
      roctracer_activity_types type = ROCTRACER_ACTIVITY_ASYNC)
      : roctracerBase(id, domain, begin, end, type),
        kind(kind),
        op(op),
        device(device),
        queue(queue),
        kernelName(kernelName) {}
  uint32_t kind;
  uint32_t op;
  int device;
  uint64_t queue;
  std::string kernelName;
};
