/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "ActivityType.h"
#include "GenericTraceActivity.h"
#include "RoctracerActivityBuffer.h"

#include <atomic>
#ifdef HAS_ROCTRACER
#define __HIP_PLATFORM_AMD__
#include <roctracer.h>
#include <roctracer_hcc.h>
#include <roctracer_hip.h>
#include <roctracer_ext.h>
#include <roctracer_roctx.h>

#endif
#include <functional>
#include <list>
#include <memory>
#include <set>
#include <vector>
#include <map>
#include <unordered_map>
#include <deque>


namespace KINETO_NAMESPACE {

using namespace libkineto;

class ApiIdList
{
public:
  ApiIdList();
  bool invertMode() { return m_invert; }
  void setInvertMode(bool invert) { m_invert = invert; }
  void add(std::string apiName);
  void remove(std::string apiName);
  bool loadUserPrefs();
  bool contains(uint32_t apiId);
  const std::unordered_map<uint32_t, uint32_t> &filterList() { return m_filter; }

private:
  std::map<std::string, uint32_t> m_ids;
  std::unordered_map<uint32_t, uint32_t> m_filter;
  bool m_invert;

  void loadApiNames();
};

struct roctracerRow {
  roctracerRow(uint64_t id, uint32_t domain, uint32_t cid, uint32_t pid
             , uint32_t tid, uint64_t begin, uint64_t end)
    : id(id), domain(domain), cid(cid), pid(pid), tid(tid), begin(begin), end(end) {}
  uint64_t id;  // correlation_id
  uint32_t domain;
  uint32_t cid;
  uint32_t pid;
  uint32_t tid;
  uint64_t begin;
  uint64_t end;
};

struct kernelRow : public roctracerRow {
  kernelRow(uint64_t id, uint32_t domain, uint32_t cid, uint32_t pid
          , uint32_t tid, uint64_t begin, uint64_t end
          , const void *faddr, hipFunction_t function
          , unsigned int gx, unsigned int gy, unsigned int gz
          , unsigned int wx, unsigned int wy, unsigned int wz
          , size_t gss, hipStream_t stream)
    : roctracerRow(id, domain, cid, pid, tid, begin, end), functionAddr(faddr)
    , function(function), gridX(gx), gridY(gy), gridZ(gz)
    , workgroupX(wx), workgroupY(wy), workgroupZ(wz), groupSegmentSize(gss)
    , stream(stream) {}
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

struct copyRow : public roctracerRow {
  copyRow(uint64_t id, uint32_t domain, uint32_t cid, uint32_t pid
             , uint32_t tid, uint64_t begin, uint64_t end
             , const void* src, const void *dst, size_t size, hipMemcpyKind kind
             , hipStream_t stream)
    : roctracerRow(id, domain, cid, pid, tid, begin, end)
    , src(src), dst(dst), size(size), kind(kind), stream(stream) {}
  const void *src;
  const void *dst;
  size_t size;
  hipMemcpyKind kind;
  hipStream_t stream;
};

struct mallocRow : public roctracerRow {
  mallocRow(uint64_t id, uint32_t domain, uint32_t cid, uint32_t pid
             , uint32_t tid, uint64_t begin, uint64_t end
             , const void* ptr, size_t size)
    : roctracerRow(id, domain, cid, pid, tid, begin, end)
    , ptr(ptr), size(size) {}
  const void *ptr;
  size_t size;
};


class RoctracerActivityInterface {
 public:
  enum CorrelationFlowType {
    Default,
    User
  };

  RoctracerActivityInterface();
  RoctracerActivityInterface(const RoctracerActivityInterface&) = delete;
  RoctracerActivityInterface& operator=(const RoctracerActivityInterface&) = delete;

  virtual ~RoctracerActivityInterface();

  static RoctracerActivityInterface& singleton();

  static void pushCorrelationID(int id, CorrelationFlowType type);
  static void popCorrelationID(CorrelationFlowType type);

  void enableActivities(
    const std::set<ActivityType>& selected_activities);
  void disableActivities(
    const std::set<ActivityType>& selected_activities);
  void clearActivities();

  int processActivities(ActivityLogger& logger);

  void setMaxBufferSize(int size);

  std::atomic_bool stopCollection{false};

 private:
  bool m_registered{false};
  void endTracing();

#ifdef HAS_ROCTRACER
  roctracer_pool_t *m_hccPool{NULL};
  static void api_callback(uint32_t domain, uint32_t cid, const void* callback_data, void* arg);
  static void activity_callback(const char* begin, const char* end, void* arg);

  //Name cache
  uint32_t m_nextStringId{2};
  std::map<uint32_t, std::string> m_strings;
  std::map<std::string, uint32_t> m_reverseStrings;
  std::map<activity_correlation_id_t, uint32_t> m_kernelNames;

  ApiIdList m_loggedIds;

  // Api callback data
  std::deque<roctracerRow> m_rows;
  std::deque<kernelRow> m_kernelRows;
  std::deque<copyRow> m_copyRows;
  std::deque<mallocRow> m_mallocRows;
  std::map<activity_correlation_id_t, GenericTraceActivity> m_kernelLaunches;
#endif

  int m_maxGpuBufferCount{0};
  std::unique_ptr<std::list<RoctracerActivityBuffer>> m_gpuTraceBuffers;
  bool m_externalCorrelationEnabled{true};
};

} // namespace KINETO_NAMESPACE

