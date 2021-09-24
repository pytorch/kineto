/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "RoctracerActivityInterface.h"

#include <cstring>
#include <chrono>
#include <time.h>

#include <cxxabi.h>
#include <sys/syscall.h>

#include "Logger.h"
#include "output_base.h"

typedef uint64_t timestamp_t;

static inline uint32_t getPid() { return syscall(__NR_getpid); }
static inline uint32_t getTid() { return syscall(__NR_gettid); }

// C++ symbol demangle
static inline const char* cxx_demangle(const char* symbol) {
  size_t funcnamesize;
  int status;
  const char* ret = (symbol != NULL) ? abi::__cxa_demangle(symbol, NULL, &funcnamesize, &status) : symbol;
  return (ret != NULL) ? ret : symbol;
}


static timestamp_t timespec_to_ns(const timespec& time) {
    return ((timestamp_t)time.tv_sec * 1000000000) + time.tv_nsec;
  }

using namespace std::chrono;

namespace KINETO_NAMESPACE {

constexpr size_t kBufSize(2 * 1024 * 1024);

RoctracerActivityInterface& RoctracerActivityInterface::singleton() {
  static RoctracerActivityInterface instance;
  return instance;
}

RoctracerActivityInterface::RoctracerActivityInterface() {
  m_gpuTraceBuffers = std::make_unique<std::list<RoctracerActivityBuffer>>();
}

RoctracerActivityInterface::~RoctracerActivityInterface() {
  disableActivities(std::set<ActivityType>());
  endTracing();
}

void RoctracerActivityInterface::pushCorrelationID(int id, CorrelationFlowType type) {
#ifdef HAS_ROCTRACER
  if (!singleton().m_externalCorrelationEnabled)
    return;
  // placeholder
#endif
}

void RoctracerActivityInterface::popCorrelationID(CorrelationFlowType type) {
#ifdef HAS_ROCTRACER
  if (!singleton().m_externalCorrelationEnabled)
    return;
  // placeholder
#endif
}

void RoctracerActivityInterface::setMaxBufferSize(int size) {
  m_maxGpuBufferCount = 1 + size / kBufSize;
}

int RoctracerActivityInterface::processActivities(
    ActivityLogger& logger) {
  // Find offset to map from monotonic clock to system clock.
  // This will break time-ordering of events but is status quo.

  timespec t0, t1, t00;
  clock_gettime(CLOCK_REALTIME, &t0);
  clock_gettime(CLOCK_MONOTONIC, &t1);
  clock_gettime(CLOCK_REALTIME, &t00);

  const timestamp_t toffset = (timespec_to_ns(t0) >> 1) + (timespec_to_ns(t00) >> 1) - timespec_to_ns(t1);

  int count = 0;
  char buff[4096];

  // Basic Api calls

  for (auto &item : m_rows) {
    GenericTraceActivity a;
    a.startTime = (item.begin + toffset) / 1000;
    a.endTime = (item.end + toffset) / 1000;
    a.id = item.id;
    a.device = item.pid;
    a.resource = item.tid;
    a.activityType = ActivityType::CUDA_RUNTIME;
    a.activityName = std::string(roctracer_op_string(ACTIVITY_DOMAIN_HIP_API, item.cid, 0));

    logger.handleGenericActivity(a);
    ++count;
  }

  // Malloc/Free calls
  for (auto &item : m_mallocRows) {
    GenericTraceActivity a;
    a.startTime = (item.begin + toffset) / 1000;
    a.endTime = (item.end + toffset) / 1000;
    a.id = item.id;
    a.device = item.pid;
    a.resource = item.tid;
    a.activityType = ActivityType::CUDA_RUNTIME;
    a.activityName = std::string(roctracer_op_string(ACTIVITY_DOMAIN_HIP_API, item.cid, 0));

    std::snprintf(buff, 4096, "%p", item.ptr);
    a.addMetadata("ptr", std::string(buff));
    if (item.cid == HIP_API_ID_hipMalloc)
      a.addMetadata("size", std::to_string(item.size));

    logger.handleGenericActivity(a);
    ++count;
  }

  // HipMemcpy calls
  for (auto &item : m_copyRows) {
    GenericTraceActivity a;
    a.startTime = (item.begin + toffset) / 1000;
    a.endTime = (item.end + toffset) / 1000;
    a.id = item.id;
    a.device = item.pid;
    a.resource = item.tid;
    a.activityType = ActivityType::CUDA_RUNTIME;
    a.activityName = std::string(roctracer_op_string(ACTIVITY_DOMAIN_HIP_API, item.cid, 0));

    std::snprintf(buff, 4096, "%p", item.src);
    a.addMetadata("src", std::string(buff));
    std::snprintf(buff, 4096, "%p", item.dst);
    a.addMetadata("dst", std::string(buff));
    a.addMetadata("size", std::to_string(item.size));
    a.addMetadata("kind", std::to_string(item.kind));
    if ((item.cid == HIP_API_ID_hipMemcpyAsync) || (item.cid == HIP_API_ID_hipMemcpyWithStream)) {
      std::snprintf(buff, 4096, "%p", item.stream);
      a.addMetadata("stream", std::string(buff));
    }

    logger.handleGenericActivity(a);
    ++count;
  }

  // Kernel Launch Api calls

  for (auto &item : m_kernelRows) {
    GenericTraceActivity a;
    a.startTime = (item.begin + toffset) / 1000;
    a.endTime = (item.end + toffset) / 1000;
    a.id = item.id;
    a.device = item.pid;
    a.resource = item.tid;
    a.activityType = ActivityType::CUDA_RUNTIME;
    a.activityName = std::string(roctracer_op_string(ACTIVITY_DOMAIN_HIP_API, item.cid, 0));

    if (item.functionAddr != NULL)
      a.addMetadata("kernel", hipKernelNameRefByPtr(item.functionAddr, item.stream));
    else if (item.function != NULL)
      a.addMetadata("kernel", hipKernelNameRef(item.function));
    std::snprintf(buff, 4096, "(%d, %d, %d)", item.gridX, item.gridY, item.gridZ);
    a.addMetadata("grid dim", std::string(buff));
    std::snprintf(buff, 4096, "(%d, %d, %d)", item.workgroupX, item.workgroupY, item.workgroupZ);
    a.addMetadata("block dim", std::string(buff));
    a.addMetadata("shared size", std::to_string(item.groupSegmentSize));
    std::snprintf(buff, 4096, "%p", item.stream);
    a.addMetadata("stream", std::string(buff));

    // Stash launches to tie to the async ops
    m_kernelLaunches[a.id] = a;

    // Stash kernel names to tie to the async ops
    std::string name;
    if (item.functionAddr != NULL)
      name = hipKernelNameRefByPtr(item.functionAddr, item.stream);
    else if (item.function != NULL)
      name = hipKernelNameRef(item.function);
    if (name.empty() == false) {
      uint32_t string_id = m_reverseStrings[name];
      if (string_id == 0) {
        string_id = m_nextStringId++;
        m_reverseStrings[name] = string_id;
        m_strings[string_id] = name;
      }
      m_kernelNames[item.id] = string_id;
    }

    logger.handleGenericActivity(a);
    ++count;
  }

  // Async Ops

  for (auto& buffer : *m_gpuTraceBuffers) {
    const roctracer_record_t* record = (const roctracer_record_t*)(buffer.data);
    const roctracer_record_t* end_record = (const roctracer_record_t*)(buffer.data + buffer.validSize);
    GenericTraceActivity a;

    while (record < end_record) {
      if ((record->domain == ACTIVITY_DOMAIN_HIP_API) && (m_loggedIds.contains(record->op))) {
        const char *name = roctracer_op_string(record->domain, record->op, record->kind);
        a.device = record->process_id;
        a.resource = record->thread_id;

        a.startTime = (record->begin_ns + toffset) / 1000;
        a.endTime = (record->end_ns + toffset) / 1000;
        a.id = record->correlation_id;

        a.activityType = ActivityType::CUDA_RUNTIME;
        a.activityName = std::string(name);

        logger.handleGenericActivity(a);
        ++count;
      }
      else if (record->domain == ACTIVITY_DOMAIN_HCC_OPS) {
        // Overlay launch metadata for kernels
        auto kit = m_kernelLaunches.find(record->correlation_id);
        if (kit != m_kernelLaunches.end()) {
          a = (*kit).second;
        }

        const char *name = roctracer_op_string(record->domain, record->op, record->kind);
        a.device = record->device_id;
        a.resource = record->queue_id;

        a.startTime = (record->begin_ns + toffset) / 1000;
        a.endTime = (record->end_ns + toffset) / 1000;
        a.id = record->correlation_id;

        a.activityType = ActivityType::CONCURRENT_KERNEL;
        a.activityName = std::string(name);

        auto it = m_kernelNames.find(record->correlation_id);
        if (it != m_kernelNames.end()) {
          a.activityName = m_strings[it->second];
        }

        logger.handleGenericActivity(a);
        ++count;
      }

      roctracer_next_record(record, &record);     
    }
  }
  return count;
}

void RoctracerActivityInterface::clearActivities() {
  m_gpuTraceBuffers->clear();
  m_rows.clear();
  m_kernelRows.clear();
  m_copyRows.clear();
  m_mallocRows.clear();
  m_kernelLaunches.clear();
}

void RoctracerActivityInterface::api_callback(uint32_t domain, uint32_t cid, const void* callback_data, void* arg)
{
  RoctracerActivityInterface *dis = &singleton();

  if (domain == ACTIVITY_DOMAIN_HIP_API && dis->m_loggedIds.contains(cid)) {
    const hip_api_data_t* data = (const hip_api_data_t*)(callback_data);

    // Pack callbacks into row structures

    static timespec timestamp;	// FIXME verify thread safety

    if (data->phase == ACTIVITY_API_PHASE_ENTER) {
      clock_gettime(CLOCK_MONOTONIC, &timestamp);  // record proper clock
    }
    else { // (data->phase == ACTIVITY_API_PHASE_EXIT)
      timespec endTime;
      timespec startTime { timestamp };
      clock_gettime(CLOCK_MONOTONIC, &endTime);  // record proper clock

      switch (cid) {
        case HIP_API_ID_hipLaunchKernel:
        case HIP_API_ID_hipExtLaunchKernel:
        case HIP_API_ID_hipLaunchCooperativeKernel:     // Should work here
          {
          auto &args = data->args.hipLaunchKernel;
          dis->m_kernelRows.emplace_back(data->correlation_id,
                              domain,
                              cid,
                              getPid(),
                              getTid(),
                              timespec_to_ns(startTime),
                              timespec_to_ns(endTime),
                              args.function_address,
                              nullptr,
                              args.numBlocks.x,
                              args.numBlocks.y,
                              args.numBlocks.z,
                              args.dimBlocks.x,
                              args.dimBlocks.y,
                              args.dimBlocks.z,
                              args.sharedMemBytes,
                              args.stream
                            );
          }
          break;
        case HIP_API_ID_hipHccModuleLaunchKernel:
        case HIP_API_ID_hipModuleLaunchKernel:
        case HIP_API_ID_hipExtModuleLaunchKernel:
          {
          auto &args = data->args.hipModuleLaunchKernel;
          dis->m_kernelRows.emplace_back(data->correlation_id,
                              domain,
                              cid,
                              getPid(),
                              getTid(),
                              timespec_to_ns(startTime),
                              timespec_to_ns(endTime),
                              nullptr,
                              args.f,
                              args.gridDimX,
                              args.gridDimY,
                              args.gridDimZ,
                              args.blockDimX,
                              args.blockDimY,
                              args.blockDimZ,
                              args.sharedMemBytes,
                              args.stream
                            );
          }
          break;
        case HIP_API_ID_hipLaunchCooperativeKernelMultiDevice:
        case HIP_API_ID_hipExtLaunchMultiKernelMultiDevice:
#if 0
          {
            auto &args = data->args.hipLaunchCooperativeKernelMultiDevice.launchParamsList__val;
            dis->m_kernelRows.emplace_back(data->correlation_id,
                              domain,
                              cid,
                              getPid(),
                              getTid(),
                              timespec_to_ns(startTime),
                              timespec_to_ns(endTime),
                              args.function_address,
                              nullptr,
                              args.numBlocks.x,
                              args.numBlocks.y,
                              args.numBlocks.z,
                              args.dimBlocks.x,
                              args.dimBlocks.y,
                              args.dimBlocks.z,
                              args.sharedMemBytes,
                              args.stream
                            );
          }
#endif
          break;
        case HIP_API_ID_hipMalloc:
            dis->m_mallocRows.emplace_back(data->correlation_id,
                              domain,
                              cid,
                              getPid(),
                              getTid(),
                              timespec_to_ns(startTime),
                              timespec_to_ns(endTime),
                              data->args.hipMalloc.ptr__val,
                              data->args.hipMalloc.size
                              );
          break;
        case HIP_API_ID_hipFree:
            dis->m_mallocRows.emplace_back(data->correlation_id,
                              domain,
                              cid,
                              getPid(),
                              getTid(),
                              timespec_to_ns(startTime),
                              timespec_to_ns(endTime),
                              data->args.hipFree.ptr,
                              0
                              );
          break;
        case HIP_API_ID_hipMemcpy:
          {
            auto &args = data->args.hipMemcpy;
            dis->m_copyRows.emplace_back(data->correlation_id,
                              domain,
                              cid,
                              getPid(),
                              getTid(),
                              timespec_to_ns(startTime),
                              timespec_to_ns(endTime),
                              args.src,
                              args.dst,
                              args.sizeBytes,
                              args.kind,
                              static_cast<hipStream_t>(0)  // use placeholder?
                              );
          }
          break;
        case HIP_API_ID_hipMemcpyAsync:
        case HIP_API_ID_hipMemcpyWithStream:
          {
            auto &args = data->args.hipMemcpyAsync;
            dis->m_copyRows.emplace_back(data->correlation_id,
                              domain,
                              cid,
                              getPid(),
                              getTid(),
                              timespec_to_ns(startTime),
                              timespec_to_ns(endTime),
                              args.src,
                              args.dst,
                              args.sizeBytes,
                              args.kind,
                              args.stream
                              );
          }
          break;
        default:
          dis->m_rows.emplace_back(data->correlation_id,
                              domain,
                              cid,
                              getPid(),
                              getTid(),
                              timespec_to_ns(startTime),
                              timespec_to_ns(endTime)
                              );
          break;
      }
    }
  }
}

void RoctracerActivityInterface::activity_callback(const char* begin, const char* end, void* arg)
{
  size_t size = end - begin;
  uint8_t *buffer = (uint8_t*) malloc(size);
  auto &gpuTraceBuffers = singleton().m_gpuTraceBuffers;
  memcpy(buffer, begin, size);
  gpuTraceBuffers->emplace_back(buffer, size);
}

void RoctracerActivityInterface::enableActivities(
    const std::set<ActivityType>& selected_activities) {
#ifdef HAS_ROCTRACER
  if (m_registered == false) {
    roctracer_set_properties(ACTIVITY_DOMAIN_HIP_API, NULL);  // Magic encantation

    // Set some api calls to ignore
    m_loggedIds.setInvertMode(true);  // Omit the specified api
    m_loggedIds.add("hipGetDevice");
    m_loggedIds.add("hipSetDevice");
    m_loggedIds.add("hipGetLastError");
    m_loggedIds.add("__hipPushCallConfiguration");
    m_loggedIds.add("__hipPopCallConfiguration");
    m_loggedIds.add("hipCtxSetCurrent");
    m_loggedIds.add("hipEventRecord");
    m_loggedIds.add("hipEventQuery");
    m_loggedIds.add("hipGetDeviceProperties");
    m_loggedIds.add("hipPeekAtLastError");
    m_loggedIds.add("hipModuleGetFunction");
    m_loggedIds.add("hipEventCreateWithFlags");

    // Enable API callbacks
    if (m_loggedIds.invertMode() == true) {
        // exclusion list - enable entire domain and turn off things in list
        roctracer_enable_domain_callback(ACTIVITY_DOMAIN_HIP_API, api_callback, NULL);
        const std::unordered_map<uint32_t, uint32_t> &filter = m_loggedIds.filterList();
        for (auto it = filter.begin(); it != filter.end(); ++it) {
            roctracer_disable_op_callback(ACTIVITY_DOMAIN_HIP_API, it->first);
        }
    }
    else {
        // inclusion list - only enable things in the list
        const std::unordered_map<uint32_t, uint32_t> &filter = m_loggedIds.filterList();
        roctracer_disable_domain_callback(ACTIVITY_DOMAIN_HIP_API);
        for (auto it = filter.begin(); it != filter.end(); ++it) {
            roctracer_enable_op_callback(ACTIVITY_DOMAIN_HIP_API, it->first, api_callback, NULL);
        }
    }
    //roctracer_enable_domain_callback(ACTIVITY_DOMAIN_ROCTX, api_callback, NULL);

    // Allocate default tracing pool
    roctracer_properties_t properties;
    memset(&properties, 0, sizeof(roctracer_properties_t));
    properties.buffer_size = 0x1000;
    roctracer_open_pool(&properties);

    // Enable async op collection
    roctracer_properties_t hcc_cb_properties;
    memset(&hcc_cb_properties, 0, sizeof(roctracer_properties_t));
    hcc_cb_properties.buffer_size = 0x4000;
    hcc_cb_properties.buffer_callback_fun = activity_callback;
    roctracer_open_pool_expl(&hcc_cb_properties, &m_hccPool);
    roctracer_enable_domain_activity_expl(ACTIVITY_DOMAIN_HCC_OPS, m_hccPool);

    m_registered = true;
  }

  for (const auto& activity : selected_activities) {
    if (activity == ActivityType::EXTERNAL_CORRELATION) {
        m_externalCorrelationEnabled = true;
    }
  }

  roctracer_start();
#endif
}

void RoctracerActivityInterface::disableActivities(
    const std::set<ActivityType>& selected_activities) {
#ifdef HAS_ROCTRACER
  roctracer_stop();
  roctracer_flush_activity_expl(m_hccPool);

  for (const auto& activity : selected_activities) {
    if (activity == ActivityType::EXTERNAL_CORRELATION) {
        m_externalCorrelationEnabled = false;
    }
  }
#endif
}

void RoctracerActivityInterface::endTracing() {
  if (m_registered == true) {
    roctracer_disable_domain_callback(ACTIVITY_DOMAIN_HIP_API);
    //roctracer_disable_domain_callback(ACTIVITY_DOMAIN_ROCTX);

    roctracer_disable_domain_activity(ACTIVITY_DOMAIN_HCC_OPS);
    roctracer_close_pool_expl(m_hccPool);
  }
}


ApiIdList::ApiIdList()
: m_invert(true)
{
}

void ApiIdList::add(std::string apiName)
{
  uint32_t cid = 0;
  if (roctracer_op_code(ACTIVITY_DOMAIN_HIP_API, apiName.c_str(), &cid, NULL) == ROCTRACER_STATUS_SUCCESS)
    m_filter[cid] = 1;
}
void ApiIdList::remove(std::string apiName)
{
  uint32_t cid = 0;
  if (roctracer_op_code(ACTIVITY_DOMAIN_HIP_API, apiName.c_str(), &cid, NULL) == ROCTRACER_STATUS_SUCCESS)
    m_filter.erase(cid);
}

bool ApiIdList::loadUserPrefs()
{
  // placeholder
  return false;
}
bool ApiIdList::contains(uint32_t apiId)
{
  return (m_filter.count(apiId) > 0) ? !m_invert : m_invert;  // XOR
}

} // namespace KINETO_NAMESPACE
