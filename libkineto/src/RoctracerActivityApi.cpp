/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "RoctracerActivityApi.h"

#include <cstring>
#include <chrono>
#include <time.h>

#include "Demangle.h"
#include "output_base.h"
#include "ThreadUtil.h"

typedef uint64_t timestamp_t;

static timestamp_t timespec_to_ns(const timespec& time) {
    return ((timestamp_t)time.tv_sec * 1000000000) + time.tv_nsec;
  }

using namespace std::chrono;

namespace KINETO_NAMESPACE {

constexpr size_t kBufSize(2 * 1024 * 1024);

RoctracerActivityApi& RoctracerActivityApi::singleton() {
  static RoctracerActivityApi instance;
  return instance;
}

RoctracerActivityApi::RoctracerActivityApi() {
  gpuTraceBuffers_ = std::make_unique<std::list<RoctracerActivityBuffer>>();
}

RoctracerActivityApi::~RoctracerActivityApi() {
  disableActivities(std::set<ActivityType>());
  endTracing();
}

void RoctracerActivityApi::pushCorrelationID(int id, CorrelationFlowType type) {
#ifdef HAS_ROCTRACER
  if (!singleton().externalCorrelationEnabled_) {
    return;
  }
  // placeholder
#endif
}

void RoctracerActivityApi::popCorrelationID(CorrelationFlowType type) {
#ifdef HAS_ROCTRACER
  if (!singleton().externalCorrelationEnabled_) {
    return;
  }
  // placeholder
#endif
}

void RoctracerActivityApi::setMaxBufferSize(int size) {
  maxGpuBufferCount_ = 1 + size / kBufSize;
}

int RoctracerActivityApi::processActivities(
    ActivityLogger& logger) {
  // Find offset to map from monotonic clock to system clock.
  // This will break time-ordering of events but is status quo.

  timespec t0, t1, t00;
  clock_gettime(CLOCK_REALTIME, &t0);
  clock_gettime(CLOCK_MONOTONIC, &t1);
  clock_gettime(CLOCK_REALTIME, &t00);

  const timestamp_t toffset = (timespec_to_ns(t0) >> 1) + (timespec_to_ns(t00) >> 1) - timespec_to_ns(t1);

  int count = 0;

  // Basic Api calls

  for (auto &item : rows_) {
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
  for (auto &item : mallocRows_) {
    GenericTraceActivity a;
    a.startTime = (item.begin + toffset) / 1000;
    a.endTime = (item.end + toffset) / 1000;
    a.id = item.id;
    a.device = item.pid;
    a.resource = item.tid;
    a.activityType = ActivityType::CUDA_RUNTIME;
    a.activityName = std::string(roctracer_op_string(ACTIVITY_DOMAIN_HIP_API, item.cid, 0));

    a.addMetadata("ptr", item.ptr);
    if (item.cid == HIP_API_ID_hipMalloc) {
      a.addMetadata("size", item.size);
    }

    logger.handleGenericActivity(a);
    ++count;
  }

  // HipMemcpy calls
  for (auto &item : copyRows_) {
    GenericTraceActivity a;
    a.startTime = (item.begin + toffset) / 1000;
    a.endTime = (item.end + toffset) / 1000;
    a.id = item.id;
    a.device = item.pid;
    a.resource = item.tid;
    a.activityType = ActivityType::CUDA_RUNTIME;
    a.activityName = std::string(roctracer_op_string(ACTIVITY_DOMAIN_HIP_API, item.cid, 0));

    a.addMetadata("src", item.src);
    a.addMetadata("dst", item.dst);
    a.addMetadata("size", item.size);
    a.addMetadata("kind", item.kind);
    if ((item.cid == HIP_API_ID_hipMemcpyAsync) || (item.cid == HIP_API_ID_hipMemcpyWithStream)) {
      a.addMetadata("stream", fmt::format("{}", reinterpret_cast<void*>(item.stream)));
    }

    logger.handleGenericActivity(a);
    ++count;
  }

  // Kernel Launch Api calls

  for (auto &item : kernelRows_) {
    GenericTraceActivity a;
    a.startTime = (item.begin + toffset) / 1000;
    a.endTime = (item.end + toffset) / 1000;
    a.id = item.id;
    a.device = item.pid;
    a.resource = item.tid;
    a.activityType = ActivityType::CUDA_RUNTIME;
    a.activityName = std::string(roctracer_op_string(ACTIVITY_DOMAIN_HIP_API, item.cid, 0));

    if (item.functionAddr != nullptr) {
      a.addMetadataQuoted(
          "kernel", demangle(hipKernelNameRefByPtr(item.functionAddr, item.stream)));
    }
    else if (item.function != nullptr) {
      a.addMetadataQuoted(
          "kernel", demangle(hipKernelNameRef(item.function)));
    }
    a.addMetadata("grid dim", fmt::format("[{}, {}, {}]", item.gridX, item.gridY, item.gridZ));
    a.addMetadata("block dim", fmt::format("[{}, {}, {}]", item.workgroupX, item.workgroupY, item.workgroupZ));
    a.addMetadata("shared size", item.groupSegmentSize);
    a.addMetadata("stream", fmt::format("{}", reinterpret_cast<void*>(item.stream)));

    // Stash launches to tie to the async ops
    kernelLaunches_[a.id] = a;

    // Stash kernel names to tie to the async ops
    std::string name;
    if (item.functionAddr != nullptr) {
      name = demangle(hipKernelNameRefByPtr(item.functionAddr, item.stream));
    }
    else if (item.function != nullptr) {
      name = demangle(hipKernelNameRef(item.function));
    }
    if (!name.empty()) {
      uint32_t string_id = reverseStrings_[name];
      if (string_id == 0) {
        string_id = nextStringId_++;
        reverseStrings_[name] = string_id;
        strings_[string_id] = name;
      }
      kernelNames_[item.id] = string_id;
    }

    logger.handleGenericActivity(a);
    ++count;
  }

  // Async Ops

  for (auto& buffer : *gpuTraceBuffers_) {
    const roctracer_record_t* record = (const roctracer_record_t*)(buffer.data);
    const roctracer_record_t* end_record = (const roctracer_record_t*)(buffer.data + buffer.validSize);
    GenericTraceActivity a;

    while (record < end_record) {
      if ((record->domain == ACTIVITY_DOMAIN_HIP_API) && (loggedIds_.contains(record->op))) {
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
        auto kit = kernelLaunches_.find(record->correlation_id);
        if (kit != kernelLaunches_.end()) {
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

        auto it = kernelNames_.find(record->correlation_id);
        if (it != kernelNames_.end()) {
          a.activityName = strings_[it->second];
        }

        logger.handleGenericActivity(a);
        ++count;
      }

      roctracer_next_record(record, &record);
    }
  }
  return count;
}

void RoctracerActivityApi::clearActivities() {
  gpuTraceBuffers_->clear();
  rows_.clear();
  kernelRows_.clear();
  copyRows_.clear();
  mallocRows_.clear();
  kernelLaunches_.clear();
}

void RoctracerActivityApi::api_callback(uint32_t domain, uint32_t cid, const void* callback_data, void* arg)
{
  RoctracerActivityApi *dis = &singleton();

  if (domain == ACTIVITY_DOMAIN_HIP_API && dis->loggedIds_.contains(cid)) {
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
          dis->kernelRows_.emplace_back(data->correlation_id,
                              domain,
                              cid,
                              processId(),
                              systemThreadId(),
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
          dis->kernelRows_.emplace_back(data->correlation_id,
                              domain,
                              cid,
                              processId(),
                              systemThreadId(),
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
            dis->kernelRows_.emplace_back(data->correlation_id,
                              domain,
                              cid,
                              processId(),
                              systemThreadId(),
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
            dis->mallocRows_.emplace_back(data->correlation_id,
                              domain,
                              cid,
                              processId(),
                              systemThreadId(),
                              timespec_to_ns(startTime),
                              timespec_to_ns(endTime),
                              data->args.hipMalloc.ptr__val,
                              data->args.hipMalloc.size
                              );
          break;
        case HIP_API_ID_hipFree:
            dis->mallocRows_.emplace_back(data->correlation_id,
                              domain,
                              cid,
                              processId(),
                              systemThreadId(),
                              timespec_to_ns(startTime),
                              timespec_to_ns(endTime),
                              data->args.hipFree.ptr,
                              0
                              );
          break;
        case HIP_API_ID_hipMemcpy:
          {
            auto &args = data->args.hipMemcpy;
            dis->copyRows_.emplace_back(data->correlation_id,
                              domain,
                              cid,
                              processId(),
                              systemThreadId(),
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
            dis->copyRows_.emplace_back(data->correlation_id,
                              domain,
                              cid,
                              processId(),
                              systemThreadId(),
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
          dis->rows_.emplace_back(data->correlation_id,
                              domain,
                              cid,
                              processId(),
                              systemThreadId(),
                              timespec_to_ns(startTime),
                              timespec_to_ns(endTime)
                              );
          break;
      }
    }
  }
}

void RoctracerActivityApi::activity_callback(const char* begin, const char* end, void* arg)
{
  size_t size = end - begin;
  uint8_t *buffer = (uint8_t*) malloc(size);
  auto &gpuTraceBuffers = singleton().gpuTraceBuffers_;
  memcpy(buffer, begin, size);
  gpuTraceBuffers->emplace_back(buffer, size);
}

void RoctracerActivityApi::enableActivities(
    const std::set<ActivityType>& selected_activities) {
#ifdef HAS_ROCTRACER
  if (!registered_) {
    roctracer_set_properties(ACTIVITY_DOMAIN_HIP_API, nullptr);  // Magic encantation

    // Set some api calls to ignore
    loggedIds_.setInvertMode(true);  // Omit the specified api
    loggedIds_.add("hipGetDevice");
    loggedIds_.add("hipSetDevice");
    loggedIds_.add("hipGetLastError");
    loggedIds_.add("__hipPushCallConfiguration");
    loggedIds_.add("__hipPopCallConfiguration");
    loggedIds_.add("hipCtxSetCurrent");
    loggedIds_.add("hipEventRecord");
    loggedIds_.add("hipEventQuery");
    loggedIds_.add("hipGetDeviceProperties");
    loggedIds_.add("hipPeekAtLastError");
    loggedIds_.add("hipModuleGetFunction");
    loggedIds_.add("hipEventCreateWithFlags");

    // Enable API callbacks
    if (loggedIds_.invertMode() == true) {
        // exclusion list - enable entire domain and turn off things in list
        roctracer_enable_domain_callback(ACTIVITY_DOMAIN_HIP_API, api_callback, nullptr);
        const std::unordered_map<uint32_t, uint32_t> &filter = loggedIds_.filterList();
        for (auto it = filter.begin(); it != filter.end(); ++it) {
            roctracer_disable_op_callback(ACTIVITY_DOMAIN_HIP_API, it->first);
        }
    }
    else {
        // inclusion list - only enable things in the list
        const std::unordered_map<uint32_t, uint32_t> &filter = loggedIds_.filterList();
        roctracer_disable_domain_callback(ACTIVITY_DOMAIN_HIP_API);
        for (auto it = filter.begin(); it != filter.end(); ++it) {
            roctracer_enable_op_callback(ACTIVITY_DOMAIN_HIP_API, it->first, api_callback, nullptr);
        }
    }
    //roctracer_enable_domain_callback(ACTIVITY_DOMAIN_ROCTX, api_callback, nullptr);

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
    roctracer_open_pool_expl(&hcc_cb_properties, &hccPool_);
    roctracer_enable_domain_activity_expl(ACTIVITY_DOMAIN_HCC_OPS, hccPool_);

    registered_ = true;
  }

  for (const auto& activity : selected_activities) {
    if (activity == ActivityType::EXTERNAL_CORRELATION) {
        externalCorrelationEnabled_ = true;
    }
  }

  roctracer_start();
#endif
}

void RoctracerActivityApi::disableActivities(
    const std::set<ActivityType>& selected_activities) {
#ifdef HAS_ROCTRACER
  roctracer_stop();
  roctracer_flush_activity_expl(hccPool_);

  for (const auto& activity : selected_activities) {
    if (activity == ActivityType::EXTERNAL_CORRELATION) {
        externalCorrelationEnabled_ = false;
    }
  }
#endif
}

void RoctracerActivityApi::endTracing() {
  if (registered_ == true) {
    roctracer_disable_domain_callback(ACTIVITY_DOMAIN_HIP_API);
    //roctracer_disable_domain_callback(ACTIVITY_DOMAIN_ROCTX);

    roctracer_disable_domain_activity(ACTIVITY_DOMAIN_HCC_OPS);
    roctracer_close_pool_expl(hccPool_);
  }
}


ApiIdList::ApiIdList()
: invert_(true)
{
}

void ApiIdList::add(std::string apiName)
{
  uint32_t cid = 0;
  if (roctracer_op_code(ACTIVITY_DOMAIN_HIP_API, apiName.c_str(), &cid, nullptr) == ROCTRACER_STATUS_SUCCESS) {
    filter_[cid] = 1;
  }
}
void ApiIdList::remove(std::string apiName)
{
  uint32_t cid = 0;
  if (roctracer_op_code(ACTIVITY_DOMAIN_HIP_API, apiName.c_str(), &cid, nullptr) == ROCTRACER_STATUS_SUCCESS) {
    filter_.erase(cid);
  }
}

bool ApiIdList::loadUserPrefs()
{
  // placeholder
  return false;
}
bool ApiIdList::contains(uint32_t apiId)
{
  return (filter_.find(apiId) != filter_.end()) ? !invert_ : invert_;  // XOR
}

} // namespace KINETO_NAMESPACE
