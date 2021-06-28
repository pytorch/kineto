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

#include "Logger.h"
#include "output_base.h"

typedef uint64_t timestamp_t;

static size_t BUFFERSIZE=1<<20;

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

// TODO: do we want this to be configurable?
// Set to 2MB to avoid constantly creating buffers (espeically for networks
// that has many small memcpy such as sparseNN)
// Consider putting this on huge pages?
constexpr size_t kBufSize(2 * 1024 * 1024);

RoctracerActivityInterface& RoctracerActivityInterface::singleton() {
  static RoctracerActivityInterface instance;
  return instance;
}

RoctracerActivityInterface::RoctracerActivityInterface() {
  gpuTraceBuffers_ = std::make_unique<std::list<RoctracerActivityBuffer>>();
}

RoctracerActivityInterface::~RoctracerActivityInterface() {
  disableActivities(std::set<ActivityType>());
  endTracing();
}

void RoctracerActivityInterface::pushCorrelationID(int id, CorrelationFlowType type) {
#ifdef HAS_ROCTRACER
  // FIXME: no type
  // FIXME: disabled
  //roctracer_activity_push_external_correlation_id(id);
#endif
}

void RoctracerActivityInterface::popCorrelationID(CorrelationFlowType type) {
#ifdef HAS_CUPTI
  // FIXME: no type
  // FIXME: disabled
  //roctracer_activity_pop_external_correlation_id(nullptr);
#endif
}

static int getSMCount() {
  // FIXME
  return -1;
}

int RoctracerActivityInterface::smCount() {
  static int sm_count = getSMCount();
  return sm_count;
}

void RoctracerActivityInterface::setMaxBufferSize(int size) {
  maxGpuBufferCount_ = 1 + size / kBufSize;
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

  for (auto& buffer : *gpuTraceBuffers_) {
    const roctracer_record_t* record = (const roctracer_record_t*)(buffer.data);
    const roctracer_record_t* end_record = (const roctracer_record_t*)(buffer.data + buffer.validSize);
    GenericTraceActivity a;

    while (record < end_record) {
      if ((record->domain == ACTIVITY_DOMAIN_HIP_API) && (m_loggedIds.contains(record->op))) {
        const char *name = roctracer_op_string(record->domain, record->op, record->kind);
        a.device = record->process_id;
        a.sysThreadId= record->thread_id;

        a.startTime = (record->begin_ns + toffset) / 1000;
        a.endTime = (record->end_ns + toffset) / 1000;
        a.correlation = record->correlation_id;

        a.activityType = ActivityType::CUDA_RUNTIME;
        a.activityName = std::string(name);

        logger.handleGenericActivity(a);
        ++count;
      }
      else if (record->domain == ACTIVITY_DOMAIN_HCC_OPS) {
        const char *name = roctracer_op_string(record->domain, record->op, record->kind);
        a.device = record->device_id;
        a.sysThreadId = record->queue_id;

        a.startTime = (record->begin_ns + toffset) / 1000;
        a.endTime = (record->end_ns + toffset) / 1000;
        a.correlation = record->correlation_id;

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
  if (gpuTraceBuffers_) {
    gpuTraceBuffers_->clear();
  }
}

void RoctracerActivityInterface::addActivityBuffer(uint8_t* buffer, size_t validSize) {
  if (!gpuTraceBuffers_) {
    gpuTraceBuffers_ = std::make_unique<std::list<RoctracerActivityBuffer>>();
  }
  gpuTraceBuffers_->emplace_back(buffer, validSize);
}

void RoctracerActivityInterface::api_callback(uint32_t domain, uint32_t cid, const void* callback_data, void* arg)
{
  if (domain == ACTIVITY_DOMAIN_HIP_API) {
    const hip_api_data_t* data = (const hip_api_data_t*)(callback_data);
    if (data->phase == 0) {
      std::string name;

      switch (cid) {
        case HIP_API_ID_hipLaunchKernel:
          {
            std::ostringstream ss;
            ss << data->args.hipLaunchKernel.function_address;
            name = ss.str();
            name = hipKernelNameRefByPtr(data->args.hipLaunchKernel.function_address, data->args.hipLaunchKernel.stream);
          }
          break;
        case HIP_API_ID_hipExtLaunchKernel:
          {
            std::ostringstream ss;
            ss << data->args.hipExtLaunchKernel.function_address;
            name = ss.str();
            name = hipKernelNameRefByPtr(data->args.hipLaunchKernel.function_address, data->args.hipLaunchKernel.stream);
          }
          break;
        case HIP_API_ID_hipHccModuleLaunchKernel:
          name = cxx_demangle(hipKernelNameRef(data->args.hipHccModuleLaunchKernel.f));
          break;
        case HIP_API_ID_hipExtModuleLaunchKernel:
          name = cxx_demangle(hipKernelNameRef(data->args.hipExtModuleLaunchKernel.f));
          break;
        case HIP_API_ID_hipModuleLaunchKernel:
          name = cxx_demangle(hipKernelNameRef(data->args.hipModuleLaunchKernel.f));
          break;
        default:
          break;
      }

      if (name.empty() == false) {
        RoctracerActivityInterface *dis = &singleton();
        uint32_t string_id = dis->m_reverseStrings[name];
        if (string_id == 0) {
          string_id = dis->m_nextStringId++;
          dis->m_reverseStrings[name] = string_id;
          dis->m_strings[string_id] = name;
        }
        dis->m_kernelNames[data->correlation_id] = string_id;
      }
    }
  }
}

void RoctracerActivityInterface::activity_callback(const char* begin, const char* end, void* arg)
{
  size_t size = end - begin;
  uint8_t *buffer = (uint8_t*) malloc(size);
  auto &gpuTraceBuffers = singleton().gpuTraceBuffers_;
  memcpy(buffer, begin, size);
  gpuTraceBuffers->emplace_back(buffer, size);
}

void RoctracerActivityInterface::hip_activity_callback(const char* begin, const char* end, void* arg)
{
}

void RoctracerActivityInterface::hcc_activity_callback(const char* begin, const char* end, void* arg)
{
}

void RoctracerActivityInterface::enableActivities(
    const std::set<ActivityType>& selected_activities) {
#ifdef HAS_ROCTRACER
  if (m_registered == false) {
    roctracer_set_properties(ACTIVITY_DOMAIN_HIP_API, NULL);  // Magic encantation

    // Enable API callbacks
    roctracer_enable_domain_callback(ACTIVITY_DOMAIN_HIP_API, api_callback, NULL);
    //roctracer_enable_domain_callback(ACTIVITY_DOMAIN_ROCTX, api_callback, NULL);

    // Allocate default tracing pool
    roctracer_properties_t properties;
    memset(&properties, 0, sizeof(roctracer_properties_t));
    properties.buffer_size = 0x1000;
    roctracer_open_pool(&properties);

#if 1
    // Log hip
    roctracer_properties_t hip_cb_properties;
    memset(&hip_cb_properties, 0, sizeof(roctracer_properties_t));
    hip_cb_properties.buffer_size = 0x4000;
    //hip_cb_properties.buffer_callback_fun = hip_activity_callback;
    hip_cb_properties.buffer_callback_fun = activity_callback;
    roctracer_open_pool_expl(&hip_cb_properties, &hipPool_);
    roctracer_enable_domain_activity_expl(ACTIVITY_DOMAIN_HIP_API, hipPool_);
    roctracer_enable_domain_activity_expl(ACTIVITY_DOMAIN_HCC_OPS, hipPool_);    // FIXME - logging on 1 thread for now
#endif
#if 0
    // Log hcc
    roctracer_properties_t hcc_cb_properties;
    memset(&hcc_cb_properties, 0, sizeof(roctracer_properties_t));
    hcc_cb_properties.buffer_size = 0x4000;
    hcc_cb_properties.buffer_callback_fun = hcc_activity_callback;
    roctracer_open_pool_expl(&hcc_cb_properties, &hccPool_);
    roctracer_enable_domain_activity_expl(ACTIVITY_DOMAIN_HCC_OPS, hccPool_);
#endif

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

    m_registered = true;
  }
  roctracer_start();
#endif
}

void RoctracerActivityInterface::disableActivities(
    const std::set<ActivityType>& selected_activities) {
#ifdef HAS_ROCTRACER
  roctracer_stop();
  roctracer_flush_activity_expl(hipPool_);
#endif
}

void RoctracerActivityInterface::endTracing() {
  if (m_registered == true) {
    roctracer_disable_domain_callback(ACTIVITY_DOMAIN_HIP_API);
    //roctracer_disable_domain_callback(ACTIVITY_DOMAIN_ROCTX);

    roctracer_disable_domain_activity(ACTIVITY_DOMAIN_HIP_API);
    roctracer_disable_domain_activity(ACTIVITY_DOMAIN_HCC_OPS);
    roctracer_close_pool_expl(hipPool_);
  }
}


ApiIdList::ApiIdList()
: m_invert(true)
{
  loadApiNames();
}

void ApiIdList::add(std::string apiName)
{
  auto it = m_ids.find(apiName);
  if (it != m_ids.end()) 
    m_filter[it->second] = 1;
}
void ApiIdList::remove(std::string apiName)
{
  auto it = m_ids.find(apiName);
  if (it != m_ids.end())
    m_filter.erase(it->second);
}

bool ApiIdList::loadUserPrefs()
{
  // FIXME: check an ENV variable that points to an exclude file
  return false;
}
bool ApiIdList::contains(uint32_t apiId)
{
  return (m_filter.count(apiId) > 0) ? !m_invert : m_invert;  // XOR
}

void ApiIdList::loadApiNames()
{
  // Build lut from apiName to apiId
  for (uint32_t i = 0; i < HIP_API_ID_NUMBER; ++i) {
    m_ids[std::string(hip_api_name(i))] = i;
  }
}


} // namespace KINETO_NAMESPACE
