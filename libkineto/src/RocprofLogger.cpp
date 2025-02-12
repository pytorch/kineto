/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "RocprofLogger.h"

#include <rocprofiler-sdk/context.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/marker/api_id.h>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>
#include <rocprofiler-sdk/cxx/name_info.hpp>

#include <time.h>
#include <unistd.h>
#include <chrono>
#include <cstring>
#include <mutex>

#include "Demangle.h"
#include "Logger.h"
#include "ThreadUtil.h"

using namespace libkineto;
using namespace std::chrono;
using namespace RocLogger;


class RocprofLoggerShared;

namespace
{
    RocprofLoggerShared *s {nullptr};
    using kernel_symbol_data_t = rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t;
    using kernel_symbol_map_t = std::unordered_map<rocprofiler_kernel_id_t, kernel_symbol_data_t>;
    using kernel_name_map_t = std::unordered_map<rocprofiler_kernel_id_t, const char *>;
    using rocprofiler::sdk::buffer_name_info;
    using agent_info_map_t = std::unordered_map<uint64_t, rocprofiler_agent_v0_t>;

    class RocprofApiIdList : public ApiIdList
    {
    public:
        RocprofApiIdList(buffer_name_info &names);
        uint32_t mapName(const std::string &apiName) override;
        std::vector<rocprofiler_tracing_operation_t> allEnabled();
    private:
        std::unordered_map<std::string, size_t> nameMap_;
    };
} //namespace


class RocprofLoggerShared
{
public:
    static RocprofLoggerShared& singleton();

    rocprofiler_client_id_t *clientId {nullptr};
    rocprofiler_tool_configure_result_t cfg = rocprofiler_tool_configure_result_t{
                                            sizeof(rocprofiler_tool_configure_result_t),
                                            &RocprofLogger::toolInit,
                                            &RocprofLogger::toolFinialize,
                                            nullptr};

    // Contexts
    rocprofiler_context_id_t utilityContext = {0};
    rocprofiler_context_id_t context = {0};

    // Buffers
    //rocprofiler_buffer_id_t client_buffer = {};

    // Manage kernel names - #betterThanRoctracer
    kernel_symbol_map_t kernel_info = {};
    kernel_name_map_t kernel_names = {};

    // Manage buffer name - #betterThanRoctracer
    buffer_name_info name_info = {};

    // Agent info
    // <rocprofiler_profile_config_id_t.handle, rocprofiler_agent_v0_t>
    agent_info_map_t agents = {};

private:
    RocprofLoggerShared() { s = this; }
    ~RocprofLoggerShared() { s = nullptr; }
};

RocprofLoggerShared &RocprofLoggerShared::singleton()
{
    static RocprofLoggerShared *instance = new RocprofLoggerShared();   // Leak this
    return *instance;
}

std::vector<rocprofiler_agent_v0_t>
get_gpu_device_agents()
{
  std::vector<rocprofiler_agent_v0_t> agents;

  // Callback used by rocprofiler_query_available_agents to return
  // agents on the device. This can include CPU agents as well. We
  // select GPU agents only (i.e. type == ROCPROFILER_AGENT_TYPE_GPU)
  rocprofiler_query_available_agents_cb_t iterate_cb = [](rocprofiler_agent_version_t agents_ver,
                                                          const void**                agents_arr,
                                                          size_t                      num_agents,
                                                          void*                       udata) {
    if(agents_ver != ROCPROFILER_AGENT_INFO_VERSION_0)
      throw std::runtime_error{"unexpected rocprofiler agent version"};
    auto* agents_v = static_cast<std::vector<rocprofiler_agent_v0_t>*>(udata);
    for(size_t i = 0; i < num_agents; ++i)
    {
      const auto* agent = static_cast<const rocprofiler_agent_v0_t*>(agents_arr[i]);
      //if(agent->type == ROCPROFILER_AGENT_TYPE_GPU) agents_v->emplace_back(*agent);
      agents_v->emplace_back(*agent);
    }
    return ROCPROFILER_STATUS_SUCCESS;
  };

  // Query the agents, only a single callback is made that contains a vector
  // of all agents.
  rocprofiler_query_available_agents(ROCPROFILER_AGENT_INFO_VERSION_0,
                                         iterate_cb,
                                         sizeof(rocprofiler_agent_t),
                                         const_cast<void*>(static_cast<const void*>(&agents)));
  return agents;
}



//
// Static setup
//
extern "C" rocprofiler_tool_configure_result_t*
rocprofiler_configure(uint32_t                 version,
                      const char*              runtime_version,
                      uint32_t                 priority,
                      rocprofiler_client_id_t* id)
{
fprintf(stderr, "rocprofiler_configure() - kineto\n");
    RocprofLoggerShared::singleton();       // CRITICAL: static init

    id->name = "kineto";
    s->clientId = id;

    // return pointer to configure data
    return &s->cfg;
}

int RocprofLogger::toolInit(rocprofiler_client_finalize_t finialize_func, void* tool_data)
{
fprintf(stderr, "RocprofLogger::toolInit()\n");
    // Gather api names
    s->name_info = rocprofiler::sdk::get_buffer_tracing_names();

    // Gather agent info
    auto agent_info = get_gpu_device_agents();
    for (auto agent : agent_info) {
        s->agents[agent.id.handle] = agent;
    }

    //
    // Setup utility context to gather code object info
    //
    rocprofiler_create_context(&s->utilityContext);
    auto code_object_ops = std::vector<rocprofiler_tracing_operation_t>{
        ROCPROFILER_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER};

    rocprofiler_configure_callback_tracing_service(s->utilityContext,
                                                   ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT,
                                                   code_object_ops.data(),
                                                   code_object_ops.size(),
                                                   RocprofLogger::code_object_callback,
                                                   nullptr);
    {
        int isValid = 0;
        rocprofiler_context_is_valid(s->utilityContext, &isValid);
        if (isValid == 0) {
            s->utilityContext.handle = 0;   // Can't destroy it, so leak it
            return -1;
        }
    }
    rocprofiler_start_context(s->utilityContext);

    //
    // select some api calls to omit, in the most inconvenient way possible
    // #betterThanRoctracer
    RocprofApiIdList apiList(s->name_info);
    apiList.setInvertMode(true);  // Omit the specified api
    apiList.add("hipGetDevice");
    apiList.add("hipSetDevice");
    apiList.add("hipGetLastError");
    apiList.add("__hipPushCallConfiguration");
    apiList.add("__hipPopCallConfiguration");
    apiList.add("hipCtxSetCurrent");
    apiList.add("hipEventRecord");
    apiList.add("hipEventQuery");
    apiList.add("hipGetDeviceProperties");
    apiList.add("hipPeekAtLastError");
    apiList.add("hipModuleGetFunction");
    apiList.add("hipEventCreateWithFlags");

    // Get a vector of the enabled api calls
    auto apis = apiList.allEnabled();

    //
    // Setup main context to collect runtime and kernel info
    //
    rocprofiler_create_context(&s->context);

    rocprofiler_configure_callback_tracing_service(s->context,
                                               ROCPROFILER_CALLBACK_TRACING_HIP_RUNTIME_API,
                                               apis.data(),
                                               apis.size(),
                                               api_callback,
                                               nullptr);

    rocprofiler_configure_callback_tracing_service(s->context,
                                               ROCPROFILER_CALLBACK_TRACING_KERNEL_DISPATCH,
                                               nullptr,
                                               0,
                                               api_callback,
                                               nullptr);

    rocprofiler_configure_callback_tracing_service(s->context,
                                               ROCPROFILER_CALLBACK_TRACING_MEMORY_COPY,
                                               nullptr,
                                               0,
                                               api_callback,
                                               nullptr);

    {
        int isValid = 0;
        rocprofiler_context_is_valid(s->context, &isValid);
        if (isValid == 0) {
            s->context.handle = 0;   // Can't destroy it, so leak it
            return -1;
        }
    }
    rocprofiler_start_context(s->context);

    return 0;
}

void RocprofLogger::toolFinialize(void* tool_data)
{
fprintf(stderr, "RocprofLogger::toolFinalize()\n");
    rocprofiler_stop_context(s->utilityContext);
    s->utilityContext.handle = 0;
    rocprofiler_stop_context(s->context);
    s->context.handle = 0;
}









class Flush {
 public:
  std::mutex mutex_;
  std::atomic<uint64_t> maxCorrelationId_;
  uint64_t maxCompletedCorrelationId_{0};
  void reportCorrelation(const uint64_t& cid) {
    uint64_t prev = maxCorrelationId_;
    while (prev < cid && !maxCorrelationId_.compare_exchange_weak(prev, cid)) {
    }
  }
};
static Flush s_flush;

RocprofLogger& RocprofLogger::singleton() {
  static RocprofLogger instance;
  return instance;
}

RocprofLogger::RocprofLogger() {}

RocprofLogger::~RocprofLogger() {
  stopLogging();
  endTracing();
}

namespace {
thread_local std::deque<uint64_t>
    t_externalIds[RocLogger::CorrelationDomain::size];
}

void RocprofLogger::pushCorrelationID(uint64_t id, CorrelationDomain type) {
  if (!singleton().externalCorrelationEnabled_) {
    return;
  }
  t_externalIds[type].push_back(id);
}

void RocprofLogger::popCorrelationID(CorrelationDomain type) {
  if (!singleton().externalCorrelationEnabled_) {
    return;
  }
  if (!t_externalIds[type].empty()) {
    t_externalIds[type].pop_back();
  } else {
    LOG(ERROR)
        << "Attempt to popCorrelationID from an empty external Ids stack";
  }
}

void RocprofLogger::clearLogs() {
  rows_.clear();
  for (int i = 0; i < CorrelationDomain::size; ++i) {
    externalCorrelations_[i].clear();
  }
}

void RocprofLogger::insert_row_to_buffer(roctracerBase* row) {
  RocprofLogger* dis = &singleton();
  std::lock_guard<std::mutex> lock(dis->rowsMutex_);
  if (dis->rows_.size() >= dis->maxBufferSize_) {
    LOG_FIRST_N(WARNING, 10)
        << "Exceeded max GPU buffer count (" << dis->rows_.size() << " > "
        << dis->maxBufferSize_ << ") - terminating tracing";
    return;
  }
  dis->rows_.push_back(row);
}

void RocprofLogger::code_object_callback(rocprofiler_callback_tracing_record_t record, rocprofiler_user_data_t* user_data, void* callback_data)
{
    if (record.kind == ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT &&
       record.operation == ROCPROFILER_CODE_OBJECT_LOAD)
    {
        if(record.phase == ROCPROFILER_CALLBACK_PHASE_UNLOAD)
        {
            // flush the buffer to ensure that any lookups for the client kernel names for the code
            // object are completed
            // NOTE: not using buffer ATM
        }
    }
    else if(record.kind == ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT &&
            record.operation == ROCPROFILER_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER)
    {
        auto* data = static_cast<kernel_symbol_data_t*>(record.payload);
        if (record.phase == ROCPROFILER_CALLBACK_PHASE_LOAD)
        {
            s->kernel_info.emplace(data->kernel_id, *data);
            s->kernel_names.emplace(data->kernel_id, demangle(data->kernel_name).c_str());
        }
        else if (record.phase == ROCPROFILER_CALLBACK_PHASE_UNLOAD)
        {
            // FIXME: clear these?  At minimum need kernel names at shutdown, async completion
            //s->kernel_info.erase(data->kernel_id);
            //s->kernel_names.erase(data->kernel_id);
        }
    }
}

void RocprofLogger::api_callback(rocprofiler_callback_tracing_record_t record, rocprofiler_user_data_t* user_data, void* callback_data)
{
    RocprofLogger* dis = &singleton();

    if (record.kind == ROCPROFILER_CALLBACK_TRACING_HIP_RUNTIME_API) {
        if (record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER) {
        } // ROCPROFILER_CALLBACK_PHASE_ENTER
        else {  // ROCPROFILER_CALLBACK_PHASE_EXIT
        } // ROCPROFILER_CALLBACK_PHASE_EXIT
    }  // ROCPROFILER_CALLBACK_TRACING_HIP_RUNTIME_API
    else if (record.kind == ROCPROFILER_CALLBACK_TRACING_KERNEL_DISPATCH) {
        if (record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER) {
            ;
        }
        else if (record.phase == ROCPROFILER_CALLBACK_PHASE_EXIT) {
        }
        else if (record.phase == ROCPROFILER_CALLBACK_PHASE_NONE) {
            // completion callback - runtime thread
        }
    } // ROCPROFILER_CALLBACK_TRACING_KERNEL_DISPATCH
    else if (record.kind == ROCPROFILER_CALLBACK_TRACING_MEMORY_COPY) {
        if (record.phase == ROCPROFILER_CALLBACK_PHASE_EXIT) {
        }
    } // ROCPROFILER_CALLBACK_TRACING_MEMORY_COPY
}

#if 0
void RocprofLogger::api_callback(
    uint32_t domain,
    uint32_t cid,
    const void* callback_data,
    void* arg) {
  RocprofLogger* dis = &singleton();

  if (domain == ACTIVITY_DOMAIN_HIP_API && dis->loggedIds_.contains(cid)) {
    const hip_api_data_t* data = (const hip_api_data_t*)(callback_data);

    // Pack callbacks into row structures

    thread_local std::unordered_map<activity_correlation_id_t, timespec>
        timestamps;

    if (data->phase == ACTIVITY_API_PHASE_ENTER) {
      timespec timestamp;
      clock_gettime(CLOCK_MONOTONIC, &timestamp); // record proper clock
      timestamps[data->correlation_id] = timestamp;
    } else { // (data->phase == ACTIVITY_API_PHASE_EXIT)
      timespec startTime;
      startTime = timestamps[data->correlation_id];
      timestamps.erase(data->correlation_id);
      timespec endTime;
      clock_gettime(CLOCK_MONOTONIC, &endTime); // record proper clock

      switch (cid) {
        case HIP_API_ID_hipLaunchKernel:
        case HIP_API_ID_hipExtLaunchKernel:
        case HIP_API_ID_hipLaunchCooperativeKernel: // Should work here
        {
          s_flush.reportCorrelation(data->correlation_id);
          auto& args = data->args.hipLaunchKernel;
          roctracerKernelRow* row = new roctracerKernelRow(
              data->correlation_id,
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
              args.stream);
          insert_row_to_buffer(row);
        } break;
        case HIP_API_ID_hipHccModuleLaunchKernel:
        case HIP_API_ID_hipModuleLaunchKernel:
        case HIP_API_ID_hipExtModuleLaunchKernel: {
          s_flush.reportCorrelation(data->correlation_id);
          auto& args = data->args.hipModuleLaunchKernel;
          roctracerKernelRow* row = new roctracerKernelRow(
              data->correlation_id,
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
              args.stream);
          insert_row_to_buffer(row);
        } break;
        case HIP_API_ID_hipLaunchCooperativeKernelMultiDevice:
        case HIP_API_ID_hipExtLaunchMultiKernelMultiDevice:
#if 0
          {
            auto &args = data->args.hipLaunchCooperativeKernelMultiDevice.launchParamsList__val;
            roctracerKernelRow* row = new roctracerKernelRow(
              data->correlation_id,
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
            insert_row_to_buffer(row);
          }
#endif
          break;
        case HIP_API_ID_hipMalloc: {
          roctracerMallocRow* row = new roctracerMallocRow(
              data->correlation_id,
              domain,
              cid,
              processId(),
              systemThreadId(),
              timespec_to_ns(startTime),
              timespec_to_ns(endTime),
              data->args.hipMalloc.ptr__val,
              data->args.hipMalloc.size);
          insert_row_to_buffer(row);
        } break;
        case HIP_API_ID_hipFree: {
          roctracerMallocRow* row = new roctracerMallocRow(
              data->correlation_id,
              domain,
              cid,
              processId(),
              systemThreadId(),
              timespec_to_ns(startTime),
              timespec_to_ns(endTime),
              data->args.hipFree.ptr,
              0);
          insert_row_to_buffer(row);
        } break;
        case HIP_API_ID_hipMemcpy: {
          auto& args = data->args.hipMemcpy;
          roctracerCopyRow* row = new roctracerCopyRow(
              data->correlation_id,
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
              static_cast<hipStream_t>(0) // use placeholder?
          );
          insert_row_to_buffer(row);
        } break;
        case HIP_API_ID_hipMemcpyAsync:
        case HIP_API_ID_hipMemcpyWithStream: {
          auto& args = data->args.hipMemcpyAsync;
          roctracerCopyRow* row = new roctracerCopyRow(
              data->correlation_id,
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
              args.stream);
          insert_row_to_buffer(row);
        } break;
        default: {
          roctracerRow* row = new roctracerRow(
              data->correlation_id,
              domain,
              cid,
              processId(),
              systemThreadId(),
              timespec_to_ns(startTime),
              timespec_to_ns(endTime));
          insert_row_to_buffer(row);
        } break;
      } // switch
      // External correlation
      for (int it = CorrelationDomain::begin; it < CorrelationDomain::end;
           ++it) {
        if (t_externalIds[it].size() > 0) {
          std::lock_guard<std::mutex> lock(dis->externalCorrelationsMutex_);
          dis->externalCorrelations_[it].emplace_back(
              data->correlation_id, t_externalIds[it].back());
        }
      }
    } // phase exit
  }
}
#endif

#if 0
void RocprofLogger::activity_callback(
    const char* begin,
    const char* end,
    void* arg) {
  // Log latest completed correlation id.  Used to ensure we have flushed all
  // data on stop
  std::unique_lock<std::mutex> lock(s_flush.mutex_);
  const roctracer_record_t* record = (const roctracer_record_t*)(begin);
  const roctracer_record_t* end_record = (const roctracer_record_t*)(end);

  while (record < end_record) {
    if (record->correlation_id > s_flush.maxCompletedCorrelationId_) {
      s_flush.maxCompletedCorrelationId_ = record->correlation_id;
    }
    roctracerAsyncRow* row = new roctracerAsyncRow(
        record->correlation_id,
        record->domain,
        record->kind,
        record->op,
        record->device_id,
        record->queue_id,
        record->begin_ns,
        record->end_ns,
        ((record->kind == HIP_OP_DISPATCH_KIND_KERNEL_) ||
         (record->kind == HIP_OP_DISPATCH_KIND_TASK_))
            ? demangle(record->kernel_name)
            : std::string());
    insert_row_to_buffer(row);
    roctracer_next_record(record, &record);
  }
}
#endif

void RocprofLogger::startLogging() {
  if (!registered_) {
  }

  externalCorrelationEnabled_ = true;
  logging_ = true;
  rocprofiler_start_context(s->context);
}

void RocprofLogger::stopLogging() {
  if (logging_ == false)
    return;
  logging_ = false;

// FIXME make this work
#if 0
  hipError_t err = hipDeviceSynchronize();
  if (err != hipSuccess) {
    LOG(ERROR) << "hipDeviceSynchronize failed with code " << err;
  }
  roctracer_flush_activity_expl(hccPool_);

  // If we are stopping the tracer, implement reliable flushing
  std::unique_lock<std::mutex> lock(s_flush.mutex_);

  auto correlationId =
      s_flush.maxCorrelationId_.load(); // load ending id from the running max

  // Poll on the worker finding the final correlation id
  int timeout = 50;
  while ((s_flush.maxCompletedCorrelationId_ < correlationId) && --timeout) {
    lock.unlock();
    roctracer_flush_activity_expl(hccPool_);
    usleep(1000);
    lock.lock();
  }
#endif

  rocprofiler_stop_context(s->context);
}

void RocprofLogger::endTracing() {
// FIXME - needed?
#if 0
  if (registered_ == true) {
    roctracer_disable_domain_callback(ACTIVITY_DOMAIN_HIP_API);
    // roctracer_disable_domain_callback(ACTIVITY_DOMAIN_ROCTX);

    roctracer_disable_domain_activity(ACTIVITY_DOMAIN_HCC_OPS);
    roctracer_close_pool_expl(hccPool_);
    hccPool_ = nullptr;
  }
#endif
}

//
// ApiIdList
//   Jump through some extra hoops
//
//
RocprofApiIdList::RocprofApiIdList(buffer_name_info &names)
: nameMap_()
{
    auto &hipapis = names[ROCPROFILER_CALLBACK_TRACING_HIP_RUNTIME_API].operations;

    for (size_t i = 0; i < hipapis.size(); ++i) {
        nameMap_.emplace(hipapis[i], i);
    }
}

uint32_t RocprofApiIdList::mapName(const std::string &apiName)
{
    auto it = nameMap_.find(apiName);
    if (it != nameMap_.end()) {
        return it->second;
    }
    return 0;
}

std::vector<rocprofiler_tracing_operation_t> RocprofApiIdList::allEnabled()
{
    std::vector<rocprofiler_tracing_operation_t> oplist;
    for (auto &it : nameMap_) {
        if (contains(it.second))
            oplist.push_back(it.second);
    }
    return oplist;
}

//
//
//
//
//


