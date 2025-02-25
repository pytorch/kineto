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
    using kernel_name_map_t = std::unordered_map<rocprofiler_kernel_id_t, std::string>;
    using rocprofiler::sdk::buffer_name_info;
    using rocprofiler::sdk::callback_name_info;
    using agent_info_map_t = std::unordered_map<uint64_t, rocprofiler_agent_v0_t>;

    // extract copy args
    struct copy_args {
        const char *dst {""};
        const char *src {""};
        size_t size {0};
        const char* copyKindStr {""};
        hipMemcpyKind copyKind {hipMemcpyDefault};
        hipStream_t stream {nullptr};
        rocprofiler_callback_tracing_kind_t kind;
        rocprofiler_tracing_operation_t operation;
    };
            auto extract_copy_args = [](rocprofiler_callback_tracing_kind_t,
                   rocprofiler_tracing_operation_t,
                   uint32_t          arg_num,
                   const void* const arg_value_addr,
                   int32_t           indirection_count,
                   const char*       arg_type,
                   const char*       arg_name,
                   const char*       arg_value_str,
                   int32_t           dereference_count,
                   void*             cb_data) -> int {

                auto &args = *(static_cast<copy_args*>(cb_data));
                if (strcmp("dst", arg_name) == 0) {
                    args.dst = arg_value_str;
                }
                else if (strcmp("src", arg_name) == 0) {
                    args.src = arg_value_str;
                }
                else if (strcmp("sizeBytes", arg_name) == 0) {
                    args.size = *(reinterpret_cast<const size_t*>(arg_value_addr));
                }
                else if (strcmp("kind", arg_name) == 0) {
                    args.copyKindStr = arg_value_str;
                    args.copyKind = *(reinterpret_cast<const hipMemcpyKind*>(arg_value_addr));
                }
                else if (strcmp("stream", arg_name) == 0) {
                    args.stream = *(reinterpret_cast<const hipStream_t*>(arg_value_addr));
                }
                return 0;
            };

    // extract kernel args
    struct kernel_args {
        //const char *stream;
        hipStream_t stream {nullptr};
        rocprofiler_callback_tracing_kind_t kind;
        rocprofiler_tracing_operation_t operation;
    };
            auto extract_kernel_args = [](rocprofiler_callback_tracing_kind_t,
                   rocprofiler_tracing_operation_t,
                   uint32_t          arg_num,
                   const void* const arg_value_addr,
                   int32_t           indirection_count,
                   const char*       arg_type,
                   const char*       arg_name,
                   const char*       arg_value_str,
                   int32_t           dereference_count,
                   void*             cb_data) -> int {

                if (strcmp("stream", arg_name) == 0) {
                    auto &args = *(static_cast<kernel_args*>(cb_data));
                    //args.stream = arg_value_str;
                    args.stream = *(reinterpret_cast<const hipStream_t*>(arg_value_addr));
                }
                return 0;
            };

    // extract malloc args
    struct malloc_args {
    const char *ptr;
        size_t size;
    };
            auto extract_malloc_args = [](rocprofiler_callback_tracing_kind_t,
                   rocprofiler_tracing_operation_t,
                   uint32_t          arg_num,
                   const void* const arg_value_addr,
                   int32_t           indirection_count,
                   const char*       arg_type,
                   const char*       arg_name,
                   const char*       arg_value_str,
                   int32_t           dereference_count,
                   void*             cb_data) -> int {

                auto &args = *(static_cast<malloc_args*>(cb_data));
                if (strcmp("ptr", arg_name) == 0) {
                    args.ptr = arg_value_str;
                }
                if (strcmp("size", arg_name) == 0) {
                    args.size = *(reinterpret_cast<const size_t*>(arg_value_addr));
                }
                return 0;
            };

    // copy api calls
    bool isCopyApi(uint32_t id) {
        switch (id) {
            case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpy:
            case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpy2D:
            case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpy2DAsync:
            case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpy2DFromArray:
            case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpy2DFromArrayAsync:
            case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpy2DToArray:
            case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpy2DToArrayAsync:
            case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpy3D:
            case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpy3DAsync:
            case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyAsync:
            case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyAtoH:
            case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyDtoD:
            case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyDtoDAsync:
            case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyDtoH:
            case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyDtoHAsync:
            case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyFromArray:
            case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyFromSymbol:
            case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyFromSymbolAsync:
            case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyHtoA:
            case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyHtoD:
            case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyHtoDAsync:
            case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyParam2D:
            case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyParam2DAsync:
            case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyPeer:
            case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyPeerAsync:
            case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyToArray:
            case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyToSymbol:
            case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyToSymbolAsync:
            case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyWithStream:
                return true;
                break;
            default:
                ;
       }
       return false;
    }

    // kernel api calls
    bool isKernelApi(uint32_t id) {
        switch (id) {
            case ROCPROFILER_HIP_RUNTIME_API_ID_hipExtLaunchKernel:
            case ROCPROFILER_HIP_RUNTIME_API_ID_hipExtLaunchMultiKernelMultiDevice:
            case ROCPROFILER_HIP_RUNTIME_API_ID_hipLaunchCooperativeKernel:
            case ROCPROFILER_HIP_RUNTIME_API_ID_hipLaunchCooperativeKernelMultiDevice:
            case ROCPROFILER_HIP_RUNTIME_API_ID_hipLaunchKernel:
            case ROCPROFILER_HIP_RUNTIME_API_ID_hipModuleLaunchCooperativeKernel:
            case ROCPROFILER_HIP_RUNTIME_API_ID_hipModuleLaunchCooperativeKernelMultiDevice:
            case ROCPROFILER_HIP_RUNTIME_API_ID_hipModuleLaunchKernel:
            case ROCPROFILER_HIP_RUNTIME_API_ID_hipExtModuleLaunchKernel:
            case ROCPROFILER_HIP_RUNTIME_API_ID_hipHccModuleLaunchKernel:
            case ROCPROFILER_HIP_RUNTIME_API_ID_hipLaunchCooperativeKernel_spt:
            case ROCPROFILER_HIP_RUNTIME_API_ID_hipLaunchKernel_spt:
                return true;
                break;
            default:
                ;
       }
       return false;
    }

    // malloc api calls
    bool isMallocApi(uint32_t id) {
        switch (id) {
            case ROCPROFILER_HIP_RUNTIME_API_ID_hipMalloc:
            case ROCPROFILER_HIP_RUNTIME_API_ID_hipFree:
                return true;
                break;
            default:
                ;
       }
       return false;
    }

    class RocprofApiIdList : public ApiIdList
    {
    public:
        RocprofApiIdList(callback_name_info &names);
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
    std::mutex kernel_lock;

    // Manage buffer name - #betterThanRoctracer
    callback_name_info name_info = {};

    // Agent info
    // <rocprofiler_profile_config_id_t.handle, rocprofiler_agent_v0_t>
    agent_info_map_t agents = {};

    std::map<uint64_t, kernel_args> kernelargs;
    std::map<uint64_t, copy_args> copyargs;

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
    RocprofLoggerShared::singleton();       // CRITICAL: static init

    id->name = "kineto";
    s->clientId = id;

    // return pointer to configure data
    return &s->cfg;
}

int RocprofLogger::toolInit(rocprofiler_client_finalize_t finialize_func, void* tool_data)
{
    // Gather api names
    s->name_info = rocprofiler::sdk::get_callback_tracing_names();

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
    rocprofiler_stop_context(s->context);

    return 0;
}

void RocprofLogger::toolFinialize(void* tool_data)
{
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

void RocprofLogger::insert_row_to_buffer(rocprofBase* row) {
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
        std::lock_guard<std::mutex> lock(s->kernel_lock);
            s->kernel_info.emplace(data->kernel_id, *data);
            s->kernel_names.emplace(data->kernel_id, demangle(data->kernel_name));
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

  thread_local std::unordered_map<uint64_t, timespec> timestamps;

  if (record.kind == ROCPROFILER_CALLBACK_TRACING_HIP_RUNTIME_API) {
    if (record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER) {
      timespec timestamp;
      clock_gettime(CLOCK_MONOTONIC, &timestamp); // record proper clock
      timestamps[record.correlation_id.internal] = timestamp;

      //---- Capture api args for copy and kernel apis
      // These will be used during dispatch and copy callbacks to complete records
      if (isCopyApi(record.operation)) {
        auto &args = s->copyargs[record.correlation_id.internal];
        rocprofiler_iterate_callback_tracing_kind_operation_args(
          record, extract_copy_args, 1/*max_deref*/
          , &args);
        args.kind = record.kind;
        args.operation = record.operation;
      }
      if (isKernelApi(record.operation)) {
        auto &args = s->kernelargs[record.correlation_id.internal];
        rocprofiler_iterate_callback_tracing_kind_operation_args(
            record, extract_kernel_args, 1/*max_deref*/
            , &args);
        args.kind = record.kind;
        args.operation = record.operation;
      }
      //-----------------------------------------------

    } // ROCPROFILER_CALLBACK_PHASE_ENTER
    else {  // ROCPROFILER_CALLBACK_PHASE_EXIT
      timespec startTime;
      startTime = timestamps[record.correlation_id.internal];
      timestamps.erase(record.correlation_id.internal);
      timespec endTime;
      clock_gettime(CLOCK_MONOTONIC, &endTime); // record proper clock

      // Kernel Launch Records
      if (isKernelApi(record.operation)) {
        // handled in dispatch callback
        s->kernelargs.erase(record.correlation_id.internal);
      }
      // Copy Records
      else if (isCopyApi(record.operation)) {
        // handled in copy callback
        // FIXME: do not remove here.  Used after the async operation
        // DO it anyway, wait for crash,  async SDMA should assert below
        s->copyargs.erase(record.correlation_id.internal);
      }
      // Malloc Records
      else if (isMallocApi(record.operation)) {
        malloc_args args;
        args.size = 0;
        rocprofiler_iterate_callback_tracing_kind_operation_args(
          record, extract_malloc_args, 1/*max_deref*/
          , &args);
        rocprofMallocRow* row = new rocprofMallocRow(
          record.correlation_id.internal,
          record.kind,
          record.operation,
          processId(),
          systemThreadId(),
          timespec_to_ns(startTime),
          timespec_to_ns(endTime),
          args.ptr,
          args.size);
        insert_row_to_buffer(row);
      }
      // Default Records
      else {
        rocprofRow* row = new rocprofRow(
          record.correlation_id.internal,
          record.kind,
          record.operation,
          processId(),
          systemThreadId(),
          timespec_to_ns(startTime),
          timespec_to_ns(endTime));
        insert_row_to_buffer(row);
      }
    } // ROCPROFILER_CALLBACK_PHASE_EXIT
  }  // ROCPROFILER_CALLBACK_TRACING_HIP_RUNTIME_API

  else if (record.kind == ROCPROFILER_CALLBACK_TRACING_KERNEL_DISPATCH) {
    if (record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER) {
        ;
    }
    else if (record.phase == ROCPROFILER_CALLBACK_PHASE_EXIT) {
      auto &dispatch = *(static_cast<rocprofiler_callback_tracing_kernel_dispatch_data_t*>
                        (record.payload));
      auto &info = dispatch.dispatch_info;

      // Lookup the stream and apiname from the enclosing hip call.
      //  These are not provided in the dispatch record 
      hipStream_t stream;
      auto kind = record.kind;
      auto operation = record.operation;
      if (s->kernelargs.count(record.correlation_id.internal) > 0) {
        // This row can be missing.  Some copy api dispatch kernels under the hood
        auto &kargs = s->kernelargs.at(record.correlation_id.internal);
        stream = kargs.stream;
        kind = kargs.kind;
        operation = kargs.operation;
      }
      else if (s->copyargs.count(record.correlation_id.internal) > 0) {
        // Grab the stream from the copy row instead
        auto &cargs = s->copyargs.at(record.correlation_id.internal);
        stream = cargs.stream;
        kind = cargs.kind;
        operation = cargs.operation;
      }

      // fetch up the timestamps
      timespec startTime;
      startTime = timestamps[record.correlation_id.internal];
      timespec endTime;
      clock_gettime(CLOCK_MONOTONIC, &endTime); // record proper clock

      rocprofKernelRow* row = new rocprofKernelRow(
        record.correlation_id.internal,
        kind,
        operation,
        processId(),
        systemThreadId(),
        timespec_to_ns(startTime),
        timespec_to_ns(endTime),
        nullptr,
        nullptr,
        info.workgroup_size.x,
        info.workgroup_size.y,
        info.workgroup_size.z,
        info.grid_size.x,
        info.grid_size.y,
        info.grid_size.z,
        info.group_segment_size,
        stream);
      insert_row_to_buffer(row);
    }
    else if (record.phase == ROCPROFILER_CALLBACK_PHASE_NONE) {
      // completion callback - runtime thread
      auto &dispatch = *(static_cast<rocprofiler_callback_tracing_kernel_dispatch_data_t*>(record.payload));
      auto &info = dispatch.dispatch_info;

      std::lock_guard<std::mutex> lock(s->kernel_lock);

      rocprofAsyncRow* row = new rocprofAsyncRow(
      record.correlation_id.internal,
      record.kind,
      record.operation,
      record.operation,	// shared op - No longer a thing.  Placeholder
      s->agents.at(info.agent_id.handle).logical_node_type_id,
      info.queue_id.handle,
      dispatch.start_timestamp,
      dispatch.end_timestamp,
      s->kernel_names.at(info.kernel_id));
    insert_row_to_buffer(row);
    }
  } // ROCPROFILER_CALLBACK_TRACING_KERNEL_DISPATCH

  else if (record.kind == ROCPROFILER_CALLBACK_TRACING_MEMORY_COPY) {
    if (record.phase == ROCPROFILER_CALLBACK_PHASE_EXIT) {
      auto &copy = *(static_cast<rocprofiler_callback_tracing_memory_copy_data_t*>(record.payload));

      // Fetch args from the enclosing hip call
      // FIXME async?  May need to remove it here rather than above
      auto &args = s->copyargs.at(record.correlation_id.internal);

      rocprofCopyRow* row = new rocprofCopyRow(
        record.correlation_id.internal,
        args.kind,
        args.operation,
        processId(),
        systemThreadId(),
        copy.start_timestamp,
        copy.end_timestamp,
        args.src,
        args.dst,
        args.size,
        args.copyKind,
        args.stream);
      insert_row_to_buffer(row);
    }
  } // ROCPROFILER_CALLBACK_TRACING_MEMORY_COPY
}

std::string RocprofLogger::opString(rocprofiler_callback_tracing_kind_t kind
                                    , rocprofiler_tracing_operation_t op)
{
  return std::string(RocprofLoggerShared::singleton().name_info[kind][op]);
}

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

  // Flushing likely not required - using callbacks only

  rocprofiler_stop_context(s->context);
}

void RocprofLogger::endTracing() {
// This should be handled in RocprofLogger::toolFinialize
}

//
// ApiIdList
//   Jump through some extra hoops
//
//
RocprofApiIdList::RocprofApiIdList(callback_name_info &names)
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


