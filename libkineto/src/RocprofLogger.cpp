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
    using rocprofiler::sdk::callback_name_info;
    using agent_info_map_t = std::unordered_map<uint64_t, rocprofiler_agent_v0_t>;

    // extract copy args
    struct copy_args {
        const char *dst {""};
        const char *src {""};
        size_t size {0};
        const char* kindStr {""};
        hipMemcpyKind kind {hipMemcpyDefault};
        const char *stream {""};
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
                    args.kindStr = arg_value_str;
                    args.kind = *(reinterpret_cast<const hipMemcpyKind*>(arg_value_addr));
                }
                else if (strcmp("stream", arg_name) == 0) {
                    args.stream = arg_value_str;
                }
                return 0;
            };

    // extract kernel args
    struct kernel_args {
        const char *stream;
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
                    args.stream = arg_value_str;
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

  thread_local std::unordered_map<uint64_t, timespec> timestamps;

  if (record.kind == ROCPROFILER_CALLBACK_TRACING_HIP_RUNTIME_API) {
fprintf(stderr, "%ld: HIP_RUNTIME_API %d %s\n", record.correlation_id.internal, record.phase, std::string(s->name_info[record.kind][record.operation]).c_str());
    if (record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER) {
      timespec timestamp;
      clock_gettime(CLOCK_MONOTONIC, &timestamp); // record proper clock
      timestamps[record.correlation_id.internal] = timestamp;

      //---- Capture api args for copy and kernel apis
      // These will be used during dispatch and copy callbacks to complete records
      if (isCopyApi(record.operation)) {
        rocprofiler_iterate_callback_tracing_kind_operation_args(
          record, extract_copy_args, 1/*max_deref*/
          , &s->copyargs[record.correlation_id.internal]);
      }
      if (isKernelApi(record.operation)) {
        rocprofiler_iterate_callback_tracing_kind_operation_args(
            record, extract_kernel_args, 1/*max_deref*/
            , &s->kernelargs[record.correlation_id.internal]);
      }
      //-----------------------------------------------

    } // ROCPROFILER_CALLBACK_PHASE_ENTER
    else {  // ROCPROFILER_CALLBACK_PHASE_EXIT
//fprintf(stderr, "%s\n", std::string(s->name_info[record.kind][record.operation]).c_str());
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
        roctracerMallocRow* row = new roctracerMallocRow(
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
        roctracerRow* row = new roctracerRow(
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
fprintf(stderr, "KERNEL_DISPATCH %d (kind = %d  operation = %d)\n", record.phase, record.kind, record.operation);
fprintf(stderr, "%s\n", std::string(s->name_info[record.kind][record.operation]).c_str());
    if (record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER) {
        ;
    }
    else if (record.phase == ROCPROFILER_CALLBACK_PHASE_EXIT) {
      auto &dispatch = *(static_cast<rocprofiler_callback_tracing_kernel_dispatch_data_t*>
                        (record.payload));
      auto &info = dispatch.dispatch_info;
 
      // Lookup the stream from the enclosing hip call.  It's not provided in the dispatch record
      std::string stream;
      if (s->kernelargs.count(record.correlation_id.internal) > 0) {
        // This row can be missing.  Some copy api dispatch kernels under the hood
        auto &kargs = s->kernelargs.at(record.correlation_id.internal);
        stream = kargs.stream;
      }
      else if (s->copyargs.count(record.correlation_id.internal) > 0) {
        // Grab the stream from the copy row instead
        auto &cargs = s->copyargs.at(record.correlation_id.internal);
        stream = cargs.stream;
      }

      // fetch up the timestamps
      timespec startTime;
      startTime = timestamps[record.correlation_id.internal];
      timespec endTime;
      clock_gettime(CLOCK_MONOTONIC, &endTime); // record proper clock

      roctracerKernelRow* row = new roctracerKernelRow(
        record.correlation_id.internal,
        record.kind,
        record.operation,
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
        //stream);
        nullptr);  // FIXME stream
      insert_row_to_buffer(row);
    }
    else if (record.phase == ROCPROFILER_CALLBACK_PHASE_NONE) {
      // completion callback - runtime thread
      auto &dispatch = *(static_cast<rocprofiler_callback_tracing_kernel_dispatch_data_t*>(record.payload));
      auto &info = dispatch.dispatch_info;

      roctracerAsyncRow* row = new roctracerAsyncRow(
      record.correlation_id.internal,
      record.kind,
      record.operation,
      record.operation,	// FIXME: domain is?
      //record->correlation_id,
      //record->domain,
      //record->kind,
      //record->op,
      s->agents.at(info.agent_id.handle).logical_node_type_id,
      info.queue_id.handle,
      dispatch.start_timestamp,
      dispatch.end_timestamp,
      s->kernel_names.at(info.kernel_id));
    insert_row_to_buffer(row);
    }
  } // ROCPROFILER_CALLBACK_TRACING_KERNEL_DISPATCH

  else if (record.kind == ROCPROFILER_CALLBACK_TRACING_MEMORY_COPY) {
fprintf(stderr, "(%d::%d) MEMORY_COPY %d (kind = %d  operation = %d)\n", processId(), systemThreadId(), record.phase, record.kind, record.operation);
fprintf(stderr, "%s\n", std::string(s->name_info[record.kind][record.operation]).c_str());
    if (record.phase == ROCPROFILER_CALLBACK_PHASE_EXIT) {
      auto &copy = *(static_cast<rocprofiler_callback_tracing_memory_copy_data_t*>(record.payload));

      // Fetch args from the enclosing hip call
      // FIXME async?  May need to remove it here rather than above
      auto &args = s->copyargs.at(record.correlation_id.internal);

      roctracerCopyRow* row = new roctracerCopyRow(
        record.correlation_id.internal,
        record.kind,
        record.operation,
        processId(),
        systemThreadId(),
        copy.start_timestamp,
        copy.end_timestamp,
        args.src,
        args.dst,
        args.size,
        args.kind,
        //args.stream);
        nullptr);  // FIXME stream
      insert_row_to_buffer(row);
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
//
//


