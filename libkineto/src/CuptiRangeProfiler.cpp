/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <Logger.h>
#include <functional>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>

// TODO(T90238193)
// @lint-ignore-every CLANGTIDY facebook-hte-RelativeInclude
#include "output_base.h"
#include "CuptiRangeProfiler.h"
#include "CuptiRangeProfilerConfig.h"
#include "Demangle.h"

namespace KINETO_NAMESPACE {

const ActivityType kProfActivityType = ActivityType::CUDA_PROFILER_RANGE;
const std::set<ActivityType> kSupportedActivities{kProfActivityType};

const std::string kProfilerName{"CuptiRangeProfiler"};

static ICuptiRBProfilerSessionFactory& getFactory() {
  static CuptiRBProfilerSessionFactory factory_;
  return factory_;
}

/* ----------------------------------------
 * Implement CuptiRangeProfilerSession
 * ----------------------------------------
 */

namespace {

CuptiProfilerPrePostCallback cuptiProfilerPreRunCb;
CuptiProfilerPrePostCallback cuptiProfilerPostRunCb;


/* Following are aliases to a set of CUPTI metrics that can be
 * used to derived measures like FLOPs etc.
 */
std::unordered_map<std::string, std::vector<std::string>> kDerivedMetrics = {
  {"kineto__cuda_core_flops", {
    "smsp__sass_thread_inst_executed_op_dadd_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_dfma_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_dmul_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_hadd_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_hfma_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_hmul_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum"}},
  {"kineto__tensor_core_insts", {
    "sm__inst_executed_pipe_tensor.sum"}},
};

} // namespace;


CuptiRangeProfilerSession::CuptiRangeProfilerSession(
    const Config& config, ICuptiRBProfilerSessionFactory& factory) {

  // CUPTI APIs can conflict with other monitoring systems like DCGM
  // or NSight / NVProf. The pre and post run hooks enable users to
  // potentially pause other tools like DCGM.
  // By the way consider adding some delay while using dcgmpause() so
  // the change takes effect inside the driver.
  if (cuptiProfilerPreRunCb) {
    cuptiProfilerPreRunCb();
  }

  const CuptiRangeProfilerConfig& cupti_config =
    CuptiRangeProfilerConfig::get(config);

  std::vector<std::string> cupti_metrics;
  const auto& requested_metrics = cupti_config.activitiesCuptiMetrics();

  for (const auto& metric : requested_metrics) {
    auto it = kDerivedMetrics.find(metric);
    if (it != kDerivedMetrics.end()) {
      // add all the fundamental metrics
      for (const auto& m : it->second) {
        cupti_metrics.push_back(m);
      }
    } else {
      cupti_metrics.push_back(metric);
    }
  }

  // Capture metrics per kernel implies using auto-range mode
  if (cupti_config.cuptiProfilerPerKernel()) {
    rangeType_ = CUPTI_AutoRange;
    replayType_ = CUPTI_KernelReplay;
  }

  LOG(INFO) << "Configuring " << cupti_metrics.size()
            << " CUPTI metrics";

  int max_ranges = cupti_config.cuptiProfilerMaxRanges();
  for (const auto& m : cupti_metrics) {
    LOG(INFO) << "    " << m;
  }

  CuptiRangeProfilerOptions opts;
  opts.metricNames = cupti_metrics;
  opts.deviceId = 0;
  opts.maxRanges = max_ranges;
  opts.numNestingLevels = 1;
  opts.cuContext = nullptr;
  opts.unitTest = false;

  for (auto device_id : CuptiRBProfilerSession::getActiveDevices()) {
    LOG(INFO) << "Init CUPTI range profiler on gpu = " << device_id
              << " max ranges = " << max_ranges;
    opts.deviceId = device_id;
    profilers_.push_back(factory.make(opts));
  }
}

void CuptiRangeProfilerSession::start() {
  for (auto& profiler: profilers_) {
    // user range or auto range
    profiler->asyncStartAndEnable(rangeType_, replayType_);
  }
}

void CuptiRangeProfilerSession::stop() {
  for (auto& profiler: profilers_) {
    profiler->disableAndStop();
  }
}

void CuptiRangeProfilerSession::addRangeEvents(
    const CuptiProfilerResult& result,
    const CuptiRBProfilerSession* profiler) {
  const auto& metricNames = result.metricNames;
  auto& activities = traceBuffer_.activities;
  bool use_kernel_names = false;
  int num_kernels = 0;

  if (rangeType_ == CUPTI_AutoRange) {
    use_kernel_names = true;
    num_kernels = profiler->getKernelNames().size();
    if (num_kernels != result.rangeVals.size()) {
      LOG(WARNING) << "Number of kernels tracked does not match the "
                   << " number of ranges collected"
                   << " kernel names size = " << num_kernels
                   << " vs ranges = " << result.rangeVals.size();
    }
  }

  // the actual times do not really matter here so
  // we can just split the total span up per range
  int64_t startTime = traceBuffer_.span.startTime,
          duration = traceBuffer_.span.endTime - startTime,
          interval = duration / result.rangeVals.size();

  int ridx = 0;
  for (const auto& measurement : result.rangeVals) {
    bool use_kernel_as_range = use_kernel_names && (ridx < num_kernels);
    traceBuffer_.emplace_activity(
        traceBuffer_.span,
        kProfActivityType,
        use_kernel_as_range ?
          demangle(profiler->getKernelNames()[ridx]) :
          measurement.rangeName
    );
    auto& event = activities.back();
    event->startTime = startTime + interval * ridx;
    event->endTime = startTime + interval * (ridx + 1);
    event->device = profiler->deviceId();

    // add metadata per counter
    for (int i = 0; i < metricNames.size(); i++) {
      event->addMetadata(metricNames[i], measurement.values[i]);
    }
    ridx++;
  }
}

void CuptiRangeProfilerSession::processTrace(ActivityLogger& logger) {
  if (profilers_.size() == 0) {
    LOG(WARNING) << "Nothing to report";
    return;
  }

  traceBuffer_.span = profilers_[0]->getProfilerTraceSpan();
  for (auto& profiler: profilers_) {
    bool verbose = VLOG_IS_ON(1);
    auto result = profiler->evaluateMetrics(verbose);

    LOG(INFO) << "Profiler Range data on gpu = " << profiler->deviceId();
    if (result.rangeVals.size() == 0) {
      LOG(WARNING) << "Skipping profiler results on gpu = "
                   << profiler->deviceId() << " as 0 ranges were found";
      continue;
    }
    addRangeEvents(result, profiler.get());
  }

  for (const auto& event : traceBuffer_.activities) {
    static_assert(
        std::is_same<
            std::remove_reference<decltype(event)>::type,
            const std::unique_ptr<GenericTraceActivity>>::value,
        "handleActivity is unsafe and relies on the caller to maintain not "
        "only lifetime but also address stability.");
    logger.handleActivity(*event);
  }

  LOG(INFO) << "CUPTI Range Profiler added " << traceBuffer_.activities.size()
            << " events";

  if (cuptiProfilerPostRunCb) {
    cuptiProfilerPostRunCb();
  }
}

std::vector<std::string> CuptiRangeProfilerSession::errors() {
  return {};
}

std::unique_ptr<DeviceInfo> CuptiRangeProfilerSession::getDeviceInfo() {
  return {};
}

std::vector<ResourceInfo> CuptiRangeProfilerSession::getResourceInfos() {
  return {};
}

/* ----------------------------------------
 * Implement CuptiRangeProfiler
 * ----------------------------------------
 */
CuptiRangeProfiler::CuptiRangeProfiler()
  : CuptiRangeProfiler(getFactory()) {}

CuptiRangeProfiler::CuptiRangeProfiler(ICuptiRBProfilerSessionFactory& factory)
  : factory_(factory) {}

void CuptiRangeProfiler::setPreRunCallback(
    CuptiProfilerPrePostCallback fn) {
  cuptiProfilerPreRunCb = fn;
}

void CuptiRangeProfiler::setPostRunCallback(
    CuptiProfilerPrePostCallback fn) {
  cuptiProfilerPostRunCb = fn;
}

const std::string& CuptiRangeProfiler::name() const {
  return kProfilerName;
}

const std::set<ActivityType>& CuptiRangeProfiler::availableActivities()
    const {
  return kSupportedActivities;
}

// TODO remove activity_types from this interface in the future
std::unique_ptr<IActivityProfilerSession> CuptiRangeProfiler::configure(
    const std::set<ActivityType>& /*activity_types*/,
    const Config& config) {
  const auto& activity_types_ = config.selectedActivityTypes();
  if (activity_types_.find(kProfActivityType) == activity_types_.end()) {
    return nullptr;
  }
  bool has_gpu_event_types = (
      activity_types_.count(ActivityType::GPU_MEMCPY) +
      activity_types_.count(ActivityType::GPU_MEMSET) +
      activity_types_.count(ActivityType::CONCURRENT_KERNEL)
    ) > 0;

  if (has_gpu_event_types) {
    LOG(WARNING) << kProfilerName << " cannot run in combination with"
                << " other cuda activity profilers, please configure"
                << " with cuda_profiler_range and optionally cpu_op/user_annotations";
    return nullptr;
  }

  return std::make_unique<CuptiRangeProfilerSession>(config, factory_);
}

std::unique_ptr<IActivityProfilerSession>
CuptiRangeProfiler::configure(
    int64_t /*ts_ms*/,
    int64_t /*duration_ms*/,
    const std::set<ActivityType>& activity_types,
    const Config& config) {
  return configure(activity_types, config);
}

/* ----------------------------------------
 * CuptiRangeProfilerInit :
 *    a small wrapper class that ensure the range profiler is created and
 *  initialized.
 * ----------------------------------------
 */
CuptiRangeProfilerInit::CuptiRangeProfilerInit() {
  // register config
  CuptiRangeProfilerConfig::registerFactory();

#ifdef HAS_CUPTI
  success = CuptiRBProfilerSession::staticInit();
#endif

  if (!success) {
    return;
  }

  // Register the activity profiler instance with libkineto api
  api().registerProfilerFactory([&]() {
    return std::make_unique<CuptiRangeProfiler>();
  });
}

} // namespace KINETO_NAMESPACE
