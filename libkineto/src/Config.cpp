/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "Config.h"

#include <stdlib.h>
#include <unistd.h>

#include <fmt/format.h>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <istream>
#include <ostream>
#include <sstream>

#include "Logger.h"

using namespace std::chrono;

using std::string;
using std::vector;

namespace KINETO_NAMESPACE {

constexpr milliseconds kDefaultSamplePeriodMsecs(1000);
constexpr milliseconds kDefaultMultiplexPeriodMsecs(1000);
constexpr milliseconds kDefaultActivitiesProfileDurationMSecs(500);
constexpr int kDefaultActivitiesExternalAPIIterations(3);
constexpr int kDefaultActivitiesExternalAPINetSizeThreshold(0);
constexpr int kDefaultActivitiesExternalAPIGpuOpCountThreshold(0);
constexpr int kDefaultActivitiesMaxGpuBufferSize(64 * 1024 * 1024);
constexpr seconds kDefaultActivitiesWarmupDurationSecs(15);
constexpr seconds kDefaultReportPeriodSecs(1);
constexpr int kDefaultSamplesPerReport(1);
constexpr int kDefaultMaxEventProfilersPerGpu(1);
constexpr seconds kMaxRequestAge(10);

// Event Profiler
const string kEventsKey = "EVENTS";
const string kMetricsKey = "METRICS";
const string kSamplePeriodKey = "SAMPLE_PERIOD_MSECS";
const string kMultiplexPeriodKey = "MULTIPLEX_PERIOD_MSECS";
const string kReportPeriodKey = "REPORT_PERIOD_SECS";
const string kSamplesPerReportKey = "SAMPLES_PER_REPORT";
const string kEventsLogFileKey = "EVENTS_LOG_FILE";
const string kEventsEnabledDevicesKey = "EVENTS_ENABLED_DEVICES";
const string kOnDemandDurationKey = "EVENTS_DURATION_SECS";
const string kMaxEventProfilersPerGpuKey = "MAX_EVENT_PROFILERS_PER_GPU";

// Activity Profiler
const string kActivitiesEnabledKey = "ACTIVITIES_ENABLED";
const string kActivityTypesKey = "ACTIVITY_TYPES";
const string kActivitiesLogFileKey = "ACTIVITIES_LOG_FILE";
const string kActivitiesDurationKey = "ACTIVITIES_DURATION_SECS";
const string kActivitiesDurationMsecsKey = "ACTIVITIES_DURATION_MSECS";
const string kActivitiesIterationsKey = "ACTIVITIES_ITERATIONS";
const string kActivitiesIterationsTargetKey = "ACTIVITIES_ITERATIONS_TARGET";
const string kActivitiesNetFilterKey = "ACTIVITIES_NET_FILTER";
const string kActivitiesMinNetSizeKey = "ACTIVITIES_MIN_NET_SIZE";
const string kActivitiesMinGpuOpCountKey = "ACTIVITIES_MIN_GPU_OP_COUNT";
const string kActivitiesWarmupDurationSecsKey = "ACTIVITIES_WARMUP_PERIOD_SECS";
const string kActivitiesMaxGpuBufferSizeKey =
    "ACTIVITIES_MAX_GPU_BUFFER_SIZE_MB";

// Valid configuration file entries for activity types
const string kActivityMemcpy = "gpu_memcpy";
const string kActivityMemset = "gpu_memset";
const string kActivityConcurrentKernel = "concurrent_kernel";
const string kActivityExternalCorrelation = "external_correlation";
const string kActivityRuntime = "cuda_runtime";

const string kDefaultLogFileFmt = "/tmp/libkineto_activities_{}.json";

// Common

// Client-side timestamp used for synchronized start across hosts for
// distributed workloads.
// Specified in milliseconds Unix time (milliseconds since epoch).
// To use, take a timestamp at request time as follows:
//    * C++: duration_cast<milliseconds>(
//               system_clock::now().time_since_epoch()).count()
//    * Python: int(time.time() * 1000)
//    * Bash: date +%s%3N
const string kRequestTimestampKey = "REQUEST_TIMESTAMP";

// Enable on-demand trigger via kill -USR2 <pid>
// When triggered in this way, /tmp/libkineto.conf will be used as config.
const string kEnableSigUsr2Key = "ENABLE_SIGUSR2";

// Verbose log level
// The actual glog is not used and --v and --vmodule has no effect.
// Instead set the verbose level and modules in the config file.
const string kLogVerboseLevelKey = "VERBOSE_LOG_LEVEL";
// By default, all modules will log verbose messages >= verboseLogLevel.
// But to reduce noise we can specify one or more modules of interest.
// A module is a C/C++ object file (source file name),
// Example argument: ActivityProfiler.cpp,output_json.cpp
const string kLogVerboseModulesKey = "VERBOSE_LOG_MODULES";

const string kConfigFileEnvVar = "KINETO_CONFIG";
const string kConfigFile = "/etc/libkineto.conf";

// Max devices supported on any system
constexpr uint8_t kMaxDevices = 8;

static std::map<std::string, std::function<AbstractConfig*(const Config&)>>&
configFactories() {
  static std::map<std::string, std::function<AbstractConfig*(const Config&)>>
      factories;
  return factories;
}

void Config::addConfigFactory(
    std::string name,
    std::function<AbstractConfig*(const Config&)> factory) {
  configFactories()[name] = factory;
}

static string defaultTraceFileName() {
  return fmt::format(kDefaultLogFileFmt, getpid());
}

Config::Config()
    : verboseLogLevel_(-1),
      samplePeriod_(kDefaultSamplePeriodMsecs),
      reportPeriod_(duration_cast<milliseconds>(kDefaultReportPeriodSecs)),
      samplesPerReport_(kDefaultSamplesPerReport),
      eventProfilerOnDemandDuration_(seconds(0)),
      eventProfilerMaxInstancesPerGpu_(kDefaultMaxEventProfilersPerGpu),
      multiplexPeriod_(kDefaultMultiplexPeriodMsecs),
      activityProfilerEnabled_(true),
      activitiesLogFile_(defaultTraceFileName()),
      activitiesMaxGpuBufferSize_(kDefaultActivitiesMaxGpuBufferSize),
      activitiesWarmupDuration_(kDefaultActivitiesWarmupDurationSecs),
      activitiesOnDemandDuration_(kDefaultActivitiesProfileDurationMSecs),
      activitiesExternalAPIIterations_(kDefaultActivitiesExternalAPIIterations),
      activitiesExternalAPINetSizeThreshold_(
          kDefaultActivitiesExternalAPINetSizeThreshold),
      activitiesExternalAPIGpuOpCountThreshold_(
          kDefaultActivitiesExternalAPIGpuOpCountThreshold),
      requestTimestamp_(milliseconds(0)),
      enableSigUsr2_(true) {
  for (const auto& p : configFactories()) {
    addFeature(p.first, p.second(*this));
  }
}

uint8_t Config::createDeviceMask(const string& val) {
  uint8_t res = 0;
  for (const auto& d : splitAndTrim(val, ',')) {
    res |= 1 << toIntRange(d, 0, kMaxDevices - 1);
  }
  return res;
}

const seconds Config::maxRequestAge() const {
  return kMaxRequestAge;
}

static char* printTime(time_point<system_clock> t, char* buf, int size) {
  std::time_t t_c = system_clock::to_time_t(t);
  std::tm lt;
  localtime_r(&t_c, &lt);
  std::strftime(buf, size, "%H:%M:%S", &lt);
  return buf;
}

static time_point<system_clock> handleRequestTimestamp(int64_t ms) {
  auto t = time_point<system_clock>(milliseconds(ms));
  auto now = system_clock::now();
  char buf[32];
  if (t > now) {
    throw std::invalid_argument(fmt::format(
        "Invalid {}: {} - time is in future",
        kRequestTimestampKey,
        printTime(t, buf, sizeof(buf))));
  } else if ((now - t) > kMaxRequestAge) {
    throw std::invalid_argument(fmt::format(
        "Invalid {}: {} - time is more than {}s in the past",
        kRequestTimestampKey,
        printTime(t, buf, sizeof(buf)),
        kMaxRequestAge.count()));
  }
  return t;
}

void Config::addActivityTypes(
  const std::vector<std::string>& selected_activities) {
  if (selected_activities.size() > 0) {
    for (const auto& activity : selected_activities) {
      if (activity == "") {
        continue;
      } else if (activity == kActivityMemcpy) {
        selectedActivityTypes_.insert(ActivityType::GPU_MEMCPY);
      } else if (activity == kActivityMemset) {
        selectedActivityTypes_.insert(ActivityType::GPU_MEMSET);
      } else if (activity == kActivityConcurrentKernel) {
        selectedActivityTypes_.insert(ActivityType::CONCURRENT_KERNEL);
      } else if (activity == kActivityExternalCorrelation) {
        selectedActivityTypes_.insert(ActivityType::EXTERNAL_CORRELATION);
      } else if (activity == kActivityRuntime) {
        selectedActivityTypes_.insert(ActivityType::CUDA_RUNTIME);
      } else {
        throw std::invalid_argument(fmt::format(
          "Invalid activity type selected: {}",
          activity
        ));
      }
    }
  }
}

bool Config::handleOption(const std::string& name, std::string& val) {
  // Event Profiler
  if (name == kEventsKey) {
    vector<string> event_names = splitAndTrim(val, ',');
    eventNames_.insert(event_names.begin(), event_names.end());
  } else if (name == kMetricsKey) {
    vector<string> metric_names = splitAndTrim(val, ',');
    metricNames_.insert(metric_names.begin(), metric_names.end());
  } else if (name == kActivityTypesKey) {
    vector<string> activity_types = splitAndTrim(toLower(val), ',');
    addActivityTypes(activity_types);
  } else if (name == kSamplePeriodKey) {
    samplePeriod_ = milliseconds(toInt32(val));
  } else if (name == kMultiplexPeriodKey) {
    multiplexPeriod_ = milliseconds(toInt32(val));
  } else if (name == kReportPeriodKey) {
    setReportPeriod(seconds(toInt32(val)));
  } else if (name == kSamplesPerReportKey) {
    samplesPerReport_ = toInt32(val);
  } else if (name == kEventsLogFileKey) {
    eventLogFile_ = val;
  } else if (name == kEventsEnabledDevicesKey) {
    eventProfilerDeviceMask_ = createDeviceMask(val);
  } else if (name == kOnDemandDurationKey) {
    eventProfilerOnDemandDuration_ = seconds(toInt32(val));
    eventProfilerOnDemandTimestamp_ = timestamp();
  } else if (name == kMaxEventProfilersPerGpuKey) {
    eventProfilerMaxInstancesPerGpu_ = toInt32(val);
  }

  // Activity Profiler
  else if (name == kActivitiesDurationKey) {
    activitiesOnDemandDuration_ =
        duration_cast<milliseconds>(seconds(toInt32(val)));
    activitiesOnDemandTimestamp_ = timestamp();
  } else if (name == kActivitiesDurationMsecsKey) {
    activitiesOnDemandDuration_ = milliseconds(toInt32(val));
    activitiesOnDemandTimestamp_ = timestamp();
  } else if (name == kActivitiesIterationsKey) {
    activitiesExternalAPIIterations_ = toInt32(val);
    activitiesOnDemandTimestamp_ = timestamp();
  } else if (name == kActivitiesIterationsTargetKey) {
    activitiesExternalAPIIterationsTarget_ = val;
  } else if (name == kActivitiesNetFilterKey) {
    activitiesExternalAPIFilter_ = splitAndTrim(val, ',');
  } else if (name == kActivitiesMinNetSizeKey) {
    activitiesExternalAPINetSizeThreshold_ = toInt32(val);
  } else if (name == kActivitiesMinGpuOpCountKey) {
    activitiesExternalAPIGpuOpCountThreshold_ = toInt32(val);
  } else if (name == kLogVerboseLevelKey) {
    verboseLogLevel_ = toInt32(val);
  } else if (name == kLogVerboseModulesKey) {
    verboseLogModules_ = splitAndTrim(val, ',');
  } else if (name == kActivitiesEnabledKey) {
    activityProfilerEnabled_ = toBool(val);
  } else if (name == kActivitiesLogFileKey) {
    activitiesLogFile_ = val;
    activitiesOnDemandTimestamp_ = timestamp();
  } else if (name == kActivitiesMaxGpuBufferSizeKey) {
    activitiesMaxGpuBufferSize_ = toInt32(val) * 1024 * 1024;
  } else if (name == kActivitiesWarmupDurationSecsKey) {
    activitiesWarmupDuration_ = seconds(toInt32(val));
  }

  // Common
  else if (name == kRequestTimestampKey) {
    requestTimestamp_ = handleRequestTimestamp(toInt64(val));
  } else if (name == kEnableSigUsr2Key) {
    enableSigUsr2_ = toBool(val);
  } else {
    return false;
  }
  return true;
}

std::chrono::milliseconds Config::activitiesOnDemandDurationDefault() const {
  return kDefaultActivitiesProfileDurationMSecs;
};

void Config::updateActivityProfilerRequestReceivedTime() {
  activitiesOnDemandTimestamp_ = high_resolution_clock::now();
}

void Config::setClientDefaults() {
  AbstractConfig::setClientDefaults();
  activitiesLogToMemory_ = true;
}

void Config::validate() {
  if (samplePeriod_.count() == 0) {
    LOG(WARNING) << "Sample period must be greater than 0, setting to 1ms";
    samplePeriod_ = milliseconds(1);
  }

  if (multiplexPeriod_ < samplePeriod_) {
    LOG(WARNING) << "Multiplex period can not be smaller "
                 << "than sample period";
    LOG(WARNING) << "Setting multiplex period to " << samplePeriod_.count()
                 << "ms";
    multiplexPeriod_ = samplePeriod_;
  }

  if ((multiplexPeriod_ % samplePeriod_).count() != 0) {
    LOG(WARNING) << "Multiplex period must be a "
                 << "multiple of sample period";
    multiplexPeriod_ = alignUp(multiplexPeriod_, samplePeriod_);
    LOG(WARNING) << "Setting multiplex period to " << multiplexPeriod_.count()
                 << "ms";
  }

  if ((reportPeriod_ % multiplexPeriod_).count() != 0 ||
      reportPeriod_.count() == 0) {
    LOG(WARNING) << "Report period must be a "
                 << "multiple of multiplex period";
    reportPeriod_ = alignUp(reportPeriod_, multiplexPeriod_);
    LOG(WARNING) << "Setting report period to " << reportPeriod_.count()
                 << "ms";
  }

  if (samplesPerReport_ < 1) {
    LOG(WARNING) << "Samples per report must be in the range "
                 << "[1, report period / sample period]";
    LOG(WARNING) << "Setting samples per report to 1";
    samplesPerReport_ = 1;
  }

  int max_samples_per_report = reportPeriod_ / samplePeriod_;
  if (samplesPerReport_ > max_samples_per_report) {
    LOG(WARNING) << "Samples per report must be in the range "
                 << "[1, report period / sample period] ([1, "
                 << reportPeriod_.count() << "ms / " << samplePeriod_.count()
                 << "ms = " << max_samples_per_report << "])";
    LOG(WARNING) << "Setting samples per report to " << max_samples_per_report;
    samplesPerReport_ = max_samples_per_report;
  }

  if (selectedActivityTypes_.size() == 0) {
    selectDefaultActivityTypes();
  }
}

void Config::setReportPeriod(milliseconds msecs) {
  reportPeriod_ = msecs;
}

void Config::printActivityProfilerConfig(std::ostream& s) const {
  s << "Log file: " << activitiesLogFile() << std::endl;
  s << fmt::format(
           "Net filter: {}",
           fmt::join(activitiesOnDemandExternalFilter(), ", "))
    << std::endl;
  s << "Target net for iteration count: " << activitiesOnDemandExternalTarget()
    << std::endl;
  s << "Net Iterations: " << activitiesOnDemandExternalIterations()
    << std::endl;
  if (hasRequestTimestamp()) {
    std::time_t t_c = system_clock::to_time_t(requestTimestamp());
    std::tm tm;
    s << "Trace request client timestamp: "
      << std::put_time(localtime_r(&t_c, &tm), "%F %T") << std::endl;
  }
  s << "Trace duration: " << activitiesOnDemandDuration().count() << "ms"
    << std::endl;
  s << "Warmup duration: " << activitiesWarmupDuration().count() << "s"
    << std::endl;
  s << "Net size threshold: " << activitiesOnDemandExternalNetSizeThreshold()
    << std::endl;
  s << "GPU op count threshold: "
    << activitiesOnDemandExternalGpuOpCountThreshold() << std::endl;
  s << "Max GPU buffer size: " << activitiesMaxGpuBufferSize() / 1024 / 1024
    << "MB" << std::endl;

  s << "Enabled activities: ";
  for (const auto& activity : selectedActivityTypes_) {
    switch(activity){
      case ActivityType::GPU_MEMCPY:
        s << kActivityMemcpy << " ";
        break;
      case ActivityType::GPU_MEMSET:
        s << kActivityMemset << " ";
        break;
      case ActivityType::CONCURRENT_KERNEL:
        s << kActivityConcurrentKernel << " ";
        break;
      case ActivityType::EXTERNAL_CORRELATION:
        s << kActivityExternalCorrelation << " ";
        break;
      case ActivityType::CUDA_RUNTIME:
        s << kActivityRuntime << " ";
        break;
      default:
        s << "UNKNOWN_ACTIVITY_NAME" << " ";
        break;
    }
  }
  s << std::endl;

  AbstractConfig::printActivityProfilerConfig(s);
}

} // namespace KINETO_NAMESPACE
