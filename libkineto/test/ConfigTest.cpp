/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "include/Config.h"

#include <fmt/format.h>
#include <gtest/gtest.h>
#include <time.h>
#include <chrono>

using namespace std::chrono;
using namespace KINETO_NAMESPACE;

TEST(ParseTest, Whitespace) {
  Config cfg;
  // Check that various types of whitespace is ignored
  EXPECT_TRUE(cfg.parse(""));
  EXPECT_TRUE(cfg.parse(" "));
  EXPECT_TRUE(cfg.parse("\t"));
  EXPECT_TRUE(cfg.parse("\n"));
  EXPECT_TRUE(cfg.parse("    "));
  EXPECT_TRUE(cfg.parse("\t \n  \t\t\n\n"));
  // Only the above characters are supported
  EXPECT_FALSE(cfg.parse("\r\n"));
}

TEST(ParseTest, Comment) {
  Config cfg;
  // Anything following a '#' should be ignored, up to a newline
  EXPECT_TRUE(cfg.parse("# comment"));
  EXPECT_TRUE(cfg.parse("   # ~!@#$"));
  EXPECT_TRUE(cfg.parse("\t#abc"));
  EXPECT_TRUE(cfg.parse("###\n##"));
  EXPECT_TRUE(cfg.parse("EVENTS=util ##ok"));
  EXPECT_TRUE(cfg.parse("EVENTS=util ## EVENTS=instruction"));
  // Whatever appears before the comment must be valid format
  EXPECT_FALSE(cfg.parse("util ## not ok"));
  EXPECT_FALSE(cfg.parse("## ok \n blah # not OK"));
  // Check that a comment does not affect config parsing
  EXPECT_TRUE(cfg.parse("SAMPLE_PERIOD_MSECS = 1 # Sample every millisecond"));
  EXPECT_EQ(cfg.samplePeriod(), milliseconds(1));
}

TEST(ParseTest, Format) {
  Config cfg;
  // The basic format is just "name = value".
  // Where both value and name can be almost anything.
  // Leading and trailing whitespace should be removed
  // for both 'name' and 'value', but internal whitespace is not.
  EXPECT_FALSE(cfg.parse("events"));
  EXPECT_TRUE(cfg.parse("events="));
  EXPECT_FALSE(cfg.parse("=events="));
  EXPECT_TRUE(cfg.parse("events=1,2,3"));
  // Only one setting per line
  EXPECT_FALSE(cfg.parse("events = 1,2,3 ; metrics = 4,5,6"));
  // Names are case sensitive
  EXPECT_TRUE(cfg.parse("EVENTS = 1,2,3 \n metrics = 4,5,6"));
  EXPECT_EQ(cfg.eventNames(), std::set<std::string>({"1", "2", "3"}));
  EXPECT_EQ(cfg.metricNames().size(), 0);
  // Leading and trailing whitespace removed for event and metric names,
  // but not internal.
  EXPECT_TRUE(
      cfg.parse("EVENTS = 1, 2, 3 \n \tMETRICS\t = \t4,\t5\t,\ts i x  "));
  EXPECT_EQ(cfg.eventNames(), std::set<std::string>({"1", "2", "3"}));
  EXPECT_EQ(cfg.metricNames(), std::set<std::string>({"4", "5", "s i x"}));
}

TEST(ParseTest, DefaultActivityTypes) {
  Config cfg;
  cfg.validate(std::chrono::system_clock::now());
  auto default_activities = defaultActivityTypes();
  EXPECT_EQ(cfg.selectedActivityTypes(),
    std::set<ActivityType>(default_activities.begin(), default_activities.end()));
}

TEST(ParseTest, ActivityTypes) {
  Config cfg;
  EXPECT_FALSE(cfg.parse("ACTIVITY_TYPES"));
  EXPECT_TRUE(cfg.parse("ACTIVITY_TYPES="));
  EXPECT_FALSE(cfg.parse("=ACTIVITY_TYPES="));

  EXPECT_EQ(cfg.selectedActivityTypes(),
    std::set<ActivityType>({ActivityType::CPU_OP,
                            ActivityType::CPU_INSTANT_EVENT,
                            ActivityType::PYTHON_FUNCTION,
                            ActivityType::USER_ANNOTATION,
                            ActivityType::GPU_USER_ANNOTATION,
                            ActivityType::GPU_MEMCPY,
                            ActivityType::GPU_MEMSET,
                            ActivityType::CONCURRENT_KERNEL,
                            ActivityType::EXTERNAL_CORRELATION,
                            ActivityType::OVERHEAD,
                            ActivityType::CUDA_RUNTIME,
                            ActivityType::CUDA_DRIVER,
                            ActivityType::CUDA_SYNC,
                            ActivityType::MTIA_RUNTIME,
                            ActivityType::MTIA_CCP_EVENTS}));

  Config cfg2;
  EXPECT_TRUE(cfg2.parse("ACTIVITY_TYPES=gpu_memcpy,gpu_MeMsEt,kernel"));
  EXPECT_EQ(cfg2.selectedActivityTypes(),
    std::set<ActivityType>({ActivityType::GPU_MEMCPY,
                            ActivityType::GPU_MEMSET,
                            ActivityType::CONCURRENT_KERNEL}));

  EXPECT_TRUE(cfg2.parse("ACTIVITY_TYPES = cuda_Runtime,"));
  EXPECT_EQ(cfg2.selectedActivityTypes(),
    std::set<ActivityType>({ActivityType::CUDA_RUNTIME}));

  // Should throw an exception because incorrect activity name
  EXPECT_FALSE(cfg2.parse("ACTIVITY_TYPES = memcopy,cuda_runtime"));

  EXPECT_TRUE(cfg2.parse("ACTIVITY_TYPES = cpu_op"));
  EXPECT_EQ(cfg2.selectedActivityTypes(),
    std::set<ActivityType>({ActivityType::CPU_OP}));

  EXPECT_TRUE(cfg2.parse("ACTIVITY_TYPES = xpu_Runtime"));
  EXPECT_EQ(cfg2.selectedActivityTypes(),
    std::set<ActivityType>({ActivityType::XPU_RUNTIME}));

  EXPECT_TRUE(cfg2.parse("ACTIVITY_TYPES=privateuse1_Runtime,privateuse1_driver"));
  EXPECT_EQ(cfg2.selectedActivityTypes(),
    std::set<ActivityType>({ActivityType::PRIVATEUSE1_RUNTIME,
                            ActivityType::PRIVATEUSE1_DRIVER}));
}

TEST(ParseTest, SamplePeriod) {
  Config cfg;
  EXPECT_TRUE(cfg.parse("SAMPLE_PERIOD_MSECS=10"));
  EXPECT_EQ(cfg.samplePeriod(), milliseconds(10));
  EXPECT_TRUE(cfg.parse("SAMPLE_PERIOD_MSECS=0"));
  cfg.validate(std::chrono::system_clock::now());
  // 0 should be adjustd up to 1
  EXPECT_EQ(cfg.samplePeriod(), milliseconds(1));
  // Negative and non-int values should fail
  EXPECT_FALSE(cfg.parse("SAMPLE_PERIOD_MSECS=-10"));
  EXPECT_FALSE(cfg.parse("SAMPLE_PERIOD_MSECS=1.5"));
  EXPECT_FALSE(cfg.parse("SAMPLE_PERIOD_MSECS="));
  EXPECT_FALSE(cfg.parse("SAMPLE_PERIOD_MSECS=string"));
  EXPECT_EQ(cfg.samplePeriod(), milliseconds(1));
}

TEST(ParseTest, MultiplexPeriod) {
  Config cfg;
  auto now = std::chrono::system_clock::now();

  EXPECT_TRUE(cfg.parse("SAMPLE_PERIOD_MSECS=100\nMULTIPLEX_PERIOD_MSECS=100"));
  EXPECT_EQ(cfg.multiplexPeriod(), milliseconds(100));
  EXPECT_TRUE(cfg.parse("MULTIPLEX_PERIOD_MSECS = 0"));
  cfg.validate(now);
  // Adjusted to match sample period
  EXPECT_EQ(cfg.multiplexPeriod(), milliseconds(100));
  EXPECT_TRUE(cfg.parse("MULTIPLEX_PERIOD_MSECS \t= \t 750 \n"));
  cfg.validate(now);
  // Adjusted to match multiple of sample period
  EXPECT_EQ(cfg.multiplexPeriod(), milliseconds(800));
  EXPECT_FALSE(cfg.parse("MULTIPLEX_PERIOD_MSECS=-10"));
  EXPECT_FALSE(cfg.parse("MULTIPLEX_PERIOD_MSECS=1.5"));
  EXPECT_FALSE(cfg.parse("MULTIPLEX_PERIOD_MSECS="));
  EXPECT_FALSE(cfg.parse("MULTIPLEX_PERIOD_MSECS=string"));
  // Previous value not affected
  EXPECT_EQ(cfg.multiplexPeriod(), milliseconds(800));
}

TEST(ParseTest, ReportPeriod) {
  Config cfg;
  EXPECT_TRUE(cfg.parse("REPORT_PERIOD_SECS=1"));
  EXPECT_EQ(cfg.reportPeriod(), seconds(1));
  // Whitespace
  EXPECT_TRUE(cfg.parse("REPORT_PERIOD_SECS  =  \t100"));
  EXPECT_EQ(cfg.reportPeriod(), seconds(100));
  // Invalid types
  EXPECT_FALSE(cfg.parse("REPORT_PERIOD_SECS=-1"));
  EXPECT_EQ(cfg.reportPeriod(), seconds(100));
}

TEST(ParseTest, SamplesPerReport) {
  Config cfg;
  auto now = std::chrono::system_clock::now();

  EXPECT_TRUE(cfg.parse(R"(
    SAMPLE_PERIOD_MSECS = 1000
    REPORT_PERIOD_SECS  =    1
    SAMPLES_PER_REPORT  =   10)"));
  cfg.validate(now);
  // Adjusted down to one sample per report
  EXPECT_EQ(cfg.samplesPerReport(), 1);
  EXPECT_TRUE(cfg.parse(R"(
    SAMPLE_PERIOD_MSECS = 1000
    REPORT_PERIOD_SECS  =   10
    SAMPLES_PER_REPORT  =   10)"));
  cfg.validate(now);
  // No adjustment needed
  EXPECT_EQ(cfg.samplesPerReport(), 10);
  EXPECT_TRUE(cfg.parse(R"(
    SAMPLE_PERIOD_MSECS = 1000
    REPORT_PERIOD_SECS  =    2
    SAMPLES_PER_REPORT  =   10)"));
  cfg.validate(now);
  // Adjusted to 2 samples per report
  EXPECT_EQ(cfg.samplesPerReport(), 2);
  EXPECT_TRUE(cfg.parse(R"(
    SAMPLE_PERIOD_MSECS = 200
    REPORT_PERIOD_SECS  =   2
    SAMPLES_PER_REPORT  =  10)"));
  cfg.validate(now);
  // No adjustment needed
  EXPECT_EQ(cfg.samplesPerReport(), 10);
  EXPECT_TRUE(cfg.parse("SAMPLES_PER_REPORT=0"));
  cfg.validate(now);
  // Adjusted up to 1
  EXPECT_EQ(cfg.samplesPerReport(), 1);
  // Invalid value types
  EXPECT_FALSE(cfg.parse("SAMPLES_PER_REPORT=-10"));
  EXPECT_FALSE(cfg.parse("SAMPLES_PER_REPORT=1.5"));
  EXPECT_EQ(cfg.samplesPerReport(), 1);

  EXPECT_TRUE(cfg.parse(R"(
    SAMPLE_PERIOD_MSECS=1000
    MULTIPLEX_PERIOD_MSECS=500 # Must be a multiple of sample period
    REPORT_PERIOD_SECS=0       # Must be non-zero multiple of multiplex period
    SAMPLES_PER_REPORT=5       # Max report period / multiplex period)"));
  cfg.validate(now);
  // Multiple adjustments
  EXPECT_EQ(cfg.samplePeriod(), milliseconds(1000));
  EXPECT_EQ(cfg.multiplexPeriod(), milliseconds(1000));
  EXPECT_EQ(cfg.reportPeriod(), seconds(1));
  EXPECT_EQ(cfg.samplesPerReport(), 1);
}

TEST(ParseTest, EnableSigUsr2) {
  Config cfg;
  EXPECT_TRUE(cfg.parse("ENABLE_SIGUSR2=yes"));
  EXPECT_TRUE(cfg.sigUsr2Enabled());
  EXPECT_TRUE(cfg.parse("ENABLE_SIGUSR2=no"));
  EXPECT_FALSE(cfg.sigUsr2Enabled());
  EXPECT_TRUE(cfg.parse("ENABLE_SIGUSR2=YES"));
  EXPECT_TRUE(cfg.sigUsr2Enabled());
  EXPECT_TRUE(cfg.parse("ENABLE_SIGUSR2=NO"));
  EXPECT_FALSE(cfg.sigUsr2Enabled());
  EXPECT_TRUE(cfg.parse("ENABLE_SIGUSR2=Y"));
  EXPECT_TRUE(cfg.sigUsr2Enabled());
  EXPECT_TRUE(cfg.parse("ENABLE_SIGUSR2=N"));
  EXPECT_FALSE(cfg.sigUsr2Enabled());
  EXPECT_TRUE(cfg.parse("ENABLE_SIGUSR2=T"));
  EXPECT_TRUE(cfg.sigUsr2Enabled());
  EXPECT_TRUE(cfg.parse("ENABLE_SIGUSR2=F"));
  EXPECT_FALSE(cfg.sigUsr2Enabled());
  EXPECT_TRUE(cfg.parse("ENABLE_SIGUSR2=true"));
  EXPECT_TRUE(cfg.sigUsr2Enabled());
  EXPECT_TRUE(cfg.parse("ENABLE_SIGUSR2=false"));
  EXPECT_FALSE(cfg.sigUsr2Enabled());
  EXPECT_FALSE(cfg.parse("ENABLE_SIGUSR2=  "));
  EXPECT_FALSE(cfg.parse("ENABLE_SIGUSR2=2"));
  EXPECT_FALSE(cfg.parse("ENABLE_SIGUSR2=-1"));
  EXPECT_FALSE(cfg.parse("ENABLE_SIGUSR2=yep"));
}

TEST(ParseTest, DeviceMask) {
  Config cfg;
  // Single device
  EXPECT_TRUE(cfg.parse("EVENTS_ENABLED_DEVICES = 0"));
  EXPECT_TRUE(cfg.eventProfilerEnabledForDevice(0));
  EXPECT_FALSE(cfg.eventProfilerEnabledForDevice(1));

  // Two devices, internal whitespace
  EXPECT_TRUE(cfg.parse("EVENTS_ENABLED_DEVICES = 1, 2"));
  EXPECT_FALSE(cfg.eventProfilerEnabledForDevice(0));
  EXPECT_TRUE(cfg.eventProfilerEnabledForDevice(1));
  EXPECT_TRUE(cfg.eventProfilerEnabledForDevice(2));
  EXPECT_FALSE(cfg.eventProfilerEnabledForDevice(3));

  // Three devices, check that previous devices are ignored
  EXPECT_TRUE(cfg.parse("EVENTS_ENABLED_DEVICES = 0, 2,4"));
  EXPECT_TRUE(cfg.eventProfilerEnabledForDevice(0));
  EXPECT_FALSE(cfg.eventProfilerEnabledForDevice(1));
  EXPECT_TRUE(cfg.eventProfilerEnabledForDevice(2));
  EXPECT_FALSE(cfg.eventProfilerEnabledForDevice(3));
  EXPECT_TRUE(cfg.eventProfilerEnabledForDevice(4));
  EXPECT_FALSE(cfg.eventProfilerEnabledForDevice(5));

  // Repeated numbers have no effect
  EXPECT_TRUE(cfg.parse("EVENTS_ENABLED_DEVICES = 0,1,1,1,2,3,2,1,3,7,7,3"));
  EXPECT_TRUE(cfg.eventProfilerEnabledForDevice(0));
  EXPECT_TRUE(cfg.eventProfilerEnabledForDevice(1));
  EXPECT_TRUE(cfg.eventProfilerEnabledForDevice(2));
  EXPECT_TRUE(cfg.eventProfilerEnabledForDevice(3));
  EXPECT_FALSE(cfg.eventProfilerEnabledForDevice(4));
  EXPECT_FALSE(cfg.eventProfilerEnabledForDevice(6));
  EXPECT_TRUE(cfg.eventProfilerEnabledForDevice(7));

  // 8 is larger than the max allowed
  EXPECT_FALSE(cfg.parse("EVENTS_ENABLED_DEVICES = 3,8"));

  // 300 cannot be held in an uint8_t
  EXPECT_FALSE(cfg.parse("EVENTS_ENABLED_DEVICES = 300"));

  // Various illegal cases
  EXPECT_FALSE(cfg.parse("EVENTS_ENABLED_DEVICES = 0,1,two,three"));
  EXPECT_FALSE(cfg.parse("EVENTS_ENABLED_DEVICES = 0,1,,2"));
  EXPECT_FALSE(cfg.parse("EVENTS_ENABLED_DEVICES = -1"));
  EXPECT_FALSE(cfg.parse("EVENTS_ENABLED_DEVICES = 1.0"));
}

TEST(ParseTest, RequestTime) {
  Config cfg;
  system_clock::time_point now = system_clock::now();
  int64_t tgood_ms =
      duration_cast<milliseconds>(now.time_since_epoch()).count();
  EXPECT_TRUE(cfg.parse(fmt::format("REQUEST_TIMESTAMP = {}", tgood_ms)));

  tgood_ms = duration_cast<milliseconds>((now - seconds(5)).time_since_epoch())
                 .count();
  EXPECT_TRUE(cfg.parse(fmt::format("REQUEST_TIMESTAMP = {}", tgood_ms)));

  int64_t tbad_ms =
      duration_cast<milliseconds>((now - seconds(20)).time_since_epoch())
          .count();
  EXPECT_FALSE(cfg.parse(fmt::format("REQUEST_TIMESTAMP = {}", tbad_ms)));

  EXPECT_FALSE(cfg.parse("REQUEST_TIMESTAMP = 0"));
  EXPECT_FALSE(cfg.parse("REQUEST_TIMESTAMP = -1"));

  tbad_ms = duration_cast<milliseconds>((now + seconds(10)).time_since_epoch())
                .count();
  EXPECT_FALSE(cfg.parse(fmt::format("REQUEST_TIMESTAMP = {}", tbad_ms)));
}

TEST(ParseTest, ProfileStartTime) {
  Config cfg;
  system_clock::time_point now = system_clock::now();
  int64_t tgood_ms =
      duration_cast<milliseconds>(now.time_since_epoch()).count();
  EXPECT_TRUE(cfg.parse(fmt::format("PROFILE_START_TIME = {}", tgood_ms)));

  // Pass given PROFILE_START_TIME = 0, a timestamp is assigned.
  tgood_ms = 0;
  EXPECT_TRUE(cfg.parse(fmt::format("PROFILE_START_TIME = {}", tgood_ms)));

  // Fail given PROFILE_START_TIME older than kMaxRequestAge from now.
  int64_t tbad_ms =
      duration_cast<milliseconds>((now - seconds(15)).time_since_epoch())
          .count();
  EXPECT_FALSE(cfg.parse(fmt::format("PROFILE_START_TIME = {}", tbad_ms)));
}
