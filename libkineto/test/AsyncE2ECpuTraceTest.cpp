/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>

#include <unistd.h>

#include <fmt/format.h>
#include <nlohmann/json.hpp>

#include "include/Config.h"
#include "src/ActivityProfilerController.h"
#include "src/ConfigLoader.h"
#include "src/DaemonConfigLoader.h"
#include "test/TestUtils.h"

using namespace KINETO_NAMESPACE;
using namespace std::chrono;
using libkineto::test::createTempTraceFile;
using libkineto::test::logUrlToPath;
using libkineto::test::TempTraceFile;

namespace {

// Stands in for the dynolog daemon: hands the real ConfigLoader poll thread one
// canned on-demand config over the same IDaemonConfigLoader seam dynolog uses.
// An empty base config lets the base config come from the host, matching a real
// on-demand deployment.
class FakeDaemonConfigLoader : public IDaemonConfigLoader {
 public:
  explicit FakeDaemonConfigLoader(std::string onDemandConfig)
      : onDemandConfig_(std::move(onDemandConfig)) {}

  std::string readBaseConfig() override {
    return "";
  }

  // Hand back the trace request only while the handler can accept one, so it is
  // delivered once and not re-scheduled while the trace is running.
  std::string readOnDemandConfig(bool activities) override {
    return activities ? onDemandConfig_ : "";
  }

  void setCommunicationFabric(bool /*enabled*/) override {}

 private:
  std::string onDemandConfig_;
};

class AsyncE2ECpuTraceTest : public ::testing::Test {
 protected:
  static ConfigLoader& loader() {
    return ConfigLoader::instance();
  }

  void SetUp() override {
    // Point the config loader at a deterministic, empty base config instead of
    // the host's /etc/libkineto.conf: the trace should depend only on the
    // injected on-demand config, not on host state. An empty base config also
    // keeps the test hermetic across repeated runs in one process -- the poll
    // thread then rebuilds the daemon config loader on every poll, whereas a
    // cached non-empty base config would suppress that rebuild after the first
    // trace and starve later runs of the on-demand config. KINETO_CONFIG is
    // read once and cached, so this must be set before the first poll.
    baseConfigPath_ =
        fmt::format("/tmp/kineto_async_e2e_empty_base_{}.conf", getpid());
    std::ofstream(baseConfigPath_).close();
    setenv("KINETO_CONFIG", baseConfigPath_.c_str(), /*overwrite=*/1);
  }

  void TearDown() override {
    // Join the poll thread before destroying the controller it dispatches to,
    // so nothing calls into a freed handler. Idempotent with the test body.
    loader().stopUpdateThread();
    controller_.reset();
    // Drop the injected fake and factory; both capture test-local state.
    loader().resetDaemonConfigLoaderForTesting();
    ConfigLoader::setDaemonConfigLoaderFactory(nullptr);
    std::remove(baseConfigPath_.c_str());
  }

  std::string baseConfigPath_;
  std::unique_ptr<ActivityProfilerController> controller_;
};

// Drives a fake dyno on-demand config through the entire asynchronous chain and
// asserts the resulting trace file. The real ConfigLoader poll thread reads and
// parses the daemon config, the controller accepts it, and
// AsyncActivityProfilerHandler runs warmup -> collect -> process on its
// background loop and writes a Chrome-trace JSON file. Everything is CPU-only,
// so it runs in OSS CI.
TEST_F(AsyncE2ECpuTraceTest, DaemonConfigDrivesTraceFileThroughFullChain) {
  // Reserve a unique name under /tmp. On-demand configs are sandboxed to /tmp,
  // and the log URL gets the pid inserted before .json, so the produced file is
  // not this exact name -- resolve the real path by parsing the same config
  // below.
  const TempTraceFile traceFile =
      createTempTraceFile("kineto_async_e2e_", ".json");
  const std::string traceId = "async-e2e-cpu-trace";

  // A start a few seconds out keeps the request inside the max-age window while
  // leaving the background loop ample time to warm up before it; short warmup
  // and duration keep the test to a few seconds of wall-clock.
  const int64_t startMs =
      duration_cast<milliseconds>(
          (system_clock::now() + seconds(5)).time_since_epoch())
          .count();
  const std::string onDemandConfig = fmt::format(
      "REQUEST_TRACE_ID={}\n"
      "PROFILE_START_TIME={}\n"
      "ACTIVITIES_WARMUP_PERIOD_SECS=1\n"
      "ACTIVITIES_DURATION_SECS=1\n"
      "ACTIVITIES_LOG_FILE={}\n",
      traceId,
      startMs,
      traceFile.path());

  // Resolve the actual output path exactly as the poll thread will: parse the
  // identical string as an on-demand config so the /tmp sandbox and pid rewrite
  // apply, then read the log URL back.
  Config resolved;
  resolved.setOnDemand(true);
  ASSERT_TRUE(resolved.parse(onDemandConfig));
  ASSERT_TRUE(resolved.activityProfilerEnabled());
  const std::string tracePath = logUrlToPath(resolved.activitiesLogUrl());
  ASSERT_FALSE(tracePath.empty());
  // The produced file uses the pid-rewritten name, which TempTraceFile does not
  // own, so remove it ourselves when the test ends.
  struct FileRemover {
    std::string path;
    ~FileRemover() {
      if (!path.empty()) {
        std::remove(path.c_str());
      }
    }
  } fileRemover{tracePath};

  // Install the fake daemon BEFORE constructing the controller: the controller
  // registers as a ConfigLoader handler in its constructor, which starts the
  // poll thread that reads from the fake on its first iteration.
  ConfigLoader::setDaemonConfigLoaderFactory([onDemandConfig]() {
    return std::make_unique<FakeDaemonConfigLoader>(onDemandConfig);
  });

  controller_ =
      std::make_unique<ActivityProfilerController>(loader(), /*cpuOnly=*/true);

  auto& asyncHandler = controller_->asyncHandlerForTesting();
  const uint64_t base = asyncHandler.completedTraceCountForTesting();

  // Wait for the background loop to finalize exactly one trace. Deterministic
  // (no sleeps) via the handler's completion condvar; the timeout is well above
  // start + warmup + duration + loop ticks so a slow CI host does not time out.
  ASSERT_TRUE(
      asyncHandler.waitForCompletedTraceCountForTesting(base + 1, seconds(30)));

  // Stop the poll thread now so it cannot re-deliver the request after the
  // controller returns to idle, which would race the assertions below.
  loader().stopUpdateThread();

  // The file is on disk once the completion count advanced. Assert it is a
  // valid Chrome trace produced for THIS request: the daemon config's
  // REQUEST_TRACE_ID surfaces as the top-level trace_id.
  std::ifstream file(tracePath);
  ASSERT_TRUE(file.good()) << "trace file not found: " << tracePath;
  const std::string jsonStr(
      (std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
  ASSERT_FALSE(jsonStr.empty());

  const nlohmann::json data = nlohmann::json::parse(jsonStr);
  ASSERT_TRUE(data.contains("traceEvents"));
  EXPECT_TRUE(data["traceEvents"].is_array());
  ASSERT_TRUE(data.contains("trace_id"));
  EXPECT_EQ(data["trace_id"].get<std::string>(), traceId);
}

} // namespace
