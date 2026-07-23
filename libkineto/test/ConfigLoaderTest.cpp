/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <utility>
#include <vector>

#include "include/Config.h"
#include "src/ConfigLoader.h"

using namespace KINETO_NAMESPACE;

namespace {

// Records how ConfigLoader dispatches to a handler and lets a test control
// whether canAcceptConfig() accepts, so the fan-out logic can be asserted
// without a real profiler or the daemon poll thread.
struct RecordingConfigHandler : ConfigLoader::ConfigHandler {
  bool canAcceptResult{true};

  int acceptCalls{0};
  const Config* lastAcceptedConfig{nullptr};

  bool canAcceptConfig() override {
    return canAcceptResult;
  }

  bool acceptConfig(const Config& cfg) override {
    ++acceptCalls;
    lastAcceptedConfig = &cfg;
    return true;
  }
};

// Drives the ConfigLoader singleton's handler-dispatch API directly. The daemon
// config loader factory is left unset, so the polling thread never reads an
// on-demand config and never calls the registered handlers -- every callback a
// test observes comes from its own notifyHandlers()/canHandlerAcceptConfig()
// call. Handlers are unregistered in TearDown because the singleton persists
// across tests in the binary.
class ConfigLoaderTest : public ::testing::Test {
 protected:
  static ConfigLoader& loader() {
    return ConfigLoader::instance();
  }

  void registerHandler(
      ConfigLoader::ConfigKind kind,
      ConfigLoader::ConfigHandler* handler) {
    loader().addHandler(kind, handler);
    registered_.emplace_back(kind, handler);
  }

  void TearDown() override {
    // Join the background config-update thread before this test process exits.
    // addHandler() starts it; leaving the join to static destruction races the
    // thread against teardown and can abort with a mutex lock on a destroyed
    // mutex.
    loader().stopUpdateThread();
    // removeHandler is a no-op for a handler already removed by the test, so
    // double removal is safe.
    for (const auto& [kind, handler] : registered_) {
      loader().removeHandler(kind, handler);
    }
    registered_.clear();
  }

 private:
  std::vector<std::pair<ConfigLoader::ConfigKind, ConfigLoader::ConfigHandler*>>
      registered_;
};

// notifyHandlers() forwards the config to every registered handler across all
// config kinds, passing through the same config object.
TEST_F(ConfigLoaderTest, NotifyHandlersForwardsConfigToAllRegisteredHandlers) {
  RecordingConfigHandler activityA;
  RecordingConfigHandler activityB;
  RecordingConfigHandler event;
  registerHandler(ConfigLoader::ConfigKind::ActivityProfiler, &activityA);
  registerHandler(ConfigLoader::ConfigKind::ActivityProfiler, &activityB);
  registerHandler(ConfigLoader::ConfigKind::EventProfiler, &event);

  Config cfg;
  loader().notifyHandlers(cfg);

  EXPECT_EQ(activityA.acceptCalls, 1);
  EXPECT_EQ(activityB.acceptCalls, 1);
  EXPECT_EQ(event.acceptCalls, 1);
  EXPECT_EQ(activityA.lastAcceptedConfig, &cfg);
  EXPECT_EQ(activityB.lastAcceptedConfig, &cfg);
  EXPECT_EQ(event.lastAcceptedConfig, &cfg);
}

// A removed handler no longer receives configs.
TEST_F(ConfigLoaderTest, RemoveHandlerStopsDispatch) {
  RecordingConfigHandler handler;
  registerHandler(ConfigLoader::ConfigKind::ActivityProfiler, &handler);

  Config first;
  loader().notifyHandlers(first);
  EXPECT_EQ(handler.acceptCalls, 1);

  loader().removeHandler(ConfigLoader::ConfigKind::ActivityProfiler, &handler);

  Config second;
  loader().notifyHandlers(second);
  EXPECT_EQ(handler.acceptCalls, 1); // unchanged: no longer registered
}

// canHandlerAcceptConfig() is true only when every handler of that kind
// accepts.
TEST_F(ConfigLoaderTest, CanHandlerAcceptConfigRequiresAllHandlersOfKind) {
  RecordingConfigHandler first;
  RecordingConfigHandler second;
  registerHandler(ConfigLoader::ConfigKind::ActivityProfiler, &first);
  registerHandler(ConfigLoader::ConfigKind::ActivityProfiler, &second);

  EXPECT_TRUE(
      loader().canHandlerAcceptConfig(
          ConfigLoader::ConfigKind::ActivityProfiler));

  second.canAcceptResult = false;
  EXPECT_FALSE(
      loader().canHandlerAcceptConfig(
          ConfigLoader::ConfigKind::ActivityProfiler));
}

// canHandlerAcceptConfig() gates per kind: a declining handler of one kind does
// not affect another kind, and a kind with no handlers accepts vacuously.
TEST_F(ConfigLoaderTest, CanHandlerAcceptConfigIsPerKind) {
  RecordingConfigHandler event;
  event.canAcceptResult = false;
  registerHandler(ConfigLoader::ConfigKind::EventProfiler, &event);

  EXPECT_FALSE(
      loader().canHandlerAcceptConfig(ConfigLoader::ConfigKind::EventProfiler));
  // ActivityProfiler has no handlers registered, so it accepts vacuously and is
  // unaffected by the declining EventProfiler handler.
  EXPECT_TRUE(
      loader().canHandlerAcceptConfig(
          ConfigLoader::ConfigKind::ActivityProfiler));
}

} // namespace
