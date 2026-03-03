/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <vector>

#include "ConfigLoader.h"
#include "DaemonConfigLoader.h"

using namespace KINETO_NAMESPACE;

// ============================================================================
// Mock IDaemonConfigLoader for testing
// ============================================================================
// This mock allows us to control what config values are returned, enabling
// isolated testing of ConfigLoader's multi-loader iteration logic.

class MockConfigLoader : public IDaemonConfigLoader {
 public:
  explicit MockConfigLoader(
      std::string baseConfig = "",
      std::string onDemandConfig = "",
      int gpuCount = 0)
      : baseConfig_(std::move(baseConfig)),
        onDemandConfig_(std::move(onDemandConfig)),
        gpuCount_(gpuCount) {}

  std::string readBaseConfig() override {
    readBaseConfigCalled_ = true;
    return baseConfig_;
  }

  std::string readOnDemandConfig(bool /*events*/, bool activities) override {
    readOnDemandConfigCalled_ = true;
    if (!activities) {
      return "";
    }
    return onDemandConfig_;
  }

  int gpuContextCount(uint32_t /*device*/) override {
    gpuContextCountCalled_ = true;
    return gpuCount_;
  }

  void setCommunicationFabric(bool enabled) override {
    setCommunicationFabricCalled_ = true;
    communicationFabricEnabled_ = enabled;
  }

  // Test helpers
  bool wasReadBaseConfigCalled() const {
    return readBaseConfigCalled_;
  }
  bool wasReadOnDemandConfigCalled() const {
    return readOnDemandConfigCalled_;
  }
  bool wasGpuContextCountCalled() const {
    return gpuContextCountCalled_;
  }
  bool wasCommunicationFabricCalled() const {
    return setCommunicationFabricCalled_;
  }
  bool isCommunicationFabricEnabled() const {
    return communicationFabricEnabled_;
  }

 private:
  std::string baseConfig_;
  std::string onDemandConfig_;
  int gpuCount_;
  bool readBaseConfigCalled_ = false;
  bool readOnDemandConfigCalled_ = false;
  bool gpuContextCountCalled_ = false;
  bool setCommunicationFabricCalled_ = false;
  bool communicationFabricEnabled_ = false;
};

// ============================================================================
// Test Fixture
// ============================================================================
// This fixture ensures proper cleanup between tests by resetting the static
// factory vector and the singleton's loader vector. Without this, factories
// registered in one test would leak into subsequent tests.

class ConfigLoaderTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Clean slate for each test - clear any previously registered factories
    // and loaders from other tests
    ConfigLoader::clearConfigLoaderFactories();
    ConfigLoader::instance().clearConfigLoaders();
  }

  void TearDown() override {
    // Clean up after each test to prevent pollution of subsequent tests
    ConfigLoader::clearConfigLoaderFactories();
    ConfigLoader::instance().clearConfigLoaders();
  }
};

// ============================================================================
// addConfigLoaderFactory Tests
// ============================================================================

TEST_F(ConfigLoaderTest, SingleFactoryRegistration) {
  bool factoryCalled = false;

  ConfigLoader::addConfigLoaderFactory(
      [&factoryCalled]() -> std::unique_ptr<IDaemonConfigLoader> {
        factoryCalled = true;
        return std::make_unique<MockConfigLoader>();
      });

  // Factory should not be called until initConfigLoaders
  EXPECT_FALSE(factoryCalled);
}

TEST_F(ConfigLoaderTest, MultipleFactoriesCanBeRegistered) {
  int factoryCallCount = 0;

  // Register first factory
  ConfigLoader::addConfigLoaderFactory(
      [&factoryCallCount]() -> std::unique_ptr<IDaemonConfigLoader> {
        factoryCallCount++;
        return std::make_unique<MockConfigLoader>("base1", "ondemand1");
      });

  // Register second factory
  ConfigLoader::addConfigLoaderFactory(
      [&factoryCallCount]() -> std::unique_ptr<IDaemonConfigLoader> {
        factoryCallCount++;
        return std::make_unique<MockConfigLoader>("base2", "ondemand2");
      });

  // Factories should not be called yet
  EXPECT_EQ(factoryCallCount, 0);
}

// ============================================================================
// initConfigLoaders Tests
// ============================================================================
// Note: initConfigLoaders is private, but we can indirectly test it through
// the public contextCountForGpu which triggers initialization.

TEST_F(ConfigLoaderTest, InitCreatesLoadersFromAllFactories) {
  std::vector<MockConfigLoader*> createdLoaders;

  // Register two factories that track their created loaders
  ConfigLoader::addConfigLoaderFactory(
      [&createdLoaders]() -> std::unique_ptr<IDaemonConfigLoader> {
        auto loader = std::make_unique<MockConfigLoader>("", "", 1);
        createdLoaders.push_back(loader.get());
        return loader;
      });

  ConfigLoader::addConfigLoaderFactory(
      [&createdLoaders]() -> std::unique_ptr<IDaemonConfigLoader> {
        auto loader = std::make_unique<MockConfigLoader>("", "", 2);
        createdLoaders.push_back(loader.get());
        return loader;
      });

  // Trigger initialization by calling contextCountForGpu
  // (this internally calls initConfigLoaders)
  int count = ConfigLoader::instance().contextCountForGpu(0);

  // Both factories should have been called
  EXPECT_EQ(createdLoaders.size(), 2);
  // First loader returns 1, so iteration should stop there
  EXPECT_EQ(count, 1);
}

TEST_F(ConfigLoaderTest, InitOnlyRunsOnce) {
  int factoryCallCount = 0;

  ConfigLoader::addConfigLoaderFactory(
      [&factoryCallCount]() -> std::unique_ptr<IDaemonConfigLoader> {
        factoryCallCount++;
        return std::make_unique<MockConfigLoader>("", "", 5);
      });

  // Call contextCountForGpu multiple times
  ConfigLoader::instance().contextCountForGpu(0);
  ConfigLoader::instance().contextCountForGpu(0);
  ConfigLoader::instance().contextCountForGpu(0);

  // Factory should only be called once (during first init)
  EXPECT_EQ(factoryCallCount, 1);
}

// ============================================================================
// Multi-Loader Iteration Tests
// ============================================================================
// These tests verify the core behavior: when multiple loaders are registered,
// ConfigLoader iterates through them and returns the first non-empty result.

TEST_F(ConfigLoaderTest, FirstLoaderWithResultIsUsed) {
  // Loader 1: returns empty (simulates no config available)
  ConfigLoader::addConfigLoaderFactory(
      []() -> std::unique_ptr<IDaemonConfigLoader> {
        return std::make_unique<MockConfigLoader>("", "", 0);
      });

  // Loader 2: returns a value
  ConfigLoader::addConfigLoaderFactory(
      []() -> std::unique_ptr<IDaemonConfigLoader> {
        return std::make_unique<MockConfigLoader>("", "", 42);
      });

  // Loader 3: also returns a value (should not be used)
  ConfigLoader::addConfigLoaderFactory(
      []() -> std::unique_ptr<IDaemonConfigLoader> {
        return std::make_unique<MockConfigLoader>("", "", 100);
      });

  int count = ConfigLoader::instance().contextCountForGpu(0);

  // Should get result from second loader (first one with value > 0)
  EXPECT_EQ(count, 42);
}

TEST_F(ConfigLoaderTest, AllLoadersEmptyReturnsZero) {
  // All loaders return 0 (no GPU context)
  ConfigLoader::addConfigLoaderFactory(
      []() -> std::unique_ptr<IDaemonConfigLoader> {
        return std::make_unique<MockConfigLoader>("", "", 0);
      });

  ConfigLoader::addConfigLoaderFactory(
      []() -> std::unique_ptr<IDaemonConfigLoader> {
        return std::make_unique<MockConfigLoader>("", "", 0);
      });

  int count = ConfigLoader::instance().contextCountForGpu(0);
  EXPECT_EQ(count, 0);
}

TEST_F(ConfigLoaderTest, NoLoadersRegisteredReturnsZero) {
  // No factories registered
  int count = ConfigLoader::instance().contextCountForGpu(0);
  EXPECT_EQ(count, 0);
}

// ============================================================================
// Factory Returning Null Tests
// ============================================================================
// Edge case: what if a factory returns nullptr?

TEST_F(ConfigLoaderTest, NullFactoryResultIsSkipped) {
  // Factory 1: returns nullptr
  ConfigLoader::addConfigLoaderFactory(
      []() -> std::unique_ptr<IDaemonConfigLoader> { return nullptr; });

  // Factory 2: returns valid loader
  ConfigLoader::addConfigLoaderFactory(
      []() -> std::unique_ptr<IDaemonConfigLoader> {
        return std::make_unique<MockConfigLoader>("", "", 7);
      });

  int count = ConfigLoader::instance().contextCountForGpu(0);

  // Should skip null and use second loader
  EXPECT_EQ(count, 7);
}

// ============================================================================
// clearConfigLoaderFactories Tests
// ============================================================================

TEST_F(ConfigLoaderTest, ClearFactoriesRemovesAllFactories) {
  int factoryCallCount = 0;

  ConfigLoader::addConfigLoaderFactory(
      [&factoryCallCount]() -> std::unique_ptr<IDaemonConfigLoader> {
        factoryCallCount++;
        return std::make_unique<MockConfigLoader>("", "", 1);
      });

  // Clear factories before initialization
  ConfigLoader::clearConfigLoaderFactories();

  // Trigger initialization
  int count = ConfigLoader::instance().contextCountForGpu(0);

  // No factories should have been called (they were cleared)
  EXPECT_EQ(factoryCallCount, 0);
  EXPECT_EQ(count, 0);
}

// ============================================================================
// clearConfigLoaders Tests
// ============================================================================

TEST_F(ConfigLoaderTest, ClearLoadersAllowsReinitialization) {
  int factoryCallCount = 0;

  ConfigLoader::addConfigLoaderFactory(
      [&factoryCallCount]() -> std::unique_ptr<IDaemonConfigLoader> {
        factoryCallCount++;
        return std::make_unique<MockConfigLoader>("", "", factoryCallCount);
      });

  // First initialization
  int count1 = ConfigLoader::instance().contextCountForGpu(0);
  EXPECT_EQ(count1, 1);
  EXPECT_EQ(factoryCallCount, 1);

  // Clear loaders (but not factories)
  ConfigLoader::instance().clearConfigLoaders();

  // Second initialization - factory should be called again
  int count2 = ConfigLoader::instance().contextCountForGpu(0);
  EXPECT_EQ(count2, 2); // factoryCallCount is now 2
  EXPECT_EQ(factoryCallCount, 2);
}
