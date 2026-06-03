/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>
#include <atomic>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

// TODO(T90238193)
// @lint-ignore-every CLANGTIDY facebook-hte-RelativeInclude
#include "include/libkineto.h"
#include "include/output_base.h"
#include "src/ActivityLoggerFactory.h"
#include "src/ActivityProfilerController.h"

using namespace KINETO_NAMESPACE;

class MockActivityLogger : public libkineto::ActivityLogger {
 public:
  explicit MockActivityLogger(const std::string& url)
      : url_(url), constructed_(true) {}

  void handleDeviceInfo(const libkineto::DeviceInfo&, int64_t) override {}
  void handleResourceInfo(const libkineto::ResourceInfo&, int64_t) override {}
  void handleOverheadInfo(
      const libkineto::ActivityLogger::OverheadInfo&,
      int64_t) override {}
  void handleTraceSpan(const libkineto::TraceSpan&) override {}
  void handleActivity(const libkineto::ITraceActivity&) override {}
  void handleGenericActivity(const libkineto::GenericTraceActivity&) override {}
  void handleTraceStart(
      const std::unordered_map<std::string, std::string>&,
      const std::string&) override {}
  void finalizeTrace(
      const libkineto::Config&,
      std::unique_ptr<libkineto::ActivityBuffers>,
      int64_t,
      std::unordered_map<std::string, std::vector<std::string>>&) override {}
  void finalizeMemoryTrace(const std::string&, const libkineto::Config&)
      override {}

  const std::string& getUrl() const {
    return url_;
  }
  bool isConstructed() const {
    return constructed_;
  }

 private:
  std::string url_;
  bool constructed_;
};

class ThrowingConstructorLogger : public libkineto::ActivityLogger {
 public:
  explicit ThrowingConstructorLogger(const std::string&) {
    throw std::runtime_error("Constructor failure");
  }

  void handleDeviceInfo(const libkineto::DeviceInfo&, int64_t) override {}
  void handleResourceInfo(const libkineto::ResourceInfo&, int64_t) override {}
  void handleOverheadInfo(
      const libkineto::ActivityLogger::OverheadInfo&,
      int64_t) override {}
  void handleTraceSpan(const libkineto::TraceSpan&) override {}
  void handleActivity(const libkineto::ITraceActivity&) override {}
  void handleGenericActivity(const libkineto::GenericTraceActivity&) override {}
  void handleTraceStart(
      const std::unordered_map<std::string, std::string>&,
      const std::string&) override {}
  void finalizeTrace(
      const libkineto::Config&,
      std::unique_ptr<libkineto::ActivityBuffers>,
      int64_t,
      std::unordered_map<std::string, std::vector<std::string>>&) override {}
  void finalizeMemoryTrace(const std::string&, const libkineto::Config&)
      override {}
};

class CountingLogger : public libkineto::ActivityLogger {
 public:
  explicit CountingLogger(const std::string& url, int& counter)
      : url_(url), counter_(counter) {
    counter_++;
  }

  void handleDeviceInfo(const libkineto::DeviceInfo&, int64_t) override {}
  void handleResourceInfo(const libkineto::ResourceInfo&, int64_t) override {}
  void handleOverheadInfo(
      const libkineto::ActivityLogger::OverheadInfo&,
      int64_t) override {}
  void handleTraceSpan(const libkineto::TraceSpan&) override {}
  void handleActivity(const libkineto::ITraceActivity&) override {}
  void handleGenericActivity(const libkineto::GenericTraceActivity&) override {}
  void handleTraceStart(
      const std::unordered_map<std::string, std::string>&,
      const std::string&) override {}
  void finalizeTrace(
      const libkineto::Config&,
      std::unique_ptr<libkineto::ActivityBuffers>,
      int64_t,
      std::unordered_map<std::string, std::vector<std::string>>&) override {}
  void finalizeMemoryTrace(const std::string&, const libkineto::Config&)
      override {}

 private:
  std::string url_;
  int& counter_;
};

// Basic public API functionality
TEST(RegisterLoggerFactoryTest, BasicPublicAPI) {
  libkineto::registerLoggerFactory("testproto", [](const std::string& path) {
    return std::make_unique<MockActivityLogger>(path);
  });

  auto& factory = KINETO_NAMESPACE::ActivityProfilerController::loggerFactory();
  auto logger = factory.makeLogger("testproto:///tmp/trace.log");

  ASSERT_NE(logger, nullptr);
  auto* mockLogger = dynamic_cast<MockActivityLogger*>(logger.get());
  ASSERT_NE(mockLogger, nullptr);
  EXPECT_EQ(mockLogger->getUrl(), "/tmp/trace.log");
  EXPECT_TRUE(mockLogger->isConstructed());
}

// Protocol case insensitivity
TEST(RegisterLoggerFactoryTest, ProtocolCaseInsensitive) {
  libkineto::registerLoggerFactory("MixedCase", [](const std::string& path) {
    return std::make_unique<MockActivityLogger>(path);
  });

  auto& factory = KINETO_NAMESPACE::ActivityProfilerController::loggerFactory();

  auto logger1 = factory.makeLogger("mixedcase:///path1");
  ASSERT_NE(logger1, nullptr);
  auto* mock1 = dynamic_cast<MockActivityLogger*>(logger1.get());
  EXPECT_EQ(mock1->getUrl(), "/path1");

  auto logger2 = factory.makeLogger("MIXEDCASE:///path2");
  ASSERT_NE(logger2, nullptr);
  auto* mock2 = dynamic_cast<MockActivityLogger*>(logger2.get());
  EXPECT_EQ(mock2->getUrl(), "/path2");

  auto logger3 = factory.makeLogger("MixedCase:///path3");
  ASSERT_NE(logger3, nullptr);
  auto* mock3 = dynamic_cast<MockActivityLogger*>(logger3.get());
  EXPECT_EQ(mock3->getUrl(), "/path3");
}

// Multiple independent protocols
TEST(RegisterLoggerFactoryTest, MultipleProtocols) {
  libkineto::registerLoggerFactory("proto1", [](const std::string& path) {
    return std::make_unique<MockActivityLogger>("proto1:" + path);
  });

  libkineto::registerLoggerFactory("proto2", [](const std::string& path) {
    return std::make_unique<MockActivityLogger>("proto2:" + path);
  });

  auto& factory = KINETO_NAMESPACE::ActivityProfilerController::loggerFactory();

  auto logger1 = factory.makeLogger("proto1:///path");
  auto* mock1 = dynamic_cast<MockActivityLogger*>(logger1.get());
  ASSERT_NE(mock1, nullptr);
  EXPECT_EQ(mock1->getUrl(), "proto1:/path");

  auto logger2 = factory.makeLogger("proto2:///path");
  auto* mock2 = dynamic_cast<MockActivityLogger*>(logger2.get());
  ASSERT_NE(mock2, nullptr);
  EXPECT_EQ(mock2->getUrl(), "proto2:/path");
}

// Unregistered protocol throws
TEST(RegisterLoggerFactoryTest, UnregisteredProtocolThrows) {
  auto& factory = KINETO_NAMESPACE::ActivityProfilerController::loggerFactory();

  EXPECT_THROW(
      factory.makeLogger("nonexistent:///path"), std::invalid_argument);

  try {
    factory.makeLogger("unknown:///path");
    FAIL() << "Expected std::invalid_argument";
  } catch (const std::invalid_argument& e) {
    std::string msg(e.what());
    EXPECT_TRUE(
        msg.find("unknown") != std::string::npos ||
        msg.find("protocol") != std::string::npos);
  }
}

// Protocol overwriting behavior and warning
TEST(RegisterLoggerFactoryTest, OverwriteProtocolWarning) {
  int counter1 = 0;
  int counter2 = 0;

  libkineto::registerLoggerFactory(
      "overwrite", [&counter1](const std::string& path) {
        return std::make_unique<CountingLogger>(path, counter1);
      });

  libkineto::registerLoggerFactory(
      "overwrite", [&counter2](const std::string& path) {
        return std::make_unique<CountingLogger>(path, counter2);
      });

  auto& factory = KINETO_NAMESPACE::ActivityProfilerController::loggerFactory();
  auto logger = factory.makeLogger("overwrite:///path");

  EXPECT_EQ(counter1, 0);
  EXPECT_EQ(counter2, 1);
  ASSERT_NE(logger, nullptr);
}

// URL protocol stripping
TEST(RegisterLoggerFactoryTest, UrlProtocolStripping) {
  std::string receivedPath;

  libkineto::registerLoggerFactory(
      "striptest", [&receivedPath](const std::string& path) {
        receivedPath = path;
        return std::make_unique<MockActivityLogger>(path);
      });

  auto& factory = KINETO_NAMESPACE::ActivityProfilerController::loggerFactory();

  factory.makeLogger("striptest:///absolute/path/file.log");
  EXPECT_EQ(receivedPath, "/absolute/path/file.log");

  factory.makeLogger("striptest://relative/path");
  EXPECT_EQ(receivedPath, "relative/path");

  factory.makeLogger("striptest://hostname:8080/path");
  EXPECT_EQ(receivedPath, "hostname:8080/path");
}

// Empty protocol string
TEST(RegisterLoggerFactoryTest, EmptyProtocol) {
  libkineto::registerLoggerFactory("", [](const std::string& path) {
    return std::make_unique<MockActivityLogger>(path);
  });

  auto& factory = KINETO_NAMESPACE::ActivityProfilerController::loggerFactory();

  auto logger = factory.makeLogger("://empty-protocol-path");
  ASSERT_NE(logger, nullptr);
  auto* mock = dynamic_cast<MockActivityLogger*>(logger.get());
  EXPECT_EQ(mock->getUrl(), "empty-protocol-path");
}

// URL with no protocol separator
TEST(RegisterLoggerFactoryTest, NoProtocolSeparator) {
  auto& factory = KINETO_NAMESPACE::ActivityProfilerController::loggerFactory();

  EXPECT_THROW(factory.makeLogger("justaplainpath"), std::invalid_argument);
}

// Factory that throws exception
TEST(RegisterLoggerFactoryTest, FactoryThrowsException) {
  libkineto::registerLoggerFactory(
      "throwing",
      [](const std::string&) -> std::unique_ptr<libkineto::ActivityLogger> {
        throw std::runtime_error("Factory intentionally failed");
      });

  auto& factory = KINETO_NAMESPACE::ActivityProfilerController::loggerFactory();

  EXPECT_THROW(factory.makeLogger("throwing:///path"), std::runtime_error);
}

// Factory that returns nullptr
TEST(RegisterLoggerFactoryTest, FactoryReturnsNullptr) {
  libkineto::registerLoggerFactory("nullfactory", [](const std::string&) {
    return std::unique_ptr<libkineto::ActivityLogger>(nullptr);
  });

  auto& factory = KINETO_NAMESPACE::ActivityProfilerController::loggerFactory();

  auto logger = factory.makeLogger("nullfactory:///path");
  EXPECT_EQ(logger, nullptr);
}

// Logger constructor throws
TEST(RegisterLoggerFactoryTest, LoggerConstructorThrows) {
  libkineto::registerLoggerFactory("badlogger", [](const std::string& path) {
    return std::make_unique<ThrowingConstructorLogger>(path);
  });

  auto& factory = KINETO_NAMESPACE::ActivityProfilerController::loggerFactory();

  EXPECT_THROW(factory.makeLogger("badlogger:///path"), std::runtime_error);
}

// Special characters in protocol name
TEST(RegisterLoggerFactoryTest, SpecialCharactersInProtocol) {
  libkineto::registerLoggerFactory(
      "my-custom_proto", [](const std::string& path) {
        return std::make_unique<MockActivityLogger>(path);
      });

  auto& factory = KINETO_NAMESPACE::ActivityProfilerController::loggerFactory();
  auto logger = factory.makeLogger("my-custom_proto:///path");
  ASSERT_NE(logger, nullptr);
}

// Special characters in URL path
TEST(RegisterLoggerFactoryTest, SpecialCharactersInPath) {
  std::string receivedPath;

  libkineto::registerLoggerFactory(
      "pathtest", [&receivedPath](const std::string& path) {
        receivedPath = path;
        return std::make_unique<MockActivityLogger>(path);
      });

  auto& factory = KINETO_NAMESPACE::ActivityProfilerController::loggerFactory();

  factory.makeLogger("pathtest:///tmp/trace-2024_06_02-v1.0.log");
  EXPECT_EQ(receivedPath, "/tmp/trace-2024_06_02-v1.0.log");

  factory.makeLogger("pathtest:///path/with spaces/file.log");
  EXPECT_EQ(receivedPath, "/path/with spaces/file.log");

  factory.makeLogger("pathtest:///path/with/query?param=value");
  EXPECT_EQ(receivedPath, "/path/with/query?param=value");
}

// Documented example from libkineto.h
TEST(RegisterLoggerFactoryTest, DocumentedPerfettoExample) {
  libkineto::registerLoggerFactory("perfetto", [](const std::string& path) {
    return std::make_unique<MockActivityLogger>(path);
  });

  auto& factory = KINETO_NAMESPACE::ActivityProfilerController::loggerFactory();
  auto logger = factory.makeLogger("perfetto:///tmp/trace.pftrace");

  ASSERT_NE(logger, nullptr);
  auto* mock = dynamic_cast<MockActivityLogger*>(logger.get());
  ASSERT_NE(mock, nullptr);
  EXPECT_EQ(mock->getUrl(), "/tmp/trace.pftrace");
}

// Concurrent registration
TEST(RegisterLoggerFactoryTest, ConcurrentRegistration) {
  std::vector<std::thread> threads;
  std::atomic<int> successCount{0};

  for (int i = 0; i < 10; i++) {
    threads.emplace_back([i, &successCount]() {
      std::string protocol = "concurrent" + std::to_string(i);
      libkineto::registerLoggerFactory(protocol, [](const std::string& path) {
        return std::make_unique<MockActivityLogger>(path);
      });
      successCount++;
    });
  }

  for (auto& t : threads) {
    t.join();
  }

  EXPECT_EQ(successCount, 10);

  auto& factory = KINETO_NAMESPACE::ActivityProfilerController::loggerFactory();
  for (int i = 0; i < 10; i++) {
    std::string url = "concurrent" + std::to_string(i) + ":///path";
    auto logger = factory.makeLogger(url);
    EXPECT_NE(logger, nullptr);
  }
}

// Very long protocol name
TEST(RegisterLoggerFactoryTest, LongProtocolName) {
  std::string longProtocol(1000, 'a');
  libkineto::registerLoggerFactory(longProtocol, [](const std::string& path) {
    return std::make_unique<MockActivityLogger>(path);
  });

  auto& factory = KINETO_NAMESPACE::ActivityProfilerController::loggerFactory();
  auto logger = factory.makeLogger(longProtocol + ":///path");
  ASSERT_NE(logger, nullptr);
}

// Very long URL path
TEST(RegisterLoggerFactoryTest, LongUrlPath) {
  std::string receivedPath;
  std::string longPath(10000, 'x');

  libkineto::registerLoggerFactory(
      "longpath", [&receivedPath](const std::string& path) {
        receivedPath = path;
        return std::make_unique<MockActivityLogger>(path);
      });

  auto& factory = KINETO_NAMESPACE::ActivityProfilerController::loggerFactory();
  factory.makeLogger("longpath://" + longPath);

  EXPECT_EQ(receivedPath, longPath);
}

// Register logger before and after using other loggers
TEST(RegisterLoggerFactoryTest, RegisterAtDifferentTimes) {
  libkineto::registerLoggerFactory("early", [](const std::string& path) {
    return std::make_unique<MockActivityLogger>("early:" + path);
  });

  auto& factory = KINETO_NAMESPACE::ActivityProfilerController::loggerFactory();

  auto logger1 = factory.makeLogger("early:///path1");
  ASSERT_NE(logger1, nullptr);
  auto* mock1 = dynamic_cast<MockActivityLogger*>(logger1.get());
  EXPECT_EQ(mock1->getUrl(), "early:/path1");

  libkineto::registerLoggerFactory("late", [](const std::string& path) {
    return std::make_unique<MockActivityLogger>("late:" + path);
  });

  auto logger2 = factory.makeLogger("late:///path2");
  ASSERT_NE(logger2, nullptr);
  auto* mock2 = dynamic_cast<MockActivityLogger*>(logger2.get());
  EXPECT_EQ(mock2->getUrl(), "late:/path2");

  auto logger3 = factory.makeLogger("early:///path3");
  ASSERT_NE(logger3, nullptr);
  auto* mock3 = dynamic_cast<MockActivityLogger*>(logger3.get());
  EXPECT_EQ(mock3->getUrl(), "early:/path3");
}

// Register logger after making logger calls with different protocols
TEST(RegisterLoggerFactoryTest, RegisterAfterMultipleLoggerCalls) {
  auto& factory = KINETO_NAMESPACE::ActivityProfilerController::loggerFactory();

  libkineto::registerLoggerFactory("first", [](const std::string& path) {
    return std::make_unique<MockActivityLogger>("1st:" + path);
  });

  for (int i = 0; i < 5; i++) {
    auto logger = factory.makeLogger("first:///call" + std::to_string(i));
    ASSERT_NE(logger, nullptr);
  }

  libkineto::registerLoggerFactory("second", [](const std::string& path) {
    return std::make_unique<MockActivityLogger>("2nd:" + path);
  });

  auto logger1 = factory.makeLogger("first:///path");
  auto logger2 = factory.makeLogger("second:///path");
  ASSERT_NE(logger1, nullptr);
  ASSERT_NE(logger2, nullptr);

  auto* mock1 = dynamic_cast<MockActivityLogger*>(logger1.get());
  auto* mock2 = dynamic_cast<MockActivityLogger*>(logger2.get());
  EXPECT_EQ(mock1->getUrl(), "1st:/path");
  EXPECT_EQ(mock2->getUrl(), "2nd:/path");
}

// Static singleton persistence
namespace {
bool g_persistence_registered = false;
}

TEST(RegisterLoggerFactoryTest, RegisterForPersistence) {
  if (!g_persistence_registered) {
    libkineto::registerLoggerFactory("persistent", [](const std::string& path) {
      return std::make_unique<MockActivityLogger>("persist:" + path);
    });
    g_persistence_registered = true;
  }

  auto& factory = KINETO_NAMESPACE::ActivityProfilerController::loggerFactory();
  auto logger = factory.makeLogger("persistent:///path");
  ASSERT_NE(logger, nullptr);
  auto* mock = dynamic_cast<MockActivityLogger*>(logger.get());
  EXPECT_EQ(mock->getUrl(), "persist:/path");
}

TEST(RegisterLoggerFactoryTest, UsePersistentLogger) {
  auto& factory = KINETO_NAMESPACE::ActivityProfilerController::loggerFactory();

  auto logger = factory.makeLogger("persistent:///another_path");
  ASSERT_NE(logger, nullptr);
  auto* mock = dynamic_cast<MockActivityLogger*>(logger.get());
  EXPECT_EQ(mock->getUrl(), "persist:/another_path");
}

// Register logger interleaved with factory usage
TEST(RegisterLoggerFactoryTest, InterleavedRegistrationAndUsage) {
  auto& factory = KINETO_NAMESPACE::ActivityProfilerController::loggerFactory();

  libkineto::registerLoggerFactory("interleave1", [](const std::string& path) {
    return std::make_unique<MockActivityLogger>("i1:" + path);
  });

  auto logger1 = factory.makeLogger("interleave1:///path");
  ASSERT_NE(logger1, nullptr);

  libkineto::registerLoggerFactory("interleave2", [](const std::string& path) {
    return std::make_unique<MockActivityLogger>("i2:" + path);
  });

  auto logger2 = factory.makeLogger("interleave2:///path");
  ASSERT_NE(logger2, nullptr);

  libkineto::registerLoggerFactory("interleave3", [](const std::string& path) {
    return std::make_unique<MockActivityLogger>("i3:" + path);
  });

  auto test1 = factory.makeLogger("interleave1:///test");
  auto test2 = factory.makeLogger("interleave2:///test");
  auto test3 = factory.makeLogger("interleave3:///test");

  ASSERT_NE(test1, nullptr);
  ASSERT_NE(test2, nullptr);
  ASSERT_NE(test3, nullptr);

  auto* mock1 = dynamic_cast<MockActivityLogger*>(test1.get());
  auto* mock2 = dynamic_cast<MockActivityLogger*>(test2.get());
  auto* mock3 = dynamic_cast<MockActivityLogger*>(test3.get());

  EXPECT_EQ(mock1->getUrl(), "i1:/test");
  EXPECT_EQ(mock2->getUrl(), "i2:/test");
  EXPECT_EQ(mock3->getUrl(), "i3:/test");
}

// Register logger from different thread, use from main thread
TEST(RegisterLoggerFactoryTest, RegisterFromThreadUseFromMain) {
  std::atomic<bool> registered{false};

  std::thread registrationThread([&registered]() {
    libkineto::registerLoggerFactory("threaded", [](const std::string& path) {
      return std::make_unique<MockActivityLogger>("thread:" + path);
    });
    registered = true;
  });

  registrationThread.join();
  ASSERT_TRUE(registered);

  auto& factory = KINETO_NAMESPACE::ActivityProfilerController::loggerFactory();
  auto logger = factory.makeLogger("threaded:///path");

  ASSERT_NE(logger, nullptr);
  auto* mock = dynamic_cast<MockActivityLogger*>(logger.get());
  EXPECT_EQ(mock->getUrl(), "thread:/path");
}

// Built-in "file" protocol remains functional
TEST(RegisterLoggerFactoryTest, BuiltInFileProtocolStillWorks) {
  auto& factory = KINETO_NAMESPACE::ActivityProfilerController::loggerFactory();

  auto logger1 = factory.makeLogger("file:///tmp/trace1.json");
  ASSERT_NE(logger1, nullptr);

  libkineto::registerLoggerFactory("custom1", [](const std::string& path) {
    return std::make_unique<MockActivityLogger>("custom1:" + path);
  });

  libkineto::registerLoggerFactory("custom2", [](const std::string& path) {
    return std::make_unique<MockActivityLogger>("custom2:" + path);
  });

  libkineto::registerLoggerFactory("custom3", [](const std::string& path) {
    return std::make_unique<MockActivityLogger>("custom3:" + path);
  });

  auto customLogger1 = factory.makeLogger("custom1:///path");
  auto customLogger2 = factory.makeLogger("custom2:///path");
  auto customLogger3 = factory.makeLogger("custom3:///path");

  ASSERT_NE(customLogger1, nullptr);
  ASSERT_NE(customLogger2, nullptr);
  ASSERT_NE(customLogger3, nullptr);

  auto logger2 = factory.makeLogger("file:///tmp/trace2.json");
  ASSERT_NE(logger2, nullptr);

  auto logger3 = factory.makeLogger("file:///tmp/trace3.json");
  ASSERT_NE(logger3, nullptr);
}

// Built-in "file" protocol is case-insensitive
TEST(RegisterLoggerFactoryTest, BuiltInFileProtocolCaseInsensitive) {
  auto& factory = KINETO_NAMESPACE::ActivityProfilerController::loggerFactory();

  auto logger1 = factory.makeLogger("file:///tmp/trace.json");
  ASSERT_NE(logger1, nullptr);

  auto logger2 = factory.makeLogger("FILE:///tmp/trace.json");
  ASSERT_NE(logger2, nullptr);

  auto logger3 = factory.makeLogger("File:///tmp/trace.json");
  ASSERT_NE(logger3, nullptr);

  auto logger4 = factory.makeLogger("FiLe:///tmp/trace.json");
  ASSERT_NE(logger4, nullptr);
}

// Overwriting built-in "file" protocol logs warning
TEST(RegisterLoggerFactoryTest, OverwritingFileProtocolLogsWarning) {
  auto& factory = KINETO_NAMESPACE::ActivityProfilerController::loggerFactory();

  auto logger1 = factory.makeLogger("file:///tmp/original.json");
  ASSERT_NE(logger1, nullptr);

  libkineto::registerLoggerFactory("file", [](const std::string& path) {
    return std::make_unique<MockActivityLogger>("overwritten:" + path);
  });

  auto logger2 = factory.makeLogger("file:///tmp/overwritten.json");
  ASSERT_NE(logger2, nullptr);

  auto* mock = dynamic_cast<MockActivityLogger*>(logger2.get());
  ASSERT_NE(mock, nullptr);
  EXPECT_EQ(mock->getUrl(), "overwritten:/tmp/overwritten.json");
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
