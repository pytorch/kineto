/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>
#include <memory>
#include <thread>
#include <vector>

// TODO(T90238193)
// @lint-ignore-every CLANGTIDY facebook-hte-RelativeInclude
#include "LoggerCollector.h"
#include "include/libkineto.h"
#include "src/Logger.h"

using namespace KINETO_NAMESPACE;

#if !USE_GOOGLE_LOG

constexpr char InfoTestStr[] = "Checking LOG(INFO)";
constexpr char WarningTestStr[] = "Checking LOG(WARNING)";
constexpr char ErrorTestStr[] = "Checking LOG(ERROR)";

TEST(LoggerObserverTest, SingleCollectorObserver) {
  // Add a LoggerObserverCollector to collect all logs during the trace.
  std::unique_ptr<LoggerCollector> lCollector =
      std::make_unique<LoggerCollector>();
  Logger::addLoggerObserver(lCollector.get());

  LOG(INFO) << InfoTestStr;
  LOG(WARNING) << WarningTestStr;
  LOG(ERROR) << ErrorTestStr;

  auto LoggerMD = lCollector->extractCollectorMetadata();
  EXPECT_TRUE(
      LoggerMD[LoggerOutputType::INFO][0].find(InfoTestStr) !=
      std::string::npos);
  EXPECT_TRUE(
      LoggerMD[LoggerOutputType::WARNING][0].find(WarningTestStr) !=
      std::string::npos);
  EXPECT_TRUE(
      LoggerMD[LoggerOutputType::ERROR][0].find(ErrorTestStr) !=
      std::string::npos);

  Logger::removeLoggerObserver(lCollector.get());
}

#define NUM_OF_MESSAGES_FOR_EACH_TYPE 10
#define NUM_OF_WRITE_THREADS 200

// Writes NUM_OF_MESSAGES_FOR_EACH_TYPE messages for each INFO, WARNING, and
// ERROR.
static void* writeSeveralMessages() {
  for (int i = 0; i < NUM_OF_MESSAGES_FOR_EACH_TYPE; i++) {
    LOG(INFO) << InfoTestStr;
    LOG(WARNING) << WarningTestStr;
    LOG(ERROR) << ErrorTestStr;
  }
  return nullptr;
}

TEST(LoggerObserverTest, FourCollectorObserver) {
  // There shouldn't be too many CUPTIActivityProfilers active at the same time.
  std::unique_ptr<LoggerCollector> lc1 = std::make_unique<LoggerCollector>();
  std::unique_ptr<LoggerCollector> lc2 = std::make_unique<LoggerCollector>();
  std::unique_ptr<LoggerCollector> lc3 = std::make_unique<LoggerCollector>();
  std::unique_ptr<LoggerCollector> lc4 = std::make_unique<LoggerCollector>();
  Logger::addLoggerObserver(lc1.get());
  Logger::addLoggerObserver(lc2.get());
  Logger::addLoggerObserver(lc3.get());
  Logger::addLoggerObserver(lc4.get());

  // Launch NUM_OF_WRITE_THREADS threads writing several messages.
  std::vector<std::thread> ListOfThreads;
  for (int i = 0; i < NUM_OF_WRITE_THREADS; i++) {
    ListOfThreads.emplace_back(writeSeveralMessages);
  }

  // Wait for all threads to finish.
  for (auto& thread : ListOfThreads) {
    thread.join();
  }

  auto lc1MD = lc1->extractCollectorMetadata();
  int InfoCount = 0, WarnCount = 0, ErrorCount = 0;
  for (auto& md : lc1MD) {
    InfoCount += md.first == LoggerOutputType::INFO ? md.second.size() : 0;
    WarnCount += md.first == LoggerOutputType::WARNING ? md.second.size() : 0;
    ErrorCount += md.first == LoggerOutputType::ERROR ? md.second.size() : 0;
  }

  EXPECT_EQ(InfoCount, NUM_OF_WRITE_THREADS * NUM_OF_MESSAGES_FOR_EACH_TYPE);
  EXPECT_EQ(WarnCount, NUM_OF_WRITE_THREADS * NUM_OF_MESSAGES_FOR_EACH_TYPE);
  EXPECT_EQ(ErrorCount, NUM_OF_WRITE_THREADS * NUM_OF_MESSAGES_FOR_EACH_TYPE);

  Logger::removeLoggerObserver(lc1.get());
  Logger::removeLoggerObserver(lc2.get());
  Logger::removeLoggerObserver(lc3.get());
  Logger::removeLoggerObserver(lc4.get());
}

#endif // !USE_GOOGLE_LOG

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
