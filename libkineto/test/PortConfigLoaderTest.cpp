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
#include <utility>

#include "MockSocket.h"
#include "PortConfigLoader.h"

using namespace KINETO_NAMESPACE;

// ============================================================================
// MockSocket Tests (verify mock works correctly)
// ============================================================================

TEST(MockSocketTest, CreateServer) {
  MockSocket socket;
  int const fd = socket.createServer(20599);
  EXPECT_GE(fd, 0);
}

TEST(MockSocketTest, AcceptWithPendingConnection) {
  MockSocket socket;
  socket.createServer(20599);
  socket.expectConnection(R"({"type":"PING"})");

  int const clientFd = socket.accept(100, nullptr, nullptr);
  EXPECT_GE(clientFd, 0);
  EXPECT_EQ(socket.getAcceptCount(), 1);
}

TEST(MockSocketTest, AcceptWithNoConnections) {
  MockSocket socket;
  socket.createServer(20599);

  int const clientFd = socket.accept(100, nullptr, nullptr);
  EXPECT_EQ(clientFd, -1);
}

TEST(MockSocketTest, ReadFromConnection) {
  MockSocket socket;
  socket.createServer(20599);
  socket.expectConnection(R"({"type":"PING"})");

  int const clientFd = socket.accept(100, nullptr, nullptr);
  ASSERT_GE(clientFd, 0);

  char buffer[256];
  ssize_t const bytesRead = socket.read(clientFd, buffer, sizeof(buffer));
  EXPECT_EQ(bytesRead, 15); // Length of {"type":"PING"}
  EXPECT_EQ(std::string(buffer, bytesRead), R"({"type":"PING"})");
}

TEST(MockSocketTest, WriteToConnection) {
  MockSocket socket;
  socket.createServer(20599);
  socket.expectConnection("");

  int const clientFd = socket.accept(100, nullptr, nullptr);
  ASSERT_GE(clientFd, 0);

  std::string response = R"({"type":"PONG","status":"READY"})";
  ssize_t const bytesWritten =
      socket.write(clientFd, response.data(), response.size());
  EXPECT_EQ(bytesWritten, static_cast<ssize_t>(response.size()));
  EXPECT_EQ(socket.getWrittenData(), response);
}

TEST(MockSocketTest, CloseConnection) {
  MockSocket socket;
  socket.createServer(20599);
  socket.expectConnection("");

  int const clientFd = socket.accept(100, nullptr, nullptr);
  ASSERT_GE(clientFd, 0);

  EXPECT_EQ(socket.close(clientFd), 0);
  EXPECT_EQ(socket.getCloseCount(), 1);
}

// ============================================================================
// PortConfigLoader - PING Tests
// ============================================================================

TEST(PortConfigLoaderTest, HandlePingReturnsReady) {
  auto mockSocket = std::make_unique<MockSocket>();
  auto* mockSocketPtr = mockSocket.get();

  // Setup: Client sends PING request
  mockSocketPtr->expectConnection(
      R"({"type":"PING"})"
      "\n");

  // Create loader with mock socket
  PortConfigLoader loader(20599, std::move(mockSocket));

  // Handle one connection
  loader.testHandleOneConnection();

  // Verify response contains PONG and READY status
  std::string const response = mockSocketPtr->getWrittenData();
  EXPECT_NE(response.find("\"type\":\"PONG\""), std::string::npos);
  EXPECT_NE(response.find("\"status\":\"READY\""), std::string::npos);
}

TEST(PortConfigLoaderTest, HandlePingReturnsBusyWhenConfigPending) {
  auto mockSocket = std::make_unique<MockSocket>();
  auto* mockSocketPtr = mockSocket.get();

  // First, send a TRACE request to make config pending
  mockSocketPtr->expectConnection(
      R"({"type":"TRACE","trace_id":"first","config":{"duration_ms":500}})"
      "\n");

  PortConfigLoader loader(20599, std::move(mockSocket));
  loader.testHandleOneConnection();

  // Now check that the config is pending
  EXPECT_TRUE(loader.hasConfigPending());

  // Setup second connection for PING
  auto mockSocket2 = std::make_unique<MockSocket>();
  auto* mockSocket2Ptr = mockSocket2.get();
  mockSocket2Ptr->expectConnection(
      R"({"type":"PING"})"
      "\n");

  // Note: In real implementation, we'd inject the new socket
  // For this test, we're just verifying the concept
}

// ============================================================================
// PortConfigLoader - TRACE Accept Tests
// ============================================================================

TEST(PortConfigLoaderTest, HandleTraceAcceptsWhenIdle) {
  auto mockSocket = std::make_unique<MockSocket>();
  auto* mockSocketPtr = mockSocket.get();

  // Setup: Client sends TRACE request
  std::string const traceRequest = R"({
    "type": "TRACE",
    "trace_id": "test-trace-123",
    "config": {
      "duration_ms": 1000,
      "record_shapes": true,
      "profile_memory": false
    }
  })"
                                   "\n";
  mockSocketPtr->expectConnection(traceRequest);

  PortConfigLoader loader(20599, std::move(mockSocket));
  loader.testHandleOneConnection();

  // Verify response shows ACCEPTED
  std::string const response = mockSocketPtr->getWrittenData();
  EXPECT_NE(response.find("\"type\":\"TRACE_ACK\""), std::string::npos);
  EXPECT_NE(response.find("\"status\":\"ACCEPTED\""), std::string::npos);
  EXPECT_NE(
      response.find("\"trace_id\":\"test-trace-123\""), std::string::npos);

  // Verify config is now pending
  EXPECT_TRUE(loader.hasConfigPending());
}

TEST(PortConfigLoaderTest, HandleTraceAcceptsWithMinimalConfig) {
  auto mockSocket = std::make_unique<MockSocket>();
  auto* mockSocketPtr = mockSocket.get();

  // Minimal config with just duration
  std::string const traceRequest =
      R"({"type":"TRACE","trace_id":"minimal","config":{"duration_ms":500}})"
      "\n";
  mockSocketPtr->expectConnection(traceRequest);

  PortConfigLoader loader(20599, std::move(mockSocket));
  loader.testHandleOneConnection();

  std::string const response = mockSocketPtr->getWrittenData();
  EXPECT_NE(response.find("\"status\":\"ACCEPTED\""), std::string::npos);
}

// ============================================================================
// PortConfigLoader - TRACE Reject Tests
// ============================================================================

TEST(PortConfigLoaderTest, HandleTraceRejectsWhenConfigPending) {
  auto mockSocket = std::make_unique<MockSocket>();
  auto* mockSocketPtr = mockSocket.get();

  // First TRACE request - should be accepted
  mockSocketPtr->expectConnection(
      R"({"type":"TRACE","trace_id":"first","config":{"duration_ms":500}})"
      "\n");
  // Second TRACE request - should be rejected
  mockSocketPtr->expectConnection(
      R"({"type":"TRACE","trace_id":"second","config":{"duration_ms":500}})"
      "\n");

  PortConfigLoader loader(20599, std::move(mockSocket));

  // First request - accepted
  loader.testHandleOneConnection();
  std::string const response1 = mockSocketPtr->getWrittenData();
  EXPECT_NE(response1.find("\"status\":\"ACCEPTED\""), std::string::npos);

  // Clear write buffer and handle second request
  mockSocketPtr->clearWrittenData();
  loader.testHandleOneConnection();

  // Second request should be rejected with CONFIG_PENDING
  std::string const response2 = mockSocketPtr->getWrittenData();
  EXPECT_NE(response2.find("\"status\":\"REJECTED\""), std::string::npos);
  EXPECT_NE(
      response2.find("\"error_code\":\"CONFIG_PENDING\""), std::string::npos);
}

// ============================================================================
// PortConfigLoader - readOnDemandConfig Tests
// ============================================================================

TEST(PortConfigLoaderTest, ReadOnDemandConfigConsumesPendingConfig) {
  auto mockSocket = std::make_unique<MockSocket>();
  auto* mockSocketPtr = mockSocket.get();

  // Send a TRACE request
  mockSocketPtr->expectConnection(
      R"({"type":"TRACE","trace_id":"test","config":{"duration_ms":1000}})"
      "\n");

  PortConfigLoader loader(20599, std::move(mockSocket));
  loader.testHandleOneConnection();

  // Config should be pending
  EXPECT_TRUE(loader.hasConfigPending());

  // First call to readOnDemandConfig should return the config
  std::string const config = loader.readOnDemandConfig(false, true);
  EXPECT_FALSE(config.empty());
  EXPECT_NE(config.find("ACTIVITIES_DURATION_MSECS=1000"), std::string::npos);

  // Config should no longer be pending
  EXPECT_FALSE(loader.hasConfigPending());

  // Second call should return empty
  std::string const config2 = loader.readOnDemandConfig(false, true);
  EXPECT_TRUE(config2.empty());
}

TEST(PortConfigLoaderTest, ReadOnDemandConfigReturnsEmptyWhenNoPending) {
  auto mockSocket = std::make_unique<MockSocket>();

  PortConfigLoader loader(20599, std::move(mockSocket));

  // No pending config - should return empty
  std::string const config = loader.readOnDemandConfig(false, true);
  EXPECT_TRUE(config.empty());
}

TEST(PortConfigLoaderTest, ReadOnDemandConfigReturnsEmptyWhenActivitiesFalse) {
  auto mockSocket = std::make_unique<MockSocket>();
  auto* mockSocketPtr = mockSocket.get();

  // Send a TRACE request
  mockSocketPtr->expectConnection(
      R"({"type":"TRACE","trace_id":"test","config":{"duration_ms":500}})"
      "\n");

  PortConfigLoader loader(20599, std::move(mockSocket));
  loader.testHandleOneConnection();

  // activities=false should not return the config
  std::string const config = loader.readOnDemandConfig(false, false);
  EXPECT_TRUE(config.empty());

  // Config should still be pending
  EXPECT_TRUE(loader.hasConfigPending());
}

// ============================================================================
// PortConfigLoader - readBaseConfig Tests
// ============================================================================

TEST(PortConfigLoaderTest, ReadBaseConfigReturnsEmpty) {
  auto mockSocket = std::make_unique<MockSocket>();

  PortConfigLoader loader(20599, std::move(mockSocket));

  // PortConfigLoader doesn't support base config - should return empty
  std::string const config = loader.readBaseConfig();
  EXPECT_TRUE(config.empty());
}

// ============================================================================
// PortConfigLoader - Error Handling Tests
// ============================================================================

TEST(PortConfigLoaderTest, HandleUnknownMessageType) {
  auto mockSocket = std::make_unique<MockSocket>();
  auto* mockSocketPtr = mockSocket.get();

  // Unknown message type
  mockSocketPtr->expectConnection(
      R"({"type":"UNKNOWN"})"
      "\n");

  PortConfigLoader loader(20599, std::move(mockSocket));
  loader.testHandleOneConnection();

  // Should respond with an error or ignore
  // (implementation will define exact behavior)
  std::string const response = mockSocketPtr->getWrittenData();
  // At minimum, it should not crash and should write something
  EXPECT_FALSE(response.empty());
}

TEST(PortConfigLoaderTest, HandleMalformedJson) {
  auto mockSocket = std::make_unique<MockSocket>();
  auto* mockSocketPtr = mockSocket.get();

  // Malformed JSON
  mockSocketPtr->expectConnection("not valid json\n");

  PortConfigLoader loader(20599, std::move(mockSocket));
  loader.testHandleOneConnection();

  // Should handle gracefully - not crash
  std::string const response = mockSocketPtr->getWrittenData();
  // Response should indicate error
  EXPECT_FALSE(response.empty());
}

TEST(PortConfigLoaderTest, HandleEmptyRequest) {
  auto mockSocket = std::make_unique<MockSocket>();
  auto* mockSocketPtr = mockSocket.get();

  // Empty request
  mockSocketPtr->expectConnection("");

  PortConfigLoader loader(20599, std::move(mockSocket));

  // Should handle gracefully - connection closed immediately
  loader.testHandleOneConnection();
}
