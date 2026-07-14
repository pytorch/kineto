/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>

#include "DaemonConfigLoader.h"
#include "ISocket.h"

namespace KINETO_NAMESPACE {

// PortConfigLoader implements IDaemonConfigLoader to receive on-demand trace
// requests via a TCP port. This provides an alternative to DaemonConfigLoader
// for Kubernetes environments where IPC Fabric is not available.
//
// Protocol:
//   PING -> READY | BUSY (health check)
//   TRACE <json> -> TRACE_ACK | TRACE_BUSY | TRACE_ERROR (trigger trace)
//
// Port is configured via KINETO_TRACE_PORT environment variable (default:
// 20599). The loader runs a TCP server thread that accepts connections and
// processes commands.
class PortConfigLoader : public IDaemonConfigLoader {
 public:
  // Create with default PosixSocket.
  PortConfigLoader();

  // Create with injected socket (for testing).
  explicit PortConfigLoader(std::unique_ptr<ISocket> socket);

  // Create with port and injected socket (for testing).
  PortConfigLoader(uint16_t port, std::unique_ptr<ISocket> socket);

  ~PortConfigLoader() override;

  // IDaemonConfigLoader interface
  std::string readBaseConfig() override {
    return ""; // Not supported for port-based loader
  }

  std::string readOnDemandConfig(bool events, bool activities) override;

  int gpuContextCount(uint32_t /*device*/) override {
    return 0; // Not applicable for port-based loader
  }

  void setCommunicationFabric(bool enabled) override {
    // No-op for port-based loader
  }

  // Check if a trace request is pending (Config Pending Guard).
  bool hasConfigPending() const;

  // Test hook: handle one connection manually (for unit tests).
  void testHandleOneConnection();

 private:
  // Start the TCP server thread.
  void startServer();

  // Server thread main loop.
  void serverLoop();

  // Handle a single client connection.
  void handleClient(int clientFd);

  // Handle PING command.
  std::string handlePing() const;

  // Handle TRACE command.
  std::string handleTrace(std::string_view jsonPayload);

  std::unique_ptr<ISocket> socket_;
  std::unique_ptr<std::thread> serverThread_;
  std::atomic<bool> running_{false};

  // Config Pending Guard: prevents overwriting pending config.
  mutable std::mutex configMutex_;
  bool configPending_{false};
  std::string pendingConfig_;

  // Port configuration.
  uint16_t port_{20599};
  int serverFd_{-1};
};

} // namespace KINETO_NAMESPACE
