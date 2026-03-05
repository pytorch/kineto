/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "PortConfigLoader.h"

#include <ISocket.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#include <cstdlib>
#include <memory>
#include <string>

#include <utility>

#include "ConfigLoader.h"
#include "ILoggerObserver.h"
#include "Logger.h"
#include "TraceProtocol.h"

namespace KINETO_NAMESPACE {

namespace {

// Default socket implementation using POSIX sockets.
class PosixSocket : public ISocket {
 public:
  int createServer(uint16_t port) override {
    int const fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
      return -1;
    }

    int opt = 1;
    if (::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0) {
      ::close(fd);
      return -1;
    }

    struct sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(port);

    if (::bind(fd, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) <
        0) {
      ::close(fd);
      return -1;
    }

    if (::listen(fd, 5) < 0) {
      ::close(fd);
      return -1;
    }

    return fd;
  }

  int accept(int serverFd, struct sockaddr* addr, socklen_t* addrlen) override {
    return ::accept(serverFd, addr, addrlen);
  }

  ssize_t read(int fd, void* buf, size_t count) override {
    return ::read(fd, buf, count);
  }

  ssize_t write(int fd, const void* buf, size_t count) override {
    return ::write(fd, buf, count);
  }

  int close(int fd) override {
    return ::close(fd);
  }
};

} // namespace

PortConfigLoader::PortConfigLoader()
    : PortConfigLoader(std::make_unique<PosixSocket>()) {}

PortConfigLoader::PortConfigLoader(std::unique_ptr<ISocket> socket)
    : socket_(std::move(socket)) {
  // Read port from environment variable
  const char* portEnv = std::getenv("KINETO_TRACE_PORT");
  if (portEnv != nullptr) {
    port_ = static_cast<uint16_t>(std::atoi(portEnv));
  }
  startServer();
}

PortConfigLoader::PortConfigLoader(
    uint16_t port,
    std::unique_ptr<ISocket> socket)
    : socket_(std::move(socket)), port_(port) {
  // Don't start server automatically in test mode
  // Tests will call testHandleOneConnection() directly
}

PortConfigLoader::~PortConfigLoader() {
  running_ = false;
  if (serverFd_ >= 0) {
    socket_->close(serverFd_);
  }
  if (serverThread_ && serverThread_->joinable()) {
    serverThread_->join();
  }
}

std::string PortConfigLoader::readOnDemandConfig(
    bool /* events */,
    bool activities) {
  std::lock_guard<std::mutex> const lock(configMutex_);
  if (!activities || !configPending_) {
    return "";
  }

  std::string config = std::move(pendingConfig_);
  configPending_ = false;
  pendingConfig_.clear();
  LOG(INFO) << "Received on-demand config from TCP port " << port_ << ":\n"
            << config;
  return config;
}

bool PortConfigLoader::hasConfigPending() const {
  std::lock_guard<std::mutex> const lock(configMutex_);
  return configPending_;
}

void PortConfigLoader::testHandleOneConnection() {
  // Initialize server if not already done
  if (serverFd_ < 0) {
    serverFd_ = socket_->createServer(port_);
  }

  struct sockaddr_in clientAddr{};
  socklen_t addrLen = sizeof(clientAddr);
  int const clientFd = socket_->accept(
      serverFd_, reinterpret_cast<struct sockaddr*>(&clientAddr), &addrLen);

  if (clientFd >= 0) {
    handleClient(clientFd);
    socket_->close(clientFd);
  }
}

void PortConfigLoader::startServer() {
  serverFd_ = socket_->createServer(port_);
  if (serverFd_ < 0) {
    LOG(WARNING) << "PortConfigLoader: Failed to create server on port "
                 << port_;
    return;
  }

  running_ = true;
  serverThread_ = std::make_unique<std::thread>([this]() { serverLoop(); });
}

void PortConfigLoader::serverLoop() {
  while (running_) {
    struct sockaddr_in clientAddr{};
    socklen_t addrLen = sizeof(clientAddr);
    int const clientFd = socket_->accept(
        serverFd_, reinterpret_cast<struct sockaddr*>(&clientAddr), &addrLen);

    if (clientFd < 0) {
      if (running_) {
        LOG(WARNING) << "PortConfigLoader: Accept failed";
      }
      continue;
    }

    handleClient(clientFd);
    socket_->close(clientFd);
  }
}

void PortConfigLoader::handleClient(int clientFd) {
  char buffer[4096];
  ssize_t const bytesRead = socket_->read(clientFd, buffer, sizeof(buffer) - 1);
  if (bytesRead <= 0) {
    return;
  }
  buffer[bytesRead] = '\0';

  std::string const request(buffer, bytesRead);
  std::string response;

  // Parse JSON to determine message type
  std::string const msgType = extractJsonString(request, "type", "");

  if (msgType == "PING") {
    response = handlePing();
  } else if (msgType == "TRACE") {
    response = handleTrace(request);
  } else {
    response = R"({"type":"ERROR","message":"unknown_command"})"
               "\n";
  }

  socket_->write(clientFd, response.c_str(), response.size());
}

std::string PortConfigLoader::handlePing() const {
  std::lock_guard<std::mutex> const lock(configMutex_);
  if (configPending_) {
    return R"({"type":"PONG","status":"BUSY"})"
           "\n";
  }
  return R"({"type":"PONG","status":"READY"})"
         "\n";
}

std::string PortConfigLoader::handleTrace(std::string_view jsonPayload) {
  std::lock_guard<std::mutex> const lock(configMutex_);

  // Extract trace_id for response
  std::string const traceId =
      extractJsonString(jsonPayload, "trace_id", "unknown");

  // Config Pending Guard: reject if config not yet consumed
  if (configPending_) {
    return R"({"type":"TRACE_ACK","status":"REJECTED","trace_id":")" + traceId +
        R"(","error_code":"CONFIG_PENDING","error_message":"A trace config is pending"})"
        "\n";
  }

  // Parse JSON and build config string
  int const durationMs = extractJsonInt(jsonPayload, "duration_ms", 5000);
  std::string const activities =
      extractJsonString(jsonPayload, "activities", "CUDA,CPU");
  bool const recordShapes =
      extractJsonBool(jsonPayload, "record_shapes", false);
  bool const profileMemory =
      extractJsonBool(jsonPayload, "profile_memory", false);
  bool const withStack = extractJsonBool(jsonPayload, "with_stack", false);
  bool const withFlops = extractJsonBool(jsonPayload, "with_flops", false);
  bool const withModules = extractJsonBool(jsonPayload, "with_modules", false);
  std::string const outputDir =
      extractJsonString(jsonPayload, "output_dir", "");

  pendingConfig_ = buildConfigString(
      durationMs,
      activities,
      recordShapes,
      profileMemory,
      withStack,
      withFlops,
      withModules,
      outputDir,
      traceId);
  configPending_ = true;

  return R"({"type":"TRACE_ACK","status":"ACCEPTED","trace_id":")" + traceId +
      R"("})"
      "\n";
}

void PortConfigLoader::registerFactory() {
  // Use the new addConfigLoaderFactory API which allows multiple config loaders
  // to coexist (e.g., DaemonConfigLoader + PortConfigLoader)
  ConfigLoader::addConfigLoaderFactory(
      []() -> std::unique_ptr<IDaemonConfigLoader> {
        return std::make_unique<PortConfigLoader>();
      });
}

} // namespace KINETO_NAMESPACE
