/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <sys/socket.h>
#include <cstdint>
#include <cstring>
#include <queue>
#include <string>

#include "src/ISocket.h"

namespace KINETO_NAMESPACE {

// Mock socket implementation for unit testing.
// Uses in-memory buffers instead of real sockets, enabling deterministic
// tests without binding to real ports.
class MockSocket : public ISocket {
 public:
  MockSocket() = default;

  // Pre-configure an expected connection with request data.
  // When accept() is called, it will return a fake client FD and
  // subsequent read() calls will return this request data.
  void expectConnection(const std::string& requestData) {
    pendingConnections_.push(requestData);
  }

  // Check if there are pending connections
  [[nodiscard]] bool hasPendingConnections() const {
    return !pendingConnections_.empty();
  }

  // Get the data that was written to the socket (for verification)
  [[nodiscard]] const std::string& getWrittenData() const {
    return writeBuffer_;
  }

  // Clear the write buffer for the next test
  void clearWrittenData() {
    writeBuffer_.clear();
  }

  // Get the number of times accept() was called
  [[nodiscard]] int getAcceptCount() const {
    return acceptCount_;
  }

  // Get the number of times close() was called
  [[nodiscard]] int getCloseCount() const {
    return closeCount_;
  }

  // ISocket interface implementation

  int createServer(uint16_t port) override {
    serverPort_ = port;
    return kFakeServerFd;
  }

  int accept(int serverFd, struct sockaddr* /*addr*/, socklen_t* /*addrlen*/)
      override {
    if (serverFd != kFakeServerFd) {
      return -1;
    }

    if (pendingConnections_.empty()) {
      // No more connections expected - simulate blocking or error
      return -1;
    }

    // Set up the read buffer with the expected request data
    currentReadBuffer_ = pendingConnections_.front();
    pendingConnections_.pop();
    readPos_ = 0;

    acceptCount_++;
    return kFakeClientFd;
  }

  ssize_t read(int fd, void* buf, size_t count) override {
    if (fd != kFakeClientFd) {
      return -1;
    }

    if (readPos_ >= currentReadBuffer_.size()) {
      return 0; // EOF
    }

    size_t const toRead = std::min(count, currentReadBuffer_.size() - readPos_);
    std::memcpy(buf, currentReadBuffer_.data() + readPos_, toRead);
    readPos_ += toRead;
    return static_cast<ssize_t>(toRead);
  }

  ssize_t write(int fd, const void* buf, size_t count) override {
    if (fd != kFakeClientFd) {
      return -1;
    }

    writeBuffer_.append(static_cast<const char*>(buf), count);
    return static_cast<ssize_t>(count);
  }

  int close(int fd) override {
    if (fd == kFakeClientFd || fd == kFakeServerFd) {
      closeCount_++;
      return 0;
    }
    return -1;
  }

 private:
  static constexpr int kFakeServerFd = 100;
  static constexpr int kFakeClientFd = 101;

  uint16_t serverPort_{0};
  std::queue<std::string> pendingConnections_;
  std::string currentReadBuffer_;
  size_t readPos_{0};
  std::string writeBuffer_;
  int acceptCount_{0};
  int closeCount_{0};
};

} // namespace KINETO_NAMESPACE
