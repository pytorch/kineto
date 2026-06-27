/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <sys/socket.h>
#include <cstddef>
#include <cstdint>
#include <memory>

#include <sys/types.h>

namespace KINETO_NAMESPACE {

// Abstract socket interface for dependency injection and testability.
// This allows PortConfigLoader to be tested without binding to real ports.
class ISocket {
 public:
  virtual ~ISocket() = default;

  // Create a TCP server socket bound to the given port.
  // Returns the server file descriptor, or -1 on error.
  virtual int createServer(uint16_t port) = 0;

  // Accept a connection on the server socket.
  // Returns the client file descriptor, or -1 on error.
  virtual int
  accept(int server_fd, struct sockaddr* addr, socklen_t* addrlen) = 0;

  // Read from a file descriptor.
  // Returns the number of bytes read, 0 on EOF, or -1 on error.
  virtual ssize_t read(int fd, void* buf, size_t count) = 0;

  // Write to a file descriptor.
  // Returns the number of bytes written, or -1 on error.
  virtual ssize_t write(int fd, const void* buf, size_t count) = 0;

  // Close a file descriptor.
  // Returns 0 on success, or -1 on error.
  virtual int close(int fd) = 0;
};

// Factory function type for creating socket implementations
using SocketFactory = std::unique_ptr<ISocket> (*)();

} // namespace KINETO_NAMESPACE
