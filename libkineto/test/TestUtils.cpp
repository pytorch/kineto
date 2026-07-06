/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "test/TestUtils.h"

#include <gtest/gtest.h>

#include <unistd.h>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace libkineto::test {

TempTraceFile::TempTraceFile(std::string_view prefix, std::string_view suffix) {
  std::string nameTemplate;
  nameTemplate.reserve(5 + prefix.size() + 6 + suffix.size());
  nameTemplate += "/tmp/";
  nameTemplate += prefix;
  nameTemplate += "XXXXXX";
  nameTemplate += suffix;

  const int fd = mkstemps(nameTemplate.data(), static_cast<int>(suffix.size()));
  if (fd < 0) {
    throw std::runtime_error("mkstemps failed for " + nameTemplate);
  }
  close(fd);
  path_ = std::move(nameTemplate);
}

TempTraceFile::~TempTraceFile() {
  if (!path_.empty()) {
    unlink(path_.c_str());
  }
}

TempTraceFile::TempTraceFile(TempTraceFile&& other) noexcept
    : path_(std::move(other.path_)) {
  other.path_.clear();
}

TempTraceFile& TempTraceFile::operator=(TempTraceFile&& other) noexcept {
  if (this != &other) {
    if (!path_.empty()) {
      unlink(path_.c_str());
    }
    path_ = std::move(other.path_);
    other.path_.clear();
  }
  return *this;
}

TempTraceFile createTempTraceFile(
    std::string_view prefix,
    std::string_view suffix) {
  return TempTraceFile(prefix, suffix);
}

} // namespace libkineto::test
