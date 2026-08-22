/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "TraceProtocol.h"

#include <cctype>
#include <cstddef>

namespace KINETO_NAMESPACE {

namespace {

// Find the position of a key in JSON, returning npos if not found.
size_t findKey(std::string_view json, std::string_view key) {
  // Look for "key" pattern
  std::string const pattern = "\"" + std::string(key) + "\"";
  return json.find(pattern);
}

// Skip whitespace and colon after key.
size_t skipToValue(std::string_view json, size_t pos) {
  while (pos < json.size() &&
         ((std::isspace(json[pos]) != 0) || json[pos] == ':')) {
    ++pos;
  }
  return pos;
}

} // namespace

int extractJsonInt(
    std::string_view json,
    std::string_view key,
    int defaultVal) {
  size_t const keyPos = findKey(json, key);
  if (keyPos == std::string_view::npos) {
    return defaultVal;
  }

  size_t pos = skipToValue(json, keyPos + key.size() + 2);
  if (pos >= json.size()) {
    return defaultVal;
  }

  // Parse integer
  int result = 0;
  bool negative = false;
  if (json[pos] == '-') {
    negative = true;
    ++pos;
  }
  while (pos < json.size() && (std::isdigit(json[pos]) != 0)) {
    result = result * 10 + (json[pos] - '0');
    ++pos;
  }
  return negative ? -result : result;
}

bool extractJsonBool(
    std::string_view json,
    std::string_view key,
    bool defaultVal) {
  size_t const keyPos = findKey(json, key);
  if (keyPos == std::string_view::npos) {
    return defaultVal;
  }

  size_t const pos = skipToValue(json, keyPos + key.size() + 2);
  if (pos >= json.size()) {
    return defaultVal;
  }

  if (json.substr(pos, 4) == "true") {
    return true;
  } else if (json.substr(pos, 5) == "false") {
    return false;
  }
  return defaultVal;
}

std::string extractJsonString(
    std::string_view json,
    std::string_view key,
    std::string_view defaultVal) {
  size_t const keyPos = findKey(json, key);
  if (keyPos == std::string_view::npos) {
    return std::string(defaultVal);
  }

  size_t pos = skipToValue(json, keyPos + key.size() + 2);
  if (pos >= json.size() || json[pos] != '"') {
    return std::string(defaultVal);
  }

  // Skip opening quote
  ++pos;
  size_t const endPos = json.find('"', pos);
  if (endPos == std::string_view::npos) {
    return std::string(defaultVal);
  }

  return std::string(json.substr(pos, endPos - pos));
}

std::string buildConfigString(
    int durationMs,
    const std::string& activities,
    bool recordShapes,
    bool profileMemory,
    bool withStack,
    bool withFlops,
    bool withModules,
    const std::string& outputDir,
    const std::string& traceId) {
  std::string config;

  config += "ACTIVITIES_DURATION_MSECS=" + std::to_string(durationMs) + "\n";

  if (!activities.empty()) {
    config += "ACTIVITIES=" + activities + "\n";
  }

  if (recordShapes) {
    config += "PROFILE_REPORT_INPUT_SHAPES=true\n";
  }

  if (profileMemory) {
    config += "PROFILE_PROFILE_MEMORY=true\n";
  }

  if (withStack) {
    config += "PROFILE_WITH_STACK=true\n";
  }

  if (withFlops) {
    config += "PROFILE_WITH_FLOPS=true\n";
  }

  if (withModules) {
    config += "PROFILE_WITH_MODULES=true\n";
  }

  // Set the output file path if outputDir and traceId are provided
  if (!outputDir.empty() && !traceId.empty()) {
    config += "ACTIVITIES_LOG_FILE=" + outputDir + "/" + traceId + ".json\n";
  }

  return config;
}

} // namespace KINETO_NAMESPACE
