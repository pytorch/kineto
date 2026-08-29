/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <string>
#include <string_view>

namespace KINETO_NAMESPACE {

// JSON field extraction for TRACE request parsing.
// These functions use simple hand-rolled parsing to avoid adding JSON library
// dependencies.

// Extract an integer value from JSON for the given key.
// Returns defaultVal if the key is not found or parsing fails.
int extractJsonInt(std::string_view json, std::string_view key, int defaultVal);

// Extract a boolean value from JSON for the given key.
// Returns defaultVal if the key is not found or parsing fails.
bool extractJsonBool(
    std::string_view json,
    std::string_view key,
    bool defaultVal);

// Extract a string value from JSON for the given key.
// Returns defaultVal if the key is not found or parsing fails.
std::string extractJsonString(
    std::string_view json,
    std::string_view key,
    std::string_view defaultVal);

// Build a Kineto config string from parsed JSON fields.
// Uses Kineto's KEY=VALUE\n format.
std::string buildConfigString(
    int durationMs,
    const std::string& activities,
    bool recordShapes,
    bool profileMemory,
    bool withStack,
    bool withFlops,
    bool withModules,
    const std::string& outputDir = "",
    const std::string& traceId = "");

} // namespace KINETO_NAMESPACE
