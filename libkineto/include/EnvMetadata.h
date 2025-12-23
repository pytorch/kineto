/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <fmt/core.h>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <unordered_map>

namespace libkineto {

enum class EnvVar : std::uint8_t {
  PT_PROFILER_JOB_NAME,
  PT_PROFILER_JOB_VERSION,
  PT_PROFILER_JOB_ATTEMPT_INDEX,
};

inline const std::unordered_map<EnvVar, const char*> K_ENV_VAR_MAP = {
    {EnvVar::PT_PROFILER_JOB_NAME, "PT_PROFILER_JOB_NAME"},
    {EnvVar::PT_PROFILER_JOB_VERSION, "PT_PROFILER_JOB_VERSION"},
    {EnvVar::PT_PROFILER_JOB_ATTEMPT_INDEX, "PT_PROFILER_JOB_ATTEMPT_INDEX"},
};

// Returns a map of (env_var_name, env_value) for all environment variables
// that are currently set. Only includes entries where the env var exists.
inline std::unordered_map<std::string, std::string> getEnvMetadata() {
  std::unordered_map<std::string, std::string> result;
  for (const auto& [key, name] : K_ENV_VAR_MAP) {
    if (const char* val = std::getenv(name)) {
      result[name] = fmt::format("\"{}\"", val);
    }
  }
  return result;
}

} // namespace libkineto
