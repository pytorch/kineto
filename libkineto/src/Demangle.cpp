/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "Demangle.h"

#ifndef _MSC_VER
#include <cxxabi.h>
#endif
#include <string.h>
#include <string>

namespace KINETO_NAMESPACE {

static constexpr int kMaxSymbolSize = 1024;

std::string demangle(const char* name) {
#ifndef _MSC_VER
  if (!name) {
    return "";
  }

  if (strlen(name) > kMaxSymbolSize) {
    return name;
  }

  int status;
  size_t len = 0;
  char* demangled = abi::__cxa_demangle(name, nullptr, &len, &status);
  if (status != 0) {
    return name;
  }
  std::string res(demangled);
  // The returned buffer must be freed!
  free(demangled);
  return res;
#else
  // TODO: demangling on Windows
  if (!name) {
    return "";
  } else {
    return name;
  }
#endif
}

} // namespace KINETO_NAMESPACE
