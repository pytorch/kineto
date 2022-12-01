/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

namespace libkineto {
class IConfigHandler;
}

namespace KINETO_NAMESPACE {

class IConfigLoader {
 public:

  enum ConfigKind {
    ActivityProfiler = 0,
    EventProfiler,
    NumConfigKinds
  };

  virtual ~IConfigLoader() {};


  virtual void addHandler(ConfigKind kind, IConfigHandler* handler) = 0;
  virtual void removeHandler(ConfigKind kind, IConfigHandler* handler) = 0;
};

} // namespace KINETO_NAMESPAC
