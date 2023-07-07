/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cupti.h>
#include <queue>
#include <string>

namespace KINETO_NAMESPACE {

// C++ interface to CUPTI Events C API.
// Virtual methods are here mainly to allow easier testing.
class CuptiEventApi {
 public:
  explicit CuptiEventApi(CUcontext context_);
  virtual ~CuptiEventApi() {}

  CUdevice device() {
    return device_;
  }

  virtual CUpti_EventGroupSets* createGroupSets(
      std::vector<CUpti_EventID>& ids);
  virtual void destroyGroupSets(CUpti_EventGroupSets* sets);

  virtual bool setContinuousMode();

  virtual void enablePerInstance(CUpti_EventGroup eventGroup);
  virtual uint32_t instanceCount(CUpti_EventGroup eventGroup);

  virtual void enableGroupSet(CUpti_EventGroupSet& set);
  virtual void disableGroupSet(CUpti_EventGroupSet& set);

  virtual void
  readEvent(CUpti_EventGroup g, CUpti_EventID id, std::vector<int64_t>& vals);
  virtual std::vector<CUpti_EventID> eventsInGroup(CUpti_EventGroup g);

  virtual CUpti_EventID eventId(const std::string& name);

 protected:
  // Unit testing
  CuptiEventApi() : context_(nullptr), device_(0) {}

 private:
  CUcontext context_;
  CUdevice device_;
};

} // namespace KINETO_NAMESPACE
