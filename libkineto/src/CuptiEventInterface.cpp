/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "CuptiEventInterface.h"

#include <chrono>

#include "Logger.h"
#include "cupti_call.h"

using namespace std::chrono;
using std::vector;

namespace KINETO_NAMESPACE {

CuptiEventInterface::CuptiEventInterface(CUcontext context)
    : context_(context) {
  CUPTI_CALL(cuptiGetDeviceId(context_, (uint32_t*)&device_));
}

CUpti_EventGroupSets* CuptiEventInterface::createGroupSets(
    vector<CUpti_EventID>& ids) {
  CUpti_EventGroupSets* group_sets = nullptr;
  CUptiResult res = CUPTI_CALL(cuptiEventGroupSetsCreate(
      context_, sizeof(CUpti_EventID) * ids.size(), ids.data(), &group_sets));

  if (res != CUPTI_SUCCESS || group_sets == nullptr) {
    const char* errstr = nullptr;
    CUPTI_CALL(cuptiGetResultString(res, &errstr));
    throw std::system_error(EINVAL, std::generic_category(), errstr);
  }

  return group_sets;
}

void CuptiEventInterface::destroyGroupSets(CUpti_EventGroupSets* sets) {
  CUPTI_CALL(cuptiEventGroupSetsDestroy(sets));
}

bool CuptiEventInterface::setContinuousMode() {
  CUptiResult res = CUPTI_CALL(cuptiSetEventCollectionMode(
      context_, CUPTI_EVENT_COLLECTION_MODE_CONTINUOUS));
  return (res == CUPTI_SUCCESS);
}

void CuptiEventInterface::enablePerInstance(CUpti_EventGroup eventGroup) {
  uint32_t profile_all = 1;
  CUPTI_CALL(cuptiEventGroupSetAttribute(
      eventGroup,
      CUPTI_EVENT_GROUP_ATTR_PROFILE_ALL_DOMAIN_INSTANCES,
      sizeof(profile_all),
      &profile_all));
}

uint32_t CuptiEventInterface::instanceCount(CUpti_EventGroup eventGroup) {
  uint32_t instance_count = 0;
  size_t s = sizeof(instance_count);
  CUPTI_CALL(cuptiEventGroupGetAttribute(
      eventGroup, CUPTI_EVENT_GROUP_ATTR_INSTANCE_COUNT, &s, &instance_count));
  return instance_count;
}

void CuptiEventInterface::enableGroupSet(CUpti_EventGroupSet& set) {
  CUptiResult res = CUPTI_CALL_NOWARN(cuptiEventGroupSetEnable(&set));
  if (res != CUPTI_SUCCESS) {
    const char* errstr = nullptr;
    CUPTI_CALL(cuptiGetResultString(res, &errstr));
    throw std::system_error(EIO, std::generic_category(), errstr);
  }
}

void CuptiEventInterface::disableGroupSet(CUpti_EventGroupSet& set) {
  CUPTI_CALL(cuptiEventGroupSetDisable(&set));
}

void CuptiEventInterface::readEvent(
    CUpti_EventGroup grp,
    CUpti_EventID id,
    vector<int64_t>& vals) {
  size_t s = sizeof(int64_t) * vals.size();
  CUPTI_CALL(cuptiEventGroupReadEvent(
      grp,
      CUPTI_EVENT_READ_FLAG_NONE,
      id,
      &s,
      reinterpret_cast<uint64_t*>(vals.data())));
}

vector<CUpti_EventID> CuptiEventInterface::eventsInGroup(CUpti_EventGroup grp) {
  uint32_t group_size = 0;
  size_t s = sizeof(group_size);
  CUPTI_CALL(cuptiEventGroupGetAttribute(
      grp, CUPTI_EVENT_GROUP_ATTR_NUM_EVENTS, &s, &group_size));
  size_t events_size = group_size * sizeof(CUpti_EventID);
  vector<CUpti_EventID> res(group_size);
  CUPTI_CALL(cuptiEventGroupGetAttribute(
      grp, CUPTI_EVENT_GROUP_ATTR_EVENTS, &events_size, res.data()));
  return res;
}

CUpti_EventID CuptiEventInterface::eventId(const std::string& name) {
  CUpti_EventID id{0};
  CUPTI_CALL(cuptiEventGetIdFromName(device_, name.c_str(), &id));
  return id;
}

} // namespace KINETO_NAMESPACE
