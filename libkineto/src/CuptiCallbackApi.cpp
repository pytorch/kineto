/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// TODO(T90238193)
// @lint-ignore-every CLANGTIDY facebook-hte-RelativeInclude
#include "CuptiCallbackApi.h"
#include "CuptiActivityApi.h"

#include <assert.h>
#include <chrono>
#include <algorithm>
#include <mutex>

#ifdef HAS_CUPTI
#include "cupti_call.h"
#endif
#include "Logger.h"


namespace KINETO_NAMESPACE {

// limit on number of handles per callback type
constexpr size_t MAX_CB_FNS_PER_CB = 8;

// Use this value in enabledCallbacks_ set, when all cbids in a domain
// is enabled, not a specific cbid.
constexpr uint32_t MAX_CUPTI_CALLBACK_ID_ALL = 0xffffffff;

/* Callback Table :
 *  Overall goal of the design is to optimize the lookup of function
 *  pointers. The table is structured at two levels and the leaf
 *  elements in the table are std::list to enable fast access/inserts/deletes
 *
 *   <callback domain0> |
 *                     -> cb id 0 -> std::list of callbacks
 *                     ...
 *                     -> cb id n -> std::list of callbacks
 *   <callback domain1> |
 *                    ...
 *  CallbackTable is the finaly table type above
 *  See type declrartions in header file.
 */


/* callback_switchboard : is the global callback handler we register
 *  with CUPTI. The goal is to make it as efficient as possible
 *  to re-direct to the registered callback(s).
 *
 *  Few things to care about :
 *   a) use if/then switches rather than map/hash structures
 *   b) avoid dynamic memory allocations
 *   c) be aware of locking overheads
 */
#ifdef HAS_CUPTI
static void CUPTIAPI callback_switchboard(
#else
static void callback_switchboard(
#endif
   void* /* unused */,
   CUpti_CallbackDomain domain,
   CUpti_CallbackId cbid,
   const CUpti_CallbackData* cbInfo) {

  // below statement is likey going to call a mutex
  // on the singleton access
  CuptiCallbackApi::singleton()->__callback_switchboard(
      domain, cbid, cbInfo);
}


void CuptiCallbackApi::__callback_switchboard(
   CUpti_CallbackDomain domain,
   CUpti_CallbackId cbid,
   const CUpti_CallbackData* cbInfo) {
  VLOG(0) << "Callback: domain = " << domain << ", cbid = " << cbid;
  CallbackList *cblist = nullptr;

  switch (domain) {
    // add the fastest path for kernel launch callbacks
    // as these are the most frequent ones
    case CUPTI_CB_DOMAIN_RUNTIME_API:
      switch (cbid) {
        case CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000:
          cblist = &callbacks_.runtime[
            CUDA_LAUNCH_KERNEL - __RUNTIME_CB_DOMAIN_START];
          break;
        default:
          break;
      }
      // This is required to teardown cupti after profiling to prevent QPS slowdown.
      if (CuptiActivityApi::singleton().teardownCupti_) {
        if (cbInfo->callbackSite == CUPTI_API_EXIT) {
          LOG(INFO) << "  Calling cuptiFinalize in exit callsite";
          // Teardown CUPTI calling cuptiFinalize()
          CUPTI_CALL(cuptiUnsubscribe(subscriber_));
          CUPTI_CALL(cuptiFinalize());
          initSuccess_ = false;
          subscriber_ = 0;
          CuptiActivityApi::singleton().teardownCupti_ = 0;
          CuptiActivityApi::singleton().finalizeCond_.notify_all();
          return;
        }
      }
      break;

    case CUPTI_CB_DOMAIN_RESOURCE:
      switch (cbid) {
        case CUPTI_CBID_RESOURCE_CONTEXT_CREATED:
          cblist = &callbacks_.resource[
            RESOURCE_CONTEXT_CREATED - __RESOURCE_CB_DOMAIN_START];
          break;
        case CUPTI_CBID_RESOURCE_CONTEXT_DESTROY_STARTING:
          cblist = &callbacks_.resource[
            RESOURCE_CONTEXT_DESTROYED - __RESOURCE_CB_DOMAIN_START];
          break;
        default:
          break;
      }
      break;

    default:
      return;
  }

  // ignore callbacks that are not handled
  if (cblist == nullptr) {
    return;
  }

  // make a copy of the callback list so we avoid holding lock
  // in common case this should be just one func pointer copy
  std::array<CuptiCallbackFn, MAX_CB_FNS_PER_CB> callbacks;
  int num_cbs = 0;
  {
    ReaderLockGuard rl(callbackLock_);
    int i = 0;
    for (auto it = cblist->begin();
        it != cblist->end() && i < MAX_CB_FNS_PER_CB;
        it++, i++) {
      callbacks[i] = *it;
    }
    num_cbs = i;
  }

  for (int i = 0; i < num_cbs; i++) {
    auto fn = callbacks[i];
    fn(domain, cbid, cbInfo);
  }
}

std::shared_ptr<CuptiCallbackApi> CuptiCallbackApi::singleton() {
	static const std::shared_ptr<CuptiCallbackApi>
		instance = [] {
			std::shared_ptr<CuptiCallbackApi> inst =
				std::shared_ptr<CuptiCallbackApi>(new CuptiCallbackApi());
			return inst;
	}();
  return instance;
}

void CuptiCallbackApi::initCallbackApi() {
#ifdef HAS_CUPTI
  lastCuptiStatus_ = CUPTI_ERROR_UNKNOWN;
  lastCuptiStatus_ = CUPTI_CALL_NOWARN(
    cuptiSubscribe(&subscriber_,
      (CUpti_CallbackFunc)callback_switchboard,
      nullptr));
  if (lastCuptiStatus_ != CUPTI_SUCCESS) {
    VLOG(1)  << "Failed cuptiSubscribe, status: " << lastCuptiStatus_;
  }

  initSuccess_ = (lastCuptiStatus_ == CUPTI_SUCCESS);
#endif
}

CuptiCallbackApi::CallbackList* CuptiCallbackApi::CallbackTable::lookup(
    CUpti_CallbackDomain domain, CuptiCallBackID cbid) {
  size_t idx;

  switch (domain) {

    case CUPTI_CB_DOMAIN_RESOURCE:
      assert(cbid >= __RESOURCE_CB_DOMAIN_START);
      assert(cbid < __RESOURCE_CB_DOMAIN_END);
      idx = cbid - __RESOURCE_CB_DOMAIN_START;
      return &resource.at(idx);

    case CUPTI_CB_DOMAIN_RUNTIME_API:
      assert(cbid >= __RUNTIME_CB_DOMAIN_START);
      assert(cbid < __RUNTIME_CB_DOMAIN_END);
      idx = cbid - __RUNTIME_CB_DOMAIN_START;
      return &runtime.at(idx);

    default:
      LOG(WARNING) << " Unsupported callback domain : " << domain;
      return nullptr;
  }
}

bool CuptiCallbackApi::registerCallback(
    CUpti_CallbackDomain domain,
    CuptiCallBackID cbid,
    CuptiCallbackFn cbfn) {
  CallbackList* cblist = callbacks_.lookup(domain, cbid);

  if (!cblist) {
    LOG(WARNING) << "Could not register callback -- domain = " << domain
                 << " callback id = " << cbid;
    return false;
  }

  // avoid duplicates
  auto it = std::find(cblist->begin(), cblist->end(), cbfn);
  if (it != cblist->end()) {
    LOG(WARNING) << "Adding duplicate callback -- domain = " << domain
                 << " callback id = " << cbid;
    return true;
  }

  if (cblist->size() == MAX_CB_FNS_PER_CB) {
    LOG(WARNING) << "Already registered max callback -- domain = " << domain
                 << " callback id = " << cbid;
  }

  WriteLockGuard wl(callbackLock_);
  cblist->push_back(cbfn);
  return true;
}

bool CuptiCallbackApi::deleteCallback(
    CUpti_CallbackDomain domain,
    CuptiCallBackID cbid,
    CuptiCallbackFn cbfn) {
  CallbackList* cblist = callbacks_.lookup(domain, cbid);
  if (!cblist) {
    LOG(WARNING) << "Attempting to remove unsupported callback -- domain = " << domain
                 << " callback id = " << cbid;
    return false;
  }

  // Locks are not required here as
  //  https://en.cppreference.com/w/cpp/container/list/erase
  //  "References and iterators to the erased elements are invalidated.
  //   Other references and iterators are not affected."
  auto it = std::find(cblist->begin(), cblist->end(), cbfn);
  if (it == cblist->end()) {
    LOG(WARNING) << "Could not find callback to remove -- domain = " << domain
                 << " callback id = " << cbid;
    return false;
  }

  WriteLockGuard wl(callbackLock_);
  cblist->erase(it);
  return true;
}

bool CuptiCallbackApi::enableCallback(
    CUpti_CallbackDomain domain, CUpti_CallbackId cbid) {
#ifdef HAS_CUPTI
  if (initSuccess_) {
    lastCuptiStatus_ = CUPTI_CALL_NOWARN(
        cuptiEnableCallback(1, subscriber_, domain, cbid));
    enabledCallbacks_.insert({domain, cbid});
    return (lastCuptiStatus_ == CUPTI_SUCCESS);
  }
#endif
  return false;
}

bool CuptiCallbackApi::disableCallback(
    CUpti_CallbackDomain domain, CUpti_CallbackId cbid) {
#ifdef HAS_CUPTI
  enabledCallbacks_.erase({domain, cbid});
  if (initSuccess_) {
    lastCuptiStatus_ = CUPTI_CALL_NOWARN(
        cuptiEnableCallback(0, subscriber_, domain, cbid));
    return (lastCuptiStatus_ == CUPTI_SUCCESS);
  }
#endif
  return false;
}

bool CuptiCallbackApi::enableCallbackDomain(
    CUpti_CallbackDomain domain) {
#ifdef HAS_CUPTI
  if (initSuccess_) {
    lastCuptiStatus_ = CUPTI_CALL_NOWARN(
        cuptiEnableDomain(1, subscriber_, domain));
    enabledCallbacks_.insert({domain, MAX_CUPTI_CALLBACK_ID_ALL});
    return (lastCuptiStatus_ == CUPTI_SUCCESS);
  }
#endif
  return false;
}

bool CuptiCallbackApi::disableCallbackDomain(
    CUpti_CallbackDomain domain) {
#ifdef HAS_CUPTI
  enabledCallbacks_.erase({domain, MAX_CUPTI_CALLBACK_ID_ALL});
  if (initSuccess_) {
    lastCuptiStatus_ = CUPTI_CALL_NOWARN(
        cuptiEnableDomain(0, subscriber_, domain));
    return (lastCuptiStatus_ == CUPTI_SUCCESS);
  }
#endif
  return false;
}

bool CuptiCallbackApi::reenableCallbacks() {
#ifdef HAS_CUPTI
  if (initSuccess_) {
    for (auto& cbpair : enabledCallbacks_) {
      if ((uint32_t)cbpair.second == MAX_CUPTI_CALLBACK_ID_ALL) {
        lastCuptiStatus_ = CUPTI_CALL_NOWARN(
            cuptiEnableDomain(1, subscriber_, cbpair.first));
      } else {
        lastCuptiStatus_ = CUPTI_CALL_NOWARN(
            cuptiEnableCallback(1, subscriber_, cbpair.first, cbpair.second));
      }
    }
    return (lastCuptiStatus_ == CUPTI_SUCCESS);
  }
#endif
  return false;
}

} // namespace KINETO_NAMESPACE
