// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "CuptiCallbackApi.h"

#include <assert.h>
#include <chrono>
#include <algorithm>
#include <mutex>
#include <shared_mutex>

#ifdef HAS_CUPTI
#include "cupti_call.h"
#endif
#include "Logger.h"


namespace KINETO_NAMESPACE {

// limit on number of handles per callback type
constexpr size_t MAX_CB_FNS_PER_CB = 8;

// Reader Writer lock types
using ReaderWriterLock = std::shared_timed_mutex;
using ReaderLockGuard = std::shared_lock<ReaderWriterLock>;
using WriteLockGuard = std::unique_lock<ReaderWriterLock>;

static ReaderWriterLock callbackLock_;

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
  CuptiCallbackApi::singleton().__callback_switchboard(
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

CuptiCallbackApi& CuptiCallbackApi::singleton() {
  static CuptiCallbackApi instance;
  return instance;
}

CuptiCallbackApi::CuptiCallbackApi() {
#ifdef HAS_CUPTI
  lastCuptiStatus_ = CUPTI_ERROR_UNKNOWN;
  lastCuptiStatus_ = CUPTI_CALL_NOWARN(
    cuptiSubscribe(&subscriber_,
      (CUpti_CallbackFunc)callback_switchboard,
      nullptr));

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
    return (lastCuptiStatus_ == CUPTI_SUCCESS);
  }
#endif
  return false;
}

bool CuptiCallbackApi::disableCallback(
    CUpti_CallbackDomain domain, CUpti_CallbackId cbid) {
#ifdef HAS_CUPTI
  if (initSuccess_) {
    lastCuptiStatus_ = CUPTI_CALL_NOWARN(
        cuptiEnableCallback(0, subscriber_, domain, cbid));
    return (lastCuptiStatus_ == CUPTI_SUCCESS);
  }
#endif
  return false;
}

} // namespace KINETO_NAMESPACE
