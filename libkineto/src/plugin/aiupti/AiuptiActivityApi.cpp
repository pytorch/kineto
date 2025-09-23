#include "AiuptiActivityApi.h"

#include <assert.h>
#include <chrono>
#include <mutex>
#include <thread>

#include <cstdlib>
#include <string>

#include "Logger.h"

using namespace std::chrono;

namespace KINETO_NAMESPACE {

constexpr size_t kBufSize(32 * 1024 * 1024);

AiuptiActivityApi& AiuptiActivityApi::singleton() {
  static AiuptiActivityApi instance;
  return instance;
}

void AiuptiActivityApi::pushCorrelationID(int id, CorrelationFlowType type) {
#ifdef HAS_AIUPTI
  if (!singleton().externalCorrelationEnabled_) {
    return;
  }
  switch (type) {
    case Default:
      // TODO: implement AIUPTI PushExternalCorrelationId
      break;
    case User:
      // TODO: implement AIUPTI PushExternalCorrelationId
      break;
  }
#endif
}

void AiuptiActivityApi::popCorrelationID(CorrelationFlowType type) {
#ifdef HAS_AIUPTI
  if (!singleton().externalCorrelationEnabled_) {
    return;
  }
  switch (type) {
    case Default:
      // TODO: implement AIUPTI PopExternalCorrelationId
      break;
    case User:
      // TODO: implement AIUPTI PopExternalCorrelationId
      break;
  }
#endif
}

static bool
nextActivityRecord(uint8_t* buffer, size_t valid_size, Pti_Activity*& record) {
#ifdef HAS_AIUPTI
  AIUpti_ResultTypes status =
      aiuptiActivityGetNextRecord(buffer, valid_size, &record);
  if (status != AIUpti_ResultTypes::AIUPTI_SUCCESS) {
    record = nullptr;
  }
#endif

  return record != nullptr;
}

void AiuptiActivityApi::setMaxBufferSize(int size) {
  maxAiuBufferCount_ = 1 + size / kBufSize;
}

void AiuptiActivityApi::bufferRequestedTrampoline(
    uint8_t** buffer,
    size_t* size,
    size_t* maxNumRecords) {
  singleton().bufferRequested(buffer, size, maxNumRecords);
}

void AiuptiActivityApi::bufferRequested(
    uint8_t** buffer,
    size_t* size,
    size_t* maxNumRecords) {
  std::lock_guard<std::mutex> guard(mutex_);
  if (allocatedAiuTraceBuffers_.size() >= maxAiuBufferCount_) {
    stopCollection = true;
    LOG(WARNING) << "Exceeded max AIU buffer count ("
                 << allocatedAiuTraceBuffers_.size() << " > "
                 << maxAiuBufferCount_ << ") - terminating tracing";
  }

  auto buf = std::make_unique<AiuptiActivityBuffer>(kBufSize);
  *buffer = buf->data();
  *size = kBufSize;

  allocatedAiuTraceBuffers_.emplace_back(*buffer, std::move(buf));

  *maxNumRecords = 0;
}

std::unique_ptr<AiuptiActivityBufferDeque>
AiuptiActivityApi::activityBuffers() {
  {
    std::lock_guard<std::mutex> guard(mutex_);

    // Unlike other backends, aiuptiFlushAllActivities flushes all pending
    // requests and triggers bufferCompleted, which transfers
    // allocatedAiuTraceBuffers_ to readyAiuTraceBuffers_. Therefore, we check
    // the readyAiuTraceBuffers_ deque here.
    if (!readyAiuTraceBuffers_ || readyAiuTraceBuffers_->empty()) {
      return nullptr;
    }
  }

#ifdef HAS_AIUPTI
  time_point<system_clock> t1;
  AIUPTI_CALL(aiuptiFlushAllActivities());
#endif

  std::lock_guard<std::mutex> guard(mutex_);
  return std::move(readyAiuTraceBuffers_);
}

#ifdef HAS_AIUPTI
int AiuptiActivityApi::processActivitiesForBuffer(
    uint8_t* buf,
    size_t validSize,
    std::function<void(const Pti_Activity*)> handler) {
  int count = 0;
  if (buf && validSize) {
    Pti_Activity* record{nullptr};
    while (nextActivityRecord(buf, validSize, record)) {
      handler(record);
      ++count;
    }
  }
  return count;
}
#endif

const std::pair<int, int> AiuptiActivityApi::processActivities(
    AiuptiActivityBufferDeque& buffers,
    std::function<void(const Pti_Activity*)> handler) {
  std::pair<int, int> res{0, 0};
#ifdef HAS_AIUPTI
  for (auto& pair : buffers) {
    auto& buf = pair.second;
    res.first += processActivitiesForBuffer(buf->data(), buf->size(), handler);
    res.second += buf->size();
  }
#endif
  return res;
}

void AiuptiActivityApi::clearActivities() {
  // TODO(mamaral): verify
  // {
  //   std::lock_guard<std::mutex> guard(mutex_);
  //   if (allocatedAiuTraceBuffers_.empty()) {
  //     std::cout << "###DEBUG::KINETO::API::clearActivities::"
  //                  "allocatedAiuTraceBuffers_.empty() "
  //               << std::endl;
  //     return;
  //   }
  // }
#ifdef HAS_AIUPTI
  AIUPTI_CALL(aiuptiFlushAllActivities());
#endif
  // TODO(mamaral): verify
  // std::lock_guard<std::mutex> guard(mutex_);
  // readyAiuTraceBuffers_ = nullptr;
}

#ifdef HAS_AIUPTI
void AiuptiActivityApi::bufferCompletedTrampoline(
    uint8_t* buffer,
    size_t size,
    size_t validSize) {
  singleton().bufferCompleted(buffer, size, validSize);
}

void AiuptiActivityApi::bufferCompleted(
    uint8_t* buffer,
    size_t size,
    size_t validSize) {
  std::lock_guard<std::mutex> guard(mutex_);
  auto& it = allocatedAiuTraceBuffers_.front();

  if (it.first != buffer) {
    LOG(ERROR) << "bufferCompleted called with unknown buffer: "
               << (void*)buffer;
    return;
  }

  if (!readyAiuTraceBuffers_) {
    readyAiuTraceBuffers_ = std::make_unique<AiuptiActivityBufferDeque>();
  }
  it.second->setSize(validSize);
  (*readyAiuTraceBuffers_).emplace_back(it.first, std::move(it.second));
  allocatedAiuTraceBuffers_.pop_front();

  // report any records dropped from the queue; to avoid unnecessary aiupti
  // API calls, we make it report only in verbose mode (it might not happen
  // often)
  if (VLOG_IS_ON(1)) {
    size_t dropped = 0;
    AIUPTI_CALL(aiuptiActivityGetNumDroppedRecords(&dropped));
    if (dropped != 0)
      LOG(WARNING) << "Dropped " << dropped << " activity records";
  }
}
#endif

void AiuptiActivityApi::enableAiuptiActivities(
    const std::set<ActivityType>& selected_activities) {
#ifdef HAS_AIUPTI
  AIUPTI_CALL(aiuptiActivityRegisterCallbacks(
      bufferRequestedTrampoline, bufferCompletedTrampoline));
  bool activityEnabled = false;
  externalCorrelationEnabled_ = false;
  for (const auto& activity : selected_activities) {
    if (activity == ActivityType::GPU_MEMCPY) {
      AIUPTI_CALL(aiuptiActivityEnable(AIUPTI_ACTIVITY_KIND_MEMCPY));
      AIUPTI_CALL(aiuptiActivityEnable(AIUPTI_ACTIVITY_KIND_MEMCPY2));
      AIUPTI_CALL(aiuptiActivityEnable(AIUPTI_ACTIVITY_KIND_SYNCHRONIZATION));
      activityEnabled = true;
    }
    if (activity == ActivityType::GPU_MEMSET) {
      // memset requires memory be also enabled
      AIUPTI_CALL(aiuptiActivityEnable(AIUPTI_ACTIVITY_KIND_MEMORY));
      activityEnabled = true;
    }
    if (activity == ActivityType::CONCURRENT_KERNEL) {
      AIUPTI_CALL(aiuptiActivityEnable(AIUPTI_ACTIVITY_KIND_CMPT));
      activityEnabled = true;
    }
    if (activity == ActivityType::PRIVATEUSE1_RUNTIME) {
      AIUPTI_CALL(aiuptiActivityEnable(AIUPTI_ACTIVITY_KIND_RUNTIME));
      activityEnabled = true;
    }
    if (activity == ActivityType::PRIVATEUSE1_DRIVER) {
      AIUPTI_CALL(aiuptiActivityEnable(AIUPTI_ACTIVITY_KIND_DRIVER));
      activityEnabled = true;
    }
  }

  // PyTorch version older than 2.6.0 does not have profile
  // ProfilerActivity.PrivateUse1 therefore we need to enable it via environment
  // variable.
  if (activityEnabled == false) {
    const char* env_value = std::getenv("ProfilerActivity");
    if (env_value != nullptr && std::string(env_value) == "PrivateUse1") {
      AIUPTI_CALL(aiuptiActivityEnable(AIUPTI_ACTIVITY_KIND_MEMCPY));
      AIUPTI_CALL(aiuptiActivityEnable(AIUPTI_ACTIVITY_KIND_MEMCPY2));
      AIUPTI_CALL(aiuptiActivityEnable(AIUPTI_ACTIVITY_KIND_SYNCHRONIZATION));
      AIUPTI_CALL(aiuptiActivityEnable(AIUPTI_ACTIVITY_KIND_MEMORY));
      // do not track memset events because they are the same as memory
      // allocation events
      AIUPTI_CALL(aiuptiActivityEnable(AIUPTI_ACTIVITY_KIND_CMPT));
      AIUPTI_CALL(aiuptiActivityEnable(AIUPTI_ACTIVITY_KIND_RUNTIME));
      AIUPTI_CALL(aiuptiActivityEnable(AIUPTI_ACTIVITY_KIND_DRIVER));
    }
  }

  tracingEnabled_ = 1;
#endif

  stopCollection = false;
}

void AiuptiActivityApi::disablePtiActivities(
    const std::set<ActivityType>& selected_activities) {
#ifdef HAS_AIUPTI
  bool activityEnabled = false;
  for (const auto& activity : selected_activities) {
    if (activity == ActivityType::GPU_MEMCPY) {
      AIUPTI_CALL(aiuptiActivityDisable(AIUPTI_ACTIVITY_KIND_MEMCPY));
      AIUPTI_CALL(aiuptiActivityDisable(AIUPTI_ACTIVITY_KIND_MEMCPY2));
      AIUPTI_CALL(aiuptiActivityDisable(AIUPTI_ACTIVITY_KIND_SYNCHRONIZATION));
      activityEnabled = true;
    }
    if (activity == ActivityType::GPU_MEMSET) {
      AIUPTI_CALL(aiuptiActivityDisable(AIUPTI_ACTIVITY_KIND_MEMORY));
      activityEnabled = true;
    }
    if (activity == ActivityType::CONCURRENT_KERNEL) {
      AIUPTI_CALL(aiuptiActivityDisable(AIUPTI_ACTIVITY_KIND_CMPT));
      activityEnabled = true;
    }
    if (activity == ActivityType::PRIVATEUSE1_RUNTIME) {
      AIUPTI_CALL(aiuptiActivityDisable(AIUPTI_ACTIVITY_KIND_RUNTIME));
      activityEnabled = true;
    }
    if (activity == ActivityType::PRIVATEUSE1_DRIVER) {
      AIUPTI_CALL(aiuptiActivityDisable(AIUPTI_ACTIVITY_KIND_DRIVER));
      activityEnabled = true;
    }
  }

  if (activityEnabled == false) {
    const char* env_value = std::getenv("ProfilerActivity");
    if (env_value != nullptr && std::string(env_value) == "PrivateUse1") {
      AIUPTI_CALL(aiuptiActivityDisable(AIUPTI_ACTIVITY_KIND_MEMCPY));
      AIUPTI_CALL(aiuptiActivityDisable(AIUPTI_ACTIVITY_KIND_MEMCPY2));
      AIUPTI_CALL(aiuptiActivityDisable(AIUPTI_ACTIVITY_KIND_SYNCHRONIZATION));
      AIUPTI_CALL(aiuptiActivityDisable(AIUPTI_ACTIVITY_KIND_MEMORY));
      AIUPTI_CALL(aiuptiActivityDisable(AIUPTI_ACTIVITY_KIND_CMPT));
      AIUPTI_CALL(aiuptiActivityDisable(AIUPTI_ACTIVITY_KIND_RUNTIME));
      AIUPTI_CALL(aiuptiActivityDisable(AIUPTI_ACTIVITY_KIND_DRIVER));
    }
  }
  externalCorrelationEnabled_ = false;
#endif
}

} // namespace KINETO_NAMESPACE
