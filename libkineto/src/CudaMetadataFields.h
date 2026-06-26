/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "TypedMetadata.h"

namespace libkineto::CudaMetadataFields {
inline constexpr MetadataField<int64_t> kActiveBlocksPerMultiprocessor{"activeBlocksPerMultiprocessor"};
inline constexpr MetadataField<int64_t> kAllocatedRegistersPerBlock{"allocatedRegistersPerBlock"};
inline constexpr MetadataField<int64_t> kAllocatedSharedMemPerBlock{"allocatedSharedMemPerBlock"};
inline constexpr MetadataField<std::vector<int64_t>> kBlock{"block"};
inline constexpr MetadataField<int64_t> kBlockLimitBarriers{"blockLimitBarriers"};
inline constexpr MetadataField<int64_t> kBlockLimitBlocks{"blockLimitBlocks"};
inline constexpr MetadataField<int64_t> kBlockLimitRegs{"blockLimitRegs"};
inline constexpr MetadataField<int64_t> kBlockLimitSharedMem{"blockLimitSharedMem"};
inline constexpr MetadataField<int64_t> kBlockLimitWarps{"blockLimitWarps"};
inline constexpr MetadataField<double> kBlocksPerSm{"blocks per SM"};
inline constexpr MetadataField<int64_t> kBytes{"bytes"};
inline constexpr MetadataField<int64_t> kCbid{"cbid"};
inline constexpr MetadataField<int64_t> kChannel{"channel"};
inline constexpr MetadataField<int64_t> kChannelType{"channel_type"};
inline constexpr MetadataField<int64_t> kContext{"context"};
inline constexpr MetadataField<int64_t> kCorrelation{"correlation"};
inline constexpr MetadataField<std::string> kCudaSyncKind{"cuda_sync_kind"};
inline constexpr MetadataField<int64_t> kDevice{"device"};
inline constexpr MetadataField<int64_t> kEstAchievedOccupancyPercent{"est. achieved occupancy %"};
inline constexpr MetadataField<int64_t> kEventId{"event_id"};
inline constexpr MetadataField<int64_t> kFromContext{"fromContext"};
inline constexpr MetadataField<int64_t> kFromDevice{"fromDevice"};
inline constexpr MetadataField<int64_t> kGraphId{"graph id"};
inline constexpr MetadataField<int64_t> kGraphNodeId{"graph node id"};
inline constexpr MetadataField<std::vector<int64_t>> kGrid{"grid"};
inline constexpr MetadataField<int64_t> kInContext{"inContext"};
inline constexpr MetadataField<int64_t> kInDevice{"inDevice"};
inline constexpr MetadataField<std::string> kLimitingFactors{"limitingFactors"};
inline constexpr MetadataField<double> kMemoryBandwidthGbps{"memory bandwidth (GB/s)"};
inline constexpr MetadataField<int64_t> kPriority{"priority"};
inline constexpr MetadataField<int64_t> kQueued{"queued"};
inline constexpr MetadataField<int64_t> kRegistersPerThread{"registers per thread"};
inline constexpr MetadataField<int64_t> kSharedMemory{"shared memory"};
inline constexpr MetadataField<int64_t> kStream{"stream"};
inline constexpr MetadataField<int64_t> kToContext{"toContext"};
inline constexpr MetadataField<int64_t> kToDevice{"toDevice"};
inline constexpr MetadataField<int64_t> kWaitOnCudaEventId{"wait_on_cuda_event_id"};
inline constexpr MetadataField<int64_t> kWaitOnCudaEventRecordCorrId{"wait_on_cuda_event_record_corr_id"};
inline constexpr MetadataField<int64_t> kWaitOnStream{"wait_on_stream"};
inline constexpr MetadataField<double> kWarpsPerSm{"warps per SM"};
} // namespace libkineto::CudaMetadataFields
