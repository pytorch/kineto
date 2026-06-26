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

namespace libkineto::RocmMetadataFields {
inline constexpr MetadataField<std::vector<int64_t>> kBlock{"block"};
inline constexpr MetadataField<int64_t> kBytes{"bytes"};
inline constexpr MetadataField<int64_t> kCid{"cid"};
inline constexpr MetadataField<int64_t> kCorrelation{"correlation"};
inline constexpr MetadataField<int64_t> kDevice{"device"};
inline constexpr MetadataField<std::string> kDst{"dst"};
inline constexpr MetadataField<std::vector<int64_t>> kGrid{"grid"};
inline constexpr MetadataField<int64_t> kHsaQueue{"hsa_queue"};
inline constexpr MetadataField<std::string> kKernel{"kernel"};
inline constexpr MetadataField<std::string> kKind{"kind"};
inline constexpr MetadataField<double> kMemoryBandwidthGbps{"memory bandwidth (GB/s)"};
inline constexpr MetadataField<std::string> kPtr{"ptr"};
inline constexpr MetadataField<int64_t> kSharedMemory{"shared memory"};
inline constexpr MetadataField<std::string> kSrc{"src"};
inline constexpr MetadataField<int64_t> kStream{"stream"};
} // namespace libkineto::RocmMetadataFields
