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

// Catalog of typed metadata field keys.

namespace libkineto::GenericMetadataFields {
inline constexpr MetadataField<InputShapes> kInputDims{"Input Dims"};
inline constexpr MetadataField<InputShapes> kInputStrides{"Input Strides"};
inline constexpr MetadataField<std::vector<std::string>> kInputType{"Input type"};
inline constexpr MetadataField<int64_t> kSequenceNumber{"Sequence number"};
inline constexpr MetadataField<uint64_t> kFwdThreadId{"Fwd thread id"};
inline constexpr MetadataField<uint64_t> kRecordFunctionId{"Record function id"};
inline constexpr MetadataField<int64_t> kEvIdx{"Ev Idx"};
inline constexpr MetadataField<uint64_t> kAddr{"Addr"};
inline constexpr MetadataField<int64_t> kBytes{"Bytes"};
inline constexpr MetadataField<uint64_t> kTotalAllocated{"Total Allocated"};
inline constexpr MetadataField<uint64_t> kTotalReserved{"Total Reserved"};
inline constexpr MetadataField<int64_t> kDeviceType{"Device Type"};
inline constexpr MetadataField<int64_t> kDeviceId{"Device Id"};
inline constexpr MetadataField<uint64_t> kPythonId{"Python id"};
inline constexpr MetadataField<uint64_t> kPythonModuleId{"Python module id"};
inline constexpr MetadataField<uint64_t> kPythonThread{"Python thread"};
inline constexpr MetadataField<std::string> kBackend{"Backend"};
inline constexpr MetadataField<std::string> kCallFrom{"CallFrom"};
inline constexpr MetadataField<std::string> kCallStack{"Call stack"};
inline constexpr MetadataField<std::string> kModuleHierarchy{"Module Hierarchy"};
inline constexpr MetadataField<std::vector<std::string>> kConcreteInputs{"Concrete Inputs"};
} // namespace libkineto::GenericMetadataFields
