// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cupti.h>

namespace libkineto {

const char* memoryKindString(CUpti_ActivityMemoryKind kind);
const char* memcpyKindString(CUpti_ActivityMemcpyKind kind);
const char* runtimeCbidName(CUpti_CallbackId cbid);
const char* overheadKindString(CUpti_ActivityOverheadKind kind);

} // namespace libkineto
