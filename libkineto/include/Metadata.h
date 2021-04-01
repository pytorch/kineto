#pragma once

#include <string>
#include <vector>
#include <memory>

namespace libkineto {

struct DistributedMetadata{
  std::string backend_;
  int rank_;
  int32_t worldSize_;
};

struct GpuInfo{
  int id_;
  std::string name_;
  uint64_t totalMemory_;
};

struct Metadata {
  std::vector<GpuInfo> gpus_;
  std::unique_ptr<DistributedMetadata> distributed_;
};

} // namespace libkineto