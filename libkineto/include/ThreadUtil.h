#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace libkineto {

int32_t systemThreadId();
int32_t threadId();
bool setThreadName(const std::string& name);
std::string getThreadName();

int32_t processId();
std::string processName(int32_t pid);

struct ProcessInfo {
  int32_t pid;
  const std::string name;
  const std::string label;
};

struct ThreadInfo {
  ThreadInfo(int32_t tid, const std::string& name) :
    tid(tid), name(name) {}
  int32_t tid;
  const std::string name;
};

// Return a list of pids and process names for the current process
// and its parents.
std::vector<std::pair<int32_t, std::string>> pidCommandPairsOfAncestors();

} // namespace libkineto
