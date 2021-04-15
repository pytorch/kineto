#pragma once

#include <cstdint>
#include <string>

namespace libkineto {

int32_t systemThreadId();
int32_t threadId();
bool setThreadName(const std::string& name);
std::string getThreadName(int32_t tid);

int32_t processId();
std::string processName(int32_t pid);

} // namespace libkineto
