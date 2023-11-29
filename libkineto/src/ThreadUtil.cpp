/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "ThreadUtil.h"

#ifndef _WIN32
#include <pthread.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <sys/types.h>
#else // _WIN32
#include <locale>
#include <codecvt>
#define WIN32_LEAN_AND_MEAN
#define NOGDI
#include <windows.h>
#include <processthreadsapi.h>
#undef ERROR
#endif // _WIN32

#ifdef __ANDROID__
#include <sys/prctl.h>
#endif

#include <fmt/format.h>
#include <iostream>
#include <string>

namespace libkineto {

namespace {
thread_local int32_t _pid = 0;
thread_local int32_t _tid = 0;
thread_local int32_t _sysTid = 0;
}

int32_t processId() {
  if (!_pid) {
#ifndef _WIN32
    _pid = (int32_t)getpid();
#else
    _pid = (int32_t)GetCurrentProcessId();
#endif
  }
  return _pid;
}

int32_t systemThreadId() {
  if (!_sysTid) {
#ifdef __APPLE__
    _sysTid = (int32_t)syscall(SYS_thread_selfid);
#elif defined _WIN32
    _sysTid = (int32_t)GetCurrentThreadId();
#elif defined __FreeBSD__
    syscall(SYS_thr_self, &_sysTid);
#else
    _sysTid = (int32_t)syscall(SYS_gettid);
#endif
  }
  return _sysTid;
}

int32_t threadId() {
  if (!_tid) {
#ifdef __APPLE__
    uint64_t tid;
    pthread_threadid_np(nullptr, &tid);
    _tid = tid;
#elif defined _WIN32
  _tid = (int32_t)GetCurrentThreadId();
#else
  pthread_t pth = pthread_self();
  int32_t* ptr = reinterpret_cast<int32_t*>(&pth);
  _tid = *ptr;
#endif
  }
  return _tid;
}

namespace {
static constexpr size_t kMaxThreadNameLength = 16;

static constexpr const char* basename(const char* s, int off = 0) {
  return !s[off]
      ? s
      : s[off] == '/' ? basename(&s[off + 1]) : basename(s, off + 1);
}
#if defined(_WIN32)
void *getKernel32Func(const char* procName) {
  return reinterpret_cast<void*>(GetProcAddress(GetModuleHandleA("KERNEL32.DLL"), procName));
}
#endif
}

bool setThreadName(const std::string& name) {
#ifdef __APPLE__
  return 0 == pthread_setname_np(name.c_str());
#elif defined _WIN32
  // Per https://docs.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-setthreaddescription
  // Use runtime linking to set thread description
  static auto _SetThreadDescription = reinterpret_cast<decltype(&SetThreadDescription)>(getKernel32Func("SetThreadDescription"));
  if (!_SetThreadDescription) {
    return false;
  }
  std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> conv;
  std::wstring wname = conv.from_bytes(name);
  HRESULT hr = _SetThreadDescription(GetCurrentThread(), wname.c_str());
  return SUCCEEDED(hr);
#else
  return 0 == pthread_setname_np(pthread_self(), name.c_str());
#endif
}

std::string getThreadName() {
#ifndef _WIN32
  char buf[kMaxThreadNameLength] = "";
  if (
#ifndef __ANDROID__
    pthread_getname_np(pthread_self(), buf, kMaxThreadNameLength) != 0
#else
    prctl(PR_GET_NAME, buf, kMaxThreadNameLength) != 0
#endif
  ) {
    return "Unknown";
  }
  return buf;
#else // _WIN32
  static auto _GetThreadDescription = reinterpret_cast<decltype(&GetThreadDescription)>(getKernel32Func("GetThreadDescription"));
  if (!_GetThreadDescription) {
    return "Unknown";
  }
  PWSTR data;
  HRESULT hr = _GetThreadDescription(GetCurrentThread(), &data);
  if (!SUCCEEDED(hr)) {
    return "";
  }
  std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> conv;
  std::string name = conv.to_bytes(data);
  LocalFree(data);
  return name;
#endif
}

// Linux:
// Extract process name from /proc/pid/cmdline. This does not have
// the 16 character limit that /proc/pid/status and /prod/pid/comm has.
std::string processName(int32_t pid) {
#ifdef __linux__
  FILE* cmdfile = fopen(fmt::format("/proc/{}/cmdline", pid).c_str(), "r");
  if (cmdfile != nullptr) {
    char* command = nullptr;
    int scanned = fscanf(cmdfile, "%ms", &command);
    fclose(cmdfile);
    if (scanned > 0 && command) {
      std::string ret(basename(command));
      free(command);
      return ret;
    }
  }
  std::cerr << "Failed to read process name for pid " << pid << std::endl;
#endif
  return "";
}

// Max number of parent pids to collect, just for extra safeguarding.
constexpr int kMaxParentPids = 10;

// Return a pair of <parent_pid, command_of_current_pid>
static std::pair<int32_t, std::string> parentPidAndCommand(int32_t pid) {
#ifdef __linux__
  FILE* statfile = fopen(fmt::format("/proc/{}/stat", pid).c_str(), "r");
  if (statfile == nullptr) {
    return std::make_pair(0, "");
  }
  int32_t parent_pid;
  char* command = nullptr;
  int scanned = fscanf(statfile, "%*d (%m[^)]) %*c %d", &command, &parent_pid);
  fclose(statfile);
  std::pair<int32_t, std::string> ret;
  if (scanned == 2) {
    ret = std::make_pair(parent_pid, std::string(command));
  } else {
    std::cerr << "Failed to parse /proc/" << pid << "/stat" << std::endl;
    ret = std::make_pair(0, "");
  }

  // The 'm' character in the format tells fscanf to allocate memory
  // for the parsed string, which we need to free here.
  free(command);
  return ret;
#else
  return std::make_pair(0, "");
#endif
}

std::vector<std::pair<int32_t, std::string>> pidCommandPairsOfAncestors() {
  std::vector<std::pair<int32_t, std::string>> pairs;
  pairs.reserve(kMaxParentPids + 1);
  int32_t curr_pid = processId();
  // Usually we want to skip the root process (PID 1), but when running
  // inside a container the process itself has PID 1, so we need to include it
  for (int i = 0; i <= kMaxParentPids && (i == 0 || curr_pid > 1); i++) {
    std::pair<int32_t, std::string> ppid_and_comm = parentPidAndCommand(curr_pid);
    pairs.push_back(std::make_pair(curr_pid, ppid_and_comm.second));
    curr_pid = ppid_and_comm.first;
  }
  return pairs;
}

} // namespace libkineto
