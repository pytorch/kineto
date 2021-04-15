#include "ThreadUtil.h"

#ifndef _MSC_VER
#include <pthread.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <sys/types.h>
#else
#include <locale>
#include <codecvt>
#include <windows.h>
#include <processthreadsapi.h>
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
#ifndef _MSC_VER
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
#elif defined _MSC_VER
    _sysTid = (int32_t)GetCurrentThreadId();
#else
    _sysTid = (int32_t)syscall(SYS_gettid);
#endif
  }
  return _sysTid;
}

int32_t threadId() {
  if (!_tid) {
#ifndef _MSC_VER
  pthread_t pth = pthread_self();
  int32_t* ptr = reinterpret_cast<int32_t*>(&pth);
  _tid = *ptr;
#else
  _tid = (int32_t)GetCurrentThreadId();
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
}

bool setThreadName(const std::string& name) {
#ifdef __APPLE__
  return 0 == pthread_setname_np(name.c_str());
#elif defined _MSC_VER
  std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> conv;
  std::wstring wname = conv.from_bytes(name);
  HRESULT hr = SetThreadDescription(GetCurrentThread(), wname.c_str());
  return SUCCEEDED(hr);
#else
  return 0 == pthread_setname_np(pthread_self(), name.c_str());
#endif
}

std::string getThreadName(int32_t tid) {
#ifdef __APPLE__
  char buf[kMaxThreadNameLength] = "";
  if (pthread_getname_np(pthread_self(), buf, kMaxThreadNameLength) != 0) {
    return "Unknown";
  }
  return buf;
#elif defined _MSC_VER
  PWSTR data;
  HRESULT hr = GetThreadDescription(GetCurrentThread(), &data);
  if (SUCCEEDED(hr)) {
    std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> conv;
    std::string name = conv.to_bytes(data);
    LocalFree(data);
    return name;
  } else {
    return "";
  }
#else
  char buf[kMaxThreadNameLength] = "Unknown";
  std::string filename = fmt::format("/proc/{}/task/{}/comm", getpid(), tid);
  FILE* comm_file = fopen(filename.c_str(), "r");
  if (comm_file) {
    size_t len = fread(buf, 1, kMaxThreadNameLength, comm_file);
    fclose(comm_file);
    // Remove newline
    if (len > 0) {
      buf[len - 1] = '\0';
    }
  } else {
    std::cerr << "Failed to open " << filename << std::endl;
  }
  return buf;
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

} // namespace libkineto
