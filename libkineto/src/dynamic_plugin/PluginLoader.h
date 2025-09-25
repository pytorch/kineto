#pragma once

#include <dlfcn.h>
#include <filesystem>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "Logger.h"
#include "DynamicPluginInterface.h"
#include "PluginProfiler.h"

namespace libkineto {

class PluginRegistry {
public:
  static PluginRegistry &instance() {
    static PluginRegistry instance;
    return instance;
  }

  int registerPluginProfiler(const KinetoPlugin_ProfilerInterface *pProfiler) {
    if (pProfiler == nullptr) {
      LOG(ERROR) << "Failed to register plugin profiler of nullptr";

      return -1;
    }

    // Store in raw registry
    rawPluginProfilers_.push_back(*pProfiler);

    // Pass to internal registry
    const auto &profiler = rawPluginProfilers_.back();
    libkineto::api().registerProfilerFactory(
        [profiler]() -> std::unique_ptr<IActivityProfiler> {
          return std::make_unique<PluginProfiler>(profiler);
        });

    return 0;
  }

  const KinetoPlugin_Registry toCRegistry() {
    return KinetoPlugin_Registry{
        .unpaddedStructSize = KINETO_PLUGIN_REGISTRY_UNPADDED_STRUCT_SIZE,
        .pRegistryHandle =
            reinterpret_cast<KinetoPlugin_RegistryHandle *>(this),
        .registerProfiler = cRegisterProfiler};
  }

private:
  PluginRegistry() = default;
  ~PluginRegistry() = default;
  PluginRegistry(const PluginRegistry &) = delete;
  PluginRegistry &operator=(const PluginRegistry &) = delete;

  static int
  cRegisterProfiler(KinetoPlugin_RegistryHandle *pRegistryHandle,
                    const KinetoPlugin_ProfilerInterface *pProfiler) {
    auto pPluginRegistry = reinterpret_cast<PluginRegistry *>(pRegistryHandle);
    return pPluginRegistry->registerPluginProfiler(pProfiler);
  }

  std::vector<KinetoPlugin_ProfilerInterface> rawPluginProfilers_;
};

inline void loadPlugins() {
  const char *pPluginLibDirPathEnvVar = KINETO_PLUGIN_LIB_DIR_PATH_ENV_VARIABLE;

  const char *pPluginLibDirPath = getenv(pPluginLibDirPathEnvVar);

  if (pPluginLibDirPath == nullptr) {
    LOG(VERBOSE) << "Environment variable " << pPluginLibDirPathEnvVar
                 << " not set";

    return;
  }

  if (unsetenv(pPluginLibDirPathEnvVar) == -1) {
    LOG(ERROR) << "Failed to unset environment variable "
               << pPluginLibDirPathEnvVar << " at unsetenv() with error "
               << strerror(errno);

    return;
  }

  std::vector<std::string> libFilePaths;
  try {
    for (const auto &entry :
         std::filesystem::directory_iterator(pPluginLibDirPath)) {
      if (entry.is_regular_file() && entry.path().extension() == ".so") {
        libFilePaths.push_back(entry.path().string());
      }
    }
  } catch (const std::filesystem::filesystem_error &e) {
    LOG(ERROR) << "Error: " << e.what();

    return;
  }

  PluginRegistry &pluginRegistry = PluginRegistry::instance();
  KinetoPlugin_Registry cPluginRegistry = pluginRegistry.toCRegistry();

  for (const auto &libFilePath : libFilePaths) {
    // Clear error state
    dlerror();

    void *pHandle = dlopen(libFilePath.c_str(), RTLD_LAZY);
    if (pHandle == nullptr) {
      char *pError = dlerror();
      LOG(WARNING) << "Failed to open " << libFilePath
                   << " at dlopen() with error " << pError;
      continue;
    }

    int (*pfxRegister)(const KinetoPlugin_Registry *pRegistry) =
        reinterpret_cast<int (*)(const KinetoPlugin_Registry *pRegistry)>(
            dlsym(pHandle, "KinetoPlugin_register"));

    if (pfxRegister == nullptr) {
      char *pError = dlerror();
      LOG(VERBOSE) << "Failed to find symbol KinetoPlugin_register() from "
                   << libFilePath << " at dlsym() with error " << pError;

      continue;
    }

    LOG(INFO) << "Found symbol KinetoPlugin_register() from " << libFilePath;

    int errorCode = pfxRegister(&cPluginRegistry);
    if (errorCode != 0) {
      LOG(ERROR) << "Failed to register plugin profiler from " << libFilePath
                 << " at pfxRegister() with error " << errorCode;
    }
  }

  return;
}

} // namespace libkineto
