// Copyright (c) Meta Platforms, Inc. and affiliates.

// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// TODO(T90238193)
// @lint-ignore-every CLANGTIDY facebook-hte-RelativeInclude
#include "ActivityTrace.h"
#include "Logger.h"

namespace libkineto {

ActivityTrace::~ActivityTrace() {
  // Allow any loggers to finish saving (or not), before marking complete.
  // This allows us to track post process sample, and save any urls.
  if(pushToLog_) {
    UST_LOGGER_MARK_COMPLETED(kPostProcessingStage);
  }
}

const std::vector<const ITraceActivity*>* ActivityTrace::activities() {
  return memLogger_->traceActivities();
};

void ActivityTrace::save(const std::string& url) {
  std::string prefix;
  // if no protocol is specified, default to file
  if (url.find("://") == url.npos) {
    prefix = "file://";
  }
  memLogger_->log(*loggerFactory_.makeLogger(prefix + url));
};


} // namespace libkineto
