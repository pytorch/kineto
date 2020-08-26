#!/bin/bash
# Copyright 2020 Facebook, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e
if [ -z "${RUNFILES}" ]; then
  RUNFILES="$(CDPATH= cd -- "$0.runfiles" && pwd)"
fi

if [ "$(uname)" = "Darwin" ]; then
  sedi="sed -i ''"
else
  sedi="sed -i"
fi

PLUGIN_RUNFILE_DIR="${RUNFILES}/org_xprof/plugin"
FRONTEND_RUNFILE_DIR="${RUNFILES}/org_xprof/frontend"

dest="/tmp/profile-pip"
mkdir -p "$dest"
cd "$dest"

# Copy all necessary files for setup.
cp "$PLUGIN_RUNFILE_DIR/README.rst" .

# Copy plugin python files.
cd ${PLUGIN_RUNFILE_DIR}
find . -name '*.py' | cpio -updL $dest
cd $dest
chmod -R 755 .
cp ${BUILD_WORKSPACE_DIRECTORY}/bazel-bin/plugin/tensorboard_plugin_profile/protobuf/*_pb2.py tensorboard_plugin_profile/protobuf/

find tensorboard_plugin_profile/protobuf -name \*.py -exec $sedi -e '
    s/^from plugin.tensorboard_plugin_profile/from tensorboard_plugin_profile/
  ' {} +

# Copy static files.
cd tensorboard_plugin_profile
mkdir -p static
cd static
cp "$PLUGIN_RUNFILE_DIR/tensorboard_plugin_profile/static/index.html" .
cp "$PLUGIN_RUNFILE_DIR/tensorboard_plugin_profile/static/index.js" .
cp "$PLUGIN_RUNFILE_DIR/tensorboard_plugin_profile/static/materialicons.woff2" .
cp "$PLUGIN_RUNFILE_DIR/trace_viewer/trace_viewer_index.html" .
cp "$PLUGIN_RUNFILE_DIR/trace_viewer/trace_viewer_index.js" .
cp -LR "$FRONTEND_RUNFILE_DIR/bundle.js" .
cp -LR "$FRONTEND_RUNFILE_DIR/styles.css" .
cp -LR "$FRONTEND_RUNFILE_DIR/zone.js" .
