# Copyright 2020 Facebook, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Install and run the TensorBoard plugin for performance analysis.

   Usage: python3 install_and_run.py --envdir ENVDIR --logdir LOGDIR
                  [--port PORT] [--version 2.2]
"""

# Lint as: python3

import argparse
import os
import subprocess

NO_CHECK = '||True'


def run(*args):
  """Runs a shell command."""
  subprocess.run(' '.join(args), shell=True, check=True)


class VirtualEnv(object):
  """Creates and runs programs in a virtual environment."""

  def __init__(self, envdir):
    self.envdir = envdir
    run('virtualenv', '-p', 'python3', self.envdir)

  def run(self, program, *args):
    run(os.path.join(self.envdir, 'bin', program), *args)

  def cleanup(self):
    """Clean up all existing TB profiler related installation."""
    self.run('pip3', 'uninstall', '-q', '-y', 'tensorboard_plugin_profile',
             'tensorboard', 'tensorflow-estimator', 'tensorflow', 'tbp-nightly',
             'tb-nightly', 'tf-estimator-nightly', 'tf-nightly', NO_CHECK)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--envdir', help='Virtual environment', required=True)
  parser.add_argument('--logdir', help='TensorBoard logdir', required=True)
  parser.add_argument(
      '--port',
      help='TensorBoard port',
      type=str,
      required=False,
      default='6006')
  parser.add_argument(
      '--version',
      help='TensorFlow profiler version, e.g. nightly, 2.2',
      required=False,
      default='nightly')
  args = parser.parse_args()
  venv = VirtualEnv(args.envdir)
  venv.cleanup()
  if args.version == 'nightly':
    venv.run('pip3', 'install', '-q', '-U', 'tf-nightly')
    venv.run('pip3', 'install', '-q', '-U', 'tb-nightly')
    venv.run('pip3', 'install', '-q', '-U', 'tbp-nightly')
  else:
    venv.run('pip3', 'install', '-q', '-U', 'tensorflow==' + args.version)
    venv.run('pip3', 'install', '-q', '-U', 'tensorboard==' + args.version)
    venv.run('pip3', 'install', '-q', '-U',
             'tensorboard_plugin_profile==' + args.version)

  tensorboard = os.path.join(args.envdir, 'bin/tensorboard')
  # There is a bug that in Mac OS the shebang of tensorboard script is not
  # correctly updated to the python3 of the virtual env. Directly invoke with
  # python inside the virtual env to walk around.
  venv.run('python3', tensorboard, '--logdir=' + args.logdir,
           '--port=' + args.port, '--bind_all')


if __name__ == '__main__':
  main()
