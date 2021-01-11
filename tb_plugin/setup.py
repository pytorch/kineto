# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# --------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import setuptools

setuptools.setup(
    name="tensorboard_plugin_torch_profiler",
    version="0.1.0",
    description="PyTorch Profiler TensorBoard Plugin",
    packages=setuptools.find_packages(),
    package_data={
        "tensorboard_plugin_torch_profiler": ["static/**"],
    },
    install_requires=[
        "tensorboard",
        "pandas"
    ],
    entry_points={
        "tensorboard_plugins": [
            "torch_profiler = tensorboard_plugin_torch_profiler.plugin:TorchProfilerPlugin",
        ],
    },
)
