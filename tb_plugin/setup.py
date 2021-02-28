# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# --------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import setuptools

def get_version():
    with open("version.txt", encoding="utf-8") as f:
        version = f.read().strip()

    if os.getenv('TORCH_TB_PROFILER_BUILD_VERSION'):
        version = os.getenv('TORCH_TB_PROFILER_BUILD_VERSION')
    return version

INSTALL_REQUIRED = [
    "pandas",
    "tensorboard >= 1.15, !=2.1.0"
]

TESTS_REQUIRED = INSTALL_REQUIRED + [
    "torch >= 1.8",
    "torchvision >= 0.8"
]

setuptools.setup(
    name="torch_tb_profiler",
    version=get_version(),
    description="PyTorch Profiler TensorBoard Plugin",
    long_description="PyTorch Profiler TensorBoard Plugin : \
        https://github.com/pytorch/kineto/tree/master/tb_plugin",
    url="https://github.com/pytorch/kineto/tree/master/tb_plugin",
    author="Pytorch Team",
    author_email="packages@pytorch.org",
    packages=setuptools.find_packages(),
    package_data={
        "torch_tb_profiler": ["static/**"],
    },
    entry_points={
        "tensorboard_plugins": [
            "torch_profiler = torch_tb_profiler.plugin:TorchProfilerPlugin",
        ],
    },
    python_requires=">= 2.7, != 3.0.*, != 3.1.*",
    install_requires=INSTALL_REQUIRED,
    tests_require=TESTS_REQUIRED,
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    license='BSD-3',
    keywords='pytorch tensorboard profile plugin',
)
