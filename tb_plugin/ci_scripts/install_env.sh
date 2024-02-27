#!/bin/bash

set -ex

# install cuda
#if [ "$CUDA_VERSION" = "cu101" ]; then
#    wget https://developer.download.nvidia.com/compute/cuda/10.1/Prod/local_installers/cuda_10.1.243_418.87.00_linux.run
#    sudo sh cuda_10.1.243_418.87.00_linux.run
#elif [ "$CUDA_VERSION" = "cu102" ]; then
#    wget https://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda_10.2.89_440.33.01_linux.run
#    sudo sh cuda_10.2.89_440.33.01_linux.run
#elif [ "$CUDA_VERSION" = "cu111" ]; then
#    wget https://developer.download.nvidia.com/compute/cuda/11.1.0/local_installers/cuda_11.1.0_455.23.05_linux.run
#    sudo sh cuda_11.1.0_455.23.05_linux.run
#elif [ "$CUDA_VERSION" = "cu112" ]; then
#    wget https://developer.download.nvidia.com/compute/cuda/11.2.0/local_installers/cuda_11.2.0_460.27.04_linux.run
#    sudo sh cuda_11.2.0_460.27.04_linux.run
#fi



# install pytorch
# bug fix: added mpmath=1.3.0 due to AttributeError: module 'mpmath' has no attribute 'rational'
pip install numpy tensorboard typing-extensions pillow pytest mpmath==1.3.0
if [ "$PYTORCH_VERSION" = "nightly" ]; then
    pip install --pre torch -f "https://download.pytorch.org/whl/nightly/$CUDA_VERSION/torch_nightly.html"
    pip install --pre torchvision --no-deps -f "https://download.pytorch.org/whl/nightly/$CUDA_VERSION/torch_nightly.html"
elif [ "$PYTORCH_VERSION" = "2.0" ]; then
    pip install --pre torch -f "https://download.pytorch.org/whl/test/$CUDA_VERSION/torch_test.html"
    pip install --pre torchvision --no-deps -f "https://download.pytorch.org/whl/test/$CUDA_VERSION/torch_test.html"
elif [ "$PYTORCH_VERSION" = "stable" ]; then
    pip install torch torchvision
fi

python -c "import torch; print(torch.__version__, torch.version.git_version); from torch.autograd import kineto_available; print(kineto_available())"
