#!/bin/bash

# Install Python packages using pip with Tsinghua mirror
pip install timm==1.0.7 thop efficientnet_pytorch==0.7.1 einops grad-cam==1.5.4 dill==0.3.8 albumentations==1.4.11 pytorch_wavelets==1.3.0 tidecv PyWavelets opencv-python causal-conv1d>=1.2.0 mamba-ssm -i https://pypi.tuna.tsinghua.edu.cn/simple

# Install and upgrade openmim
pip install -U openmim -i https://pypi.tuna.tsinghua.edu.cn/simple

# Install mmengine and mmcv using mim
mim install mmengine -i https://pypi.tuna.tsinghua.edu.cn/simple
mim install "mmcv>=2.0.0" -i https://pypi.tuna.tsinghua.edu.cn/simple