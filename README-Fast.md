# IndusFine-Detr 和 IndusRT-Det 说明

## Installation

### Prerequisites

- Python ≥ 3.8
- PyTorch ≥ 2.0 (compatible with Ultralytics 8.3.3+ base environment)
- CUDA ≥ 11.7 (for GPU acceleration)

### Environment Setup

```bash
# Install Python packages using pip with Tsinghua mirror
pip install timm==1.0.7 thop efficientnet_pytorch==0.7.1 einops grad-cam==1.5.4 dill==0.3.8 albumentations==1.4.11 pytorch_wavelets==1.3.0 tidecv PyWavelets opencv-python -i https://pypi.tuna.tsinghua.edu.cn/simple

# Install and upgrade openmim
pip install -U openmim -i https://pypi.tuna.tsinghua.edu.cn/simple

# Install mmengine and mmcv using mim
mim install mmengine -i https://pypi.tuna.tsinghua.edu.cn/simple
mim install "mmcv>=2.0.0" -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## Configuration Files && train val scripts 

配置文件在路径`ultralytics/cfg/models/CustomCfg` 下，可以根据自己的数据集和任务进行修改。

训练 Detr 时，不要指定`optimizer='SGD'`，按照默认的 Adam 即可。

如果想要优化模型，那么自己定义或者修改`ultralytics/cfg/models/CustomCfg`下的文件即可，我在`ultralytics/nn/modules/extra_blocks.py`定义了很多新的模块，方便自己定义模型。

所有的第三方注意力机制在`ultralytics/nn/modules/attention.py`中定义。

运行过程中少包了直接 `pip install xxx` 即可。

训练和验证文件为根目录下的`train.py`和`val.py`。指定好配置文件路径即可，最好是绝对路径。

使用小波变换下采样要关闭 amp 混合精度，否则可能会报错。

## 数据集

一般公开数据集直接提供的是`coco`格式的数据集，需要转换为`yolo`格式的数据集。

使用`GPT`直接写一个脚本进行数据集划分和转换，并生成对应的配置文件即可。数据集配置文件参考`ultralytics/cfg/datasets`下的官方文件配置文件。

论文中使用的 `NEU-DET`,`PKU-Market-PCB`,`Deep-PCB`在百度上非常好找，直接下载即可。或者直接去对应论文中找链接并下载。