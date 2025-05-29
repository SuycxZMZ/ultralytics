from ultralytics import SAM
import torch
import cv2
import numpy as np

# 检查是否有 GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 初始化模型
model = SAM("sam2.1_b.pt").to(device)

# 单张图片推理保存[7,8](@ref)
results = model("/Users/yuansu/Code/ultralytics/test-videos/C0015.MP4", 
                bboxes=[[935.892713359717, 303.2470150842958, 2092.8360645529124, 2160], 
                                        [2589.1790454660427, 106.85231040643858, 3788.9721504071344, 1760.1386425127644]],
                save=True)
