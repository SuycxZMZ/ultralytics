from ultralytics import SAM
import cv2
import numpy as np

# 初始化模型
model = SAM("sam2.1_b.pt")

# # 单张图片推理保存[7,8](@ref)
# results = model("test-img.jpg")
# results[0].save("result.jpg")  # 保存带掩码的叠加图片
# results[0].show()  # 显示结果（可选）

# 视频参数配置
video_path = "test-video.mp4"
output_path = "test-video-out.mp4"

# 获取原始视频参数[6,7](@ref)
cap = cv2.VideoCapture(video_path)
original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# 配置无损MP4编码器[4,8](@ref)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用标准MP4编码
out = cv2.VideoWriter(
    output_path,
    fourcc,
    fps,
    (original_width, original_height),
    isColor=True
)

# 设置编码参数避免压缩[4,5](@ref)
out.set(cv2.VIDEOWRITER_PROP_QUALITY, 100)  # 最高质量
out.set(cv2.VIDEOWRITER_PROP_NSTRIPES, 2)  # 并行编码

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 执行分割推理[1](@ref)
    results = model(frame)

    # 获取带掩码的BGR格式帧[6,7](@ref)
    seg_frame = results[0].plot()  # 自动生成可视化结果
    seg_frame = cv2.cvtColor(seg_frame, cv2.COLOR_RGB2BGR)  # 转换颜色空间

    # 写入无损视频帧
    out.write(seg_frame)

# 释放资源[6,8](@ref)
cap.release()
out.release()