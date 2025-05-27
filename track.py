import cv2
from ultralytics import YOLO
import os
import numpy as np

# ✅ 参数设置
model_name = "yolo11m-seg.pt"  # 支持 det / seg
conf_thresh = 0.4              # 置信度过滤阈值
enable_chinese_label = False   # 是否启用中文类别映射

# ✅ 中文类别映射表（可自定义）
CHINESE_LABELS = {
    "person": "研究牲",
    "car": "汽车",
    "bicycle": "自行车",
}

# ✅ 允许处理的英文类别
ALLOWED_CLASSES = ['person', 'car', 'bicycle']
# ALLOWED_CLASSES = [
#     'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
#     'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
#     'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
#     'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
#     'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
#     'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
#     'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
#     'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
#     'hair drier', 'toothbrush'
# ]

# ✅ 加载模型
model = YOLO(model_name)

# ✅ 判断任务类型
if "seg" in model_name:
    task_suffix = "-seg"
else:
    task_suffix = "-det"

# ✅ 视频路径
video_path = "/Users/yuansu/Desktop/codes/ultralytics/000000001/VID_20250520_174927.mp4"
cap = cv2.VideoCapture(video_path)

# ✅ 获取视频属性
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# ✅ 输出路径
video_dir, video_name = os.path.split(video_path)
video_basename, video_ext = os.path.splitext(video_name)
output_video_path = os.path.join(video_dir, f"{video_basename}{task_suffix}{video_ext}")
out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

# ✅ 颜色生成器：确保每个 track_id 唯一颜色
def get_color(idx):
    np.random.seed(idx)
    return tuple(map(int, np.random.randint(0, 256, size=3)))

# ✅ 主循环
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    results = model.track(frame, persist=True, verbose=False)
    annotated_frame = frame.copy()
    r = results[0]
    boxes = r.boxes
    names = r.names

    if boxes is not None and boxes.id is not None:
        ids = boxes.id.cpu().numpy().astype(int)
        for i, (box, track_id, cls_id, conf) in enumerate(zip(boxes.xyxy, ids, boxes.cls, boxes.conf)):
            if conf < conf_thresh:
                continue

            class_id = int(cls_id)
            class_name = names[class_id]
            if class_name not in ALLOWED_CLASSES:
                continue

            label_text = CHINESE_LABELS.get(class_name, class_name) if enable_chinese_label else class_name
            x1, y1, x2, y2 = map(int, box)
            color = get_color(track_id)

            # ✅ 绘制边框
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 4)

            # ✅ 绘制标签
            label = f"ID:{track_id} {label_text} {conf:.2f}"
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)
            cv2.rectangle(annotated_frame, (x1, y1 - th - baseline - 5), (x1 + tw, y1), color, -1)
            cv2.putText(annotated_frame, label, (x1, y1 - baseline - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

    # ✅ 分割任务：绘制掩码和轮廓，颜色与 box 保持一致
    if r.masks is not None and task_suffix == "-seg":
        masks = r.masks.data.cpu().numpy()        # (N, H, W)
        ids = boxes.id.cpu().numpy().astype(int)  # 与上面保持一致
        confs = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)

        for i, mask in enumerate(masks):
            if confs[i] < conf_thresh:
                continue

            class_id = classes[i]
            class_name = names[class_id]
            if class_name not in ALLOWED_CLASSES:
                continue

            label_text = CHINESE_LABELS.get(class_name, class_name) if enable_chinese_label else class_name
            track_id = ids[i]
            color = get_color(track_id)

            # resize 和二值化
            mask_resized = cv2.resize(mask, (frame_width, frame_height), interpolation=cv2.INTER_NEAREST)
            binary_mask = (mask_resized > 0.5).astype(np.uint8)

            # ✅ 绘制轮廓
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(annotated_frame, contours, -1, color, 2)

            # ✅ 绘制半透明颜色填充
            color_mask = np.zeros_like(annotated_frame, dtype=np.uint8)
            for c in range(3):
                color_mask[:, :, c] = binary_mask * color[c]
            annotated_frame = cv2.addWeighted(annotated_frame, 1.0, color_mask, 0.4, 0)

    # ✅ 写入帧并显示
    out.write(annotated_frame)
    cv2.imshow("Tracking", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ✅ 释放资源
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Tracking video saved to: {output_video_path}")