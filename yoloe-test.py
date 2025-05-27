import os
import cv2
import glob
import json
from tqdm import tqdm
from ultralytics import YOLO

# 配置
IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp"]
VIDEO_EXTS = [".mp4", ".avi", ".mov", ".mkv"]
OUTPUT_DIR = "output/yoloe"

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def is_image(file):
    return os.path.splitext(file)[1].lower() in IMAGE_EXTS

def is_video(file):
    return os.path.splitext(file)[1].lower() in VIDEO_EXTS

def save_json_results(result, output_json_path):
    data = []
    for box, cls, conf in zip(result.boxes.xyxy.cpu().numpy(),
                               result.boxes.cls.cpu().numpy(),
                               result.boxes.conf.cpu().numpy()):
        x1, y1, x2, y2 = box.tolist()
        class_name = result.names[int(cls)]
        data.append({
            "class": class_name,
            "conf": float(conf),
            "bbox": [x1, y1, x2, y2]
        })
    with open(output_json_path, "w") as f:
        json.dump(data, f, indent=2)

def process_image(model, image_path, output_path, save_json=False):
    if os.path.exists(output_path):
        return "[Skipped]"
    results = model.predict(image_path)
    results[0].save(filename=output_path)
    if save_json:
        json_path = output_path.replace(os.path.splitext(output_path)[1], ".json")
        save_json_results(results[0], json_path)
    return "[Image Processed]"

def process_video(model, video_path, output_path, save_json=False):
    if os.path.exists(output_path):
        return "[Skipped]"
    cap = cv2.VideoCapture(video_path)
    w, h = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    frame_idx = 0
    video_json = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model.predict(frame)
        vis = results[0].plot()
        vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
        out.write(vis)

        if save_json:
            for box, cls, conf in zip(results[0].boxes.xyxy.cpu().numpy(),
                                       results[0].boxes.cls.cpu().numpy(),
                                       results[0].boxes.conf.cpu().numpy()):
                x1, y1, x2, y2 = box.tolist()
                video_json.append({
                    "frame": frame_idx,
                    "class": results[0].names[int(cls)],
                    "conf": float(conf),
                    "bbox": [x1, y1, x2, y2]
                })
        frame_idx += 1

    cap.release()
    out.release()

    if save_json:
        json_path = output_path.replace(os.path.splitext(output_path)[1], ".json")
        with open(json_path, "w") as f:
            json.dump(video_json, f, indent=2)
    return "[Video Processed]"

def collect_files(input_path):
    files = []
    if os.path.isfile(input_path):
        files = [input_path]
    elif os.path.isdir(input_path):
        for ext in IMAGE_EXTS + VIDEO_EXTS:
            files.extend(glob.glob(os.path.join(input_path, f"*{ext}")))
    return files

def main():
    # ✅ 配置模型和参数
    model_path = "yoloe-11s-seg.pt"
    model = YOLO(model_path)
    custom_classes = ["person", ]
    model.set_classes(custom_classes, model.get_text_pe(custom_classes))

    input_path = "/Users/yuansu/Code/ultralytics/test-img.jpg"  # 替换为你的路径
    save_json = False  # ✅ 是否保存为 JSON 格式

    ensure_dir(OUTPUT_DIR)
    files = collect_files(input_path)

    for file in tqdm(files, desc="Processing files"):
        base = os.path.basename(file)
        name, ext = os.path.splitext(base)
        output_path = os.path.join(OUTPUT_DIR, f"{name}_out{ext}")

        if is_image(file):
            status = process_image(model, file, output_path, save_json)
        elif is_video(file):
            status = process_video(model, file, output_path, save_json)
        else:
            status = "[Unsupported Format]"
        # tqdm.write(f"{file} -> {output_path} {status}")

if __name__ == "__main__":
    main()