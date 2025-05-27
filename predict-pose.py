from ultralytics import YOLO
 
# 加载预训练的YOLOv11n模型
model = YOLO(r"yolo11x-pose.pt")
 
# 对'bus.jpg'图像进行推理，并获取结果
results = model.predict(r"/Users/yuansu/Desktop/codes/ultralytics/000000001/VID_20250520_174927.mp4", save=True, imgsz=640, conf=0.5, )
 
# # 处理返回的结果
# for result in results:
#     keypoints = result.keypoints       # 获取关键点估计信息
#     result.show()                      # 显示结果
 