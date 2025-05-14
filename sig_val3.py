from ultralytics import YOLO

if __name__ == '__main__':
    # model_path = r"E:\yolodata\yolov8s-cls.pt"
    # model=YOLO(model_path)
    model = YOLO('/home/ultralytics250406/runs/classify/cls-ADown+DIMB+PMSFA-RepHGNetV2-data0/weights/best.pt')
    model.val(
        split='val',
        # 数据文件路径，例如 coco128.yaml，分类任务直接写数据集文件夹路径即可
        data=r"/home/gasf_classification",
        imgsz=224, # 图像大小
        device=0, # 运行设备
        batch=32, # 每批次的图像数量 （-1 表示自动批次）
    ) # 训练的周期数

