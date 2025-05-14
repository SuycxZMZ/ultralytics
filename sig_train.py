from ultralytics import YOLO

if __name__ == '__main__':
    # model_path = r"E:\yolodata\yolov8s-cls.pt"
    # model=YOLO(model_path)
    model = YOLO('/home/ultralytics/ultralytics/cfg/models/mmx/yolov8-cls.yaml')
    model.train(
        # 分类任务直接写数据集文件夹路径即可
        data=r"/home/gasf_classification",
        imgsz=224, # 图像大小
        device=0, # 运行设备
        batch=64, # 每批次的图像数量 （-1 表示自动批次）
        epochs=10,
        cache=True,
        name='test-autodl',
        patience=0
    ) # 训练的周期数

