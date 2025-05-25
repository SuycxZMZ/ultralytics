import warnings, os
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/CustomCfg/yolov8-PMSFA+GLFusion+PIoU.yaml')
    model.train(
                data='ultralytics/cfg/datasets/crowdpose-pose2.yaml',
                # data='coco.yaml',
                cache=True,
                imgsz=640,
                epochs=300,
                batch=8,
                # close_mosaic=0,
                workers=8, # Windows下出现莫名其妙卡主的情况可以尝试把workers设置为0
                optimizer='SGD', # using SGD
                device='0', 
                patience=0, # set 0 to close earlystop.
                # resume=True, # 断点续训,YOLO初始化时选择last.pt,例如YOLO('last.pt')
                # amp=False, # close amp
                # fraction=0.2,
                project='runs/train/crowdpose',
                name='exp',
                )