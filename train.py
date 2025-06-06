import warnings, os
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r'/Users/yuansu/Code/ultralytics/ultralytics/cfg/models/CustomCfg/yolo11n.yaml')
    # model.load('yolov8n.pt') # loading pretrain weights
    model.train(data='coco128.yaml',
                cache=False,
                imgsz=640,
                epochs=10,
                batch=4,
                close_mosaic=0,
                workers=0, # Windows下出现莫名其妙卡主的情况可以尝试把workers设置为0, Linux服务器一般设8
                optimizer='SGD', # using SGD
                device='cpu',
                # patience=0, # set 0 to close earlystop.
                # resume=True, # 断点续训,YOLO初始化时选择last.pt,例如YOLO('last.pt')
                # amp=False, # close amp
                # fraction=0.2,
                project='runs/train',
                name='exp',
                )