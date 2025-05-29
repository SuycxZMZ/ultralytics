from ultralytics import FastSAM

# Define an inference source
source = "/Users/yuansu/Code/ultralytics/test-videos/C0015.MP4"

# Create a FastSAM model
model = FastSAM("FastSAM-s.pt").to('cpu')  # or FastSAM-x.pt

results = model(source, bboxes=[[935.892713359717, 303.2470150842958, 2092.8360645529124, 2160], [2589.1790454660427, 106.85231040643858, 3788.9721504071344, 1760.1386425127644]], save=True, conf=0.3)