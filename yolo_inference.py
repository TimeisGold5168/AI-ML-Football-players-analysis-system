from ultralytics import YOLO
import torch

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load YOLO model onto CUDA device
model = YOLO('models/last.pt').to(device)

results = model.predict('input_videos/08fd33_4.mp4',save = True)
print(results[0]) #first frame
print('======================')
for box in results[0].boxes:
    print(box)
