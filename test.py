from ultralytics import YOLO

model = YOLO('yolov8m.pt')
CLASS_NAMES = model.model.names
print(CLASS_NAMES)
