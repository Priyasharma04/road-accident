from ultralytics import YOLO

model = YOLO("weights/yolo11m.pt")

def detect(frame):
    return model(frame)