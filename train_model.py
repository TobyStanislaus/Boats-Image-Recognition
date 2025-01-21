from ultralytics import YOLO
if __name__ == '__main__':
    model = YOLO("yolov8n.pt")

    results = model.train(data="config.yaml", epochs=100, batch=16)
