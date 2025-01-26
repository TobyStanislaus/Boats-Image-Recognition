from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("yolov8n.pt")

    # Train the model with early stopping
    results = model.train(
        data="config.yaml",      
        epochs=300,              
        batch=16,               
        patience=20)              
    