from ultralytics import YOLO

def main():
    # Load the pretrained model
    model = YOLO("models/pretrained/yolov8n.pt")

    # Train the model
    model.train(
        data="data/data.yaml",
        epochs=20,
        imgsz=320,       # Smaller = faster on CPU
        batch=8,
        name="person_detector",
        project="outputs/runs",
        patience=5,
        save=True,
        plots=True
    )

if __name__ == "__main__":
    main()