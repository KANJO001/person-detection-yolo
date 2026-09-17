import cv2
from ultralytics import YOLO

# Pretrained COCO model already detects 'person' (class 0)
model = YOLO("models/trained/best.pt")

def detect(image_path, output_path="outputs/predictions/result.jpg"):
    results = model(image_path, classes=[0], conf=0.5)
    annotated = results[0].plot()
    cv2.imwrite(output_path, annotated)
    print(f"Persons detected: {len(results[0].boxes)}")
    print(f"Saved: {output_path}")

if __name__ == "__main__":
    detect("data/raw/test.jpg")   # put any image with people here