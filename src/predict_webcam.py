import cv2
from ultralytics import YOLO

model = YOLO("models/trained/best.pt")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Could not open camera.")
    exit()

print("🎥 Camera started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame, classes=[0], conf=0.5, imgsz=320)
    annotated = results[0].plot()
    cv2.imshow("Live Person Detection", annotated)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()