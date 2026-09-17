# Person Detection with YOLOv8 + PyTorch

A custom-trained computer vision model that detects people in images and live webcam feeds. Built with **PyTorch** and **Ultralytics YOLOv8**, this project includes a full pipeline from data training to real-time inference.

##  Model Performance
The model was fine-tuned on the [Blurred Person Detection Dataset](https://www.kaggle.com/datasets/ahmedrdata/blurred-person-detection-dataset-yolov8). Despite the challenging low-quality images, it achieved excellent results:

| Metric | Score |
|---|---|
| **mAP@0.5** | 0.911 |
| **Precision** | 0.923 |
| **Recall** | 0.825 |
| **Inference Speed** | ~176ms per image (CPU) |

##  Features
- **Custom Trained Model:** Fine-tuned YOLOv8n on 455 blurred images for robust person detection.
- **Image Inference:** Detect people in static image files.
- **Live Webcam Inference:** Real-time person detection using your laptop's camera.
- **PyTorch Backend:** Built entirely on the PyTorch framework.

##  Tech Stack
- **Deep Learning:** PyTorch, Ultralytics YOLOv8
- **Computer Vision:** OpenCV
- **Language:** Python 3.12

##  Project Structure
```text
person-detection-yolo/
├── data/
│   └── data.yaml              # Dataset configuration
├── models/
│   ├── pretrained/            # Original YOLOv8 weights
│   └── trained/               # Your custom trained weights (best.pt)
├── src/
│   ├── train.py               # Script to train the model
│   ├── predict_image.py       # Detect people in an image
│   └── predict_webcam.py      # Live detection via webcam
├── requirements.txt           # Python dependencies
└── README.md
