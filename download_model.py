from ultralytics import YOLO
import os

print("Downloading YOLOv8n-pose model...")
try:
    model = YOLO("yolov8n-pose.pt")
    print("Model downloaded successfully!")
except Exception as e:
    print(f"❌ Error: {e}")
