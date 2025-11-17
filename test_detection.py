#!/usr/bin/env python3
"""
Debug script untuk test deteksi objek pada segmentasi model
"""

from ultralytics import YOLO
import cv2
import numpy as np
import os

# Load models
pose_model = YOLO("yolov8n-pose.pt")
seg_model_path = os.path.join("datasets", "coco8seg_test", "weights", "best.pt")
seg_model = YOLO(seg_model_path)

# Buka webcam
cap = cv2.VideoCapture(0)

print("=" * 60)
print("DEBUG: Object Detection Test")
print("=" * 60)
print("Kelas furniture yang dicari:")
print("  56 = chair (kursi)")
print("  57 = couch (sofa)")
print("  59 = bed (kasur)")
print("  60 = dining table (meja)")
print("=" * 60)

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    
    # Deteksi pose
    pose_results = pose_model(frame, verbose=False)
    
    # Deteksi segmentasi
    seg_results = seg_model(frame, conf=0.4, verbose=False)
    
    # Print info setiap 10 frame untuk mengurangi spam
    if frame_count % 10 == 0:
        print(f"\n[Frame {frame_count}]")
        
        # Tampilkan objek yang terdeteksi
        if len(seg_results[0].boxes) > 0:
            print(f"  Deteksi {len(seg_results[0].boxes)} objek:")
            for idx, box in enumerate(seg_results[0].boxes):
                class_id = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = seg_model.names[class_id]
                print(f"    [{idx}] {class_name} (class {class_id}, conf {conf:.2f})")
        else:
            print("  Tidak ada objek terdeteksi")
        
        # Tampilkan pose
        if len(pose_results[0].keypoints) > 0:
            print("  Orang terdeteksi di frame")
        else:
            print("  Tidak ada orang terdeteksi")
    
    # Display hasil
    annotated_frame = seg_results[0].plot()
    
    # Tambahkan pose visualization
    if len(pose_results[0].keypoints) > 0:
        annotated_frame = pose_results[0].plot()
    
    cv2.imshow("Detection Debug", annotated_frame)
    
    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("\nDebug selesai!")
