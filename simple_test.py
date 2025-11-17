#!/usr/bin/env python3
"""
Simple test untuk deteksi furniture dan posisi
"""

from ultralytics import YOLO
import cv2
import numpy as np
import os
import warnings

warnings.filterwarnings('ignore')

# Load models
print("Loading models...")
pose_model = YOLO("yolov8n-pose.pt")
seg_model_path = os.path.join("datasets", "coco8seg_test", "weights", "best.pt")
seg_model = YOLO(seg_model_path)

print("Models loaded")
print("\nStarting camera feed. Press 'q' to exit.\n")

# Buka webcam
cap = cv2.VideoCapture(0)
frame_count = 0

# Kelas furniture yang kita cari
BED_LIKE_CLASSES = {
    56: 'chair',
    57: 'couch', 
    59: 'bed',
    60: 'dining_table'
}

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    
    # Pose detection
    pose_results = pose_model(frame, verbose=False)
    pose_frame = pose_results[0].plot()
    
    # Segmentation detection
    seg_results = seg_model(frame, conf=0.4, verbose=False)
    
    # Draw segmentation results
    if seg_results[0].masks is not None and len(seg_results[0].boxes) > 0:
        for idx, (box, mask) in enumerate(zip(seg_results[0].boxes, seg_results[0].masks.data)):
            class_id = int(box.cls[0])
            conf = float(box.conf[0])
            
            if class_id in BED_LIKE_CLASSES:
                # Draw bounding box
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_name = BED_LIKE_CLASSES[class_id]
                
                # Green box untuk furniture
                cv2.rectangle(pose_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(pose_frame, f"{class_name} ({conf:.2f})", (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Cek pose
    if len(pose_results[0].keypoints) > 0:
        person = pose_results[0].keypoints.xy[0]
        y_points = person[:, 1]
        
        head_y = y_points[0]
        foot_y = y_points[-1]
        frame_height = frame.shape[0]
        ratio = (foot_y - head_y) / frame_height
        
        if ratio > 0.45:
            pose_text = "Berdiri"
        elif 0.25 <= ratio <= 0.45:
            pose_text = "Duduk"
        else:
            pose_text = "Jatuh/Tidur"
        
        # Cek jika ada furniture di bawah orang
        person_center_x = np.mean(person[:, 0])
        person_center_y = np.mean(person[:, 1])
        
        furniture_detected = False
        for idx, box in enumerate(seg_results[0].boxes):
            class_id = int(box.cls[0])
            if class_id in BED_LIKE_CLASSES:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                if x1 <= person_center_x <= x2 and y1 <= person_center_y <= y2:
                    furniture_detected = True
                    furniture_name = BED_LIKE_CLASSES[class_id]
                    break
        
        # Display pose
        cv2.putText(pose_frame, f"Pose: {pose_text}", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        if furniture_detected:
            cv2.putText(pose_frame, f"On: {furniture_name}", (20, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # If laying down on furniture = tidur
            if pose_text == "Jatuh/Tidur":
                cv2.putText(pose_frame, "STATUS: TIDUR", (20, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        else:
            cv2.putText(pose_frame, "On: Lantai/Lainnya", (20, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            if pose_text == "Jatuh/Tidur":
                cv2.putText(pose_frame, "STATUS: JATUH!", (20, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv2.imshow("Fall Detection Test", pose_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Test selesai!")
