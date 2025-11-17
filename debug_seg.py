#!/usr/bin/env python3
"""
Debug script untuk cek apakah model segmentasi bisa detect furniture
"""

from ultralytics import YOLO
import cv2
import os

print("=" * 70)
print("DEBUG: Segmentation Model Test")
print("=" * 70)

# Load model
seg_model_path = os.path.join("datasets", "coco8seg_test", "weights", "best.pt")
print(f"\nLoading model from: {seg_model_path}")
print(f"Model exists: {os.path.exists(seg_model_path)}")

seg_model = YOLO(seg_model_path)
print("Model loaded\n")

# Print available classes
print("Classes dalam model:")
for class_id, class_name in seg_model.names.items():
    print(f"  {class_id}: {class_name}")

print("\n" + "=" * 70)
print("Furniture classes yang dicari:")
print("  56 = chair")
print("  57 = couch")
print("  59 = bed")
print("  60 = dining_table")
print("=" * 70 + "\n")

# Open webcam
cap = cv2.VideoCapture(0)
frame_count = 0

print("Webcam opened. Point at furniture and press 'q' to quit.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error reading frame")
        break
    
    frame_count += 1
    
    # Run inference
    results = seg_model(frame, conf=0.3, verbose=False)
    
    # Print detections every frame
    if len(results[0].boxes) > 0:
        print(f"[Frame {frame_count}] Detections: {len(results[0].boxes)}")
        for idx, box in enumerate(results[0].boxes):
            class_id = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = seg_model.names[class_id]
            print(f"  [{idx}] {class_name} (id:{class_id}, conf:{conf:.3f})")
            
            # Highlight furniture classes
            if class_id in [56, 57, 59, 60]:
                print(f"       ⭐ FURNITURE DETECTED!")
    
    # Display
    annotated = results[0].plot()
    cv2.imshow("Segmentation Debug", annotated)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("\nDebug selesai!")
