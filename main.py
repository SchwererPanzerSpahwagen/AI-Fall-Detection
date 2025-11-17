from ultralytics import YOLO
import cv2
import time
import os
import pygame
import numpy as np
import warnings

warnings.filterwarnings('ignore')

# Load model YOLO pose
pose_model = YOLO("yolov8n-pose.pt")

# Load segmentation model untuk deteksi furniture
seg_model_path = os.path.join("datasets", "coco8seg_test", "weights", "best.pt")
if os.path.exists(seg_model_path):
    seg_model = YOLO(seg_model_path)
    print("Segmentation Model (furniture detection) loaded")
else:
    seg_model = None
    print("Peringatan: Segmentation model not found")

# Buka kamera (0 = default webcam)
cap = cv2.VideoCapture(0)

print("🔍 Sistem AI Fall Detection with Furniture Detection aktif. Tekan Q untuk berhenti.")
print("=" * 70)

# Variabel pelacakan posisi dan waktu
last_pose = None
pose_start_time = time.time()
# Audio / alert state
alert_playing = False
music_loaded = False
FALL_CONFIRM = 0.5     # detik untuk konfirmasi jatuh sebelum suara menyala
RECOVER_CONFIRM = 0.5  # detik untuk konfirmasi recovery sebelum suara mati
fall_start_time = None
recover_start_time = None

# Path file audio peringatan
alert_sound = os.path.join("assets", "Efek suara jatuh.mp3")

# Inisialisasi pygame mixer untuk kontrol play/stop
try:
    pygame.mixer.init()
    if os.path.exists(alert_sound):
        try:
            pygame.mixer.music.load(alert_sound)
            music_loaded = True
        except Exception as e:
            print(f"[!] Gagal memuat file audio: {e}")
    else:
        print(f"[!] File audio tidak ditemukan: {alert_sound}")
except Exception as e:
    print(f"[!] Gagal inisialisasi audio: {e}")

# Kelas furniture yang dicari
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

    # Deteksi pose
    pose_results = pose_model(frame, verbose=False)
    annotated_frame = pose_results[0].plot()

    # Jika ada orang terdeteksi
    if len(pose_results[0].keypoints) > 0:
        person = pose_results[0].keypoints.xy[0].cpu().numpy()  # Convert to numpy
        y_points = person[:, 1]

        head_y = y_points[0]
        foot_y = y_points[-1]
        frame_height = frame.shape[0]
        ratio = (foot_y - head_y) / frame_height

        # Tentukan kondisi pose dasar
        if ratio > 0.45:
            body_position = "Berdiri"
        elif 0.25 <= ratio <= 0.45:
            body_position = "Duduk"
        else:
            body_position = "Jatuh"

        # Cek apakah ada furniture di bawah orang
        on_furniture = False
        furniture_name = None
        person_center_x = np.mean(person[:, 0])
        person_center_y = np.mean(person[:, 1])

        # Deteksi segmentasi untuk furniture
        if seg_model is not None:
            try:
                seg_results = seg_model(frame, conf=0.3, verbose=False)
                
                if len(seg_results[0].boxes) > 0:
                    for idx, box in enumerate(seg_results[0].boxes):
                        class_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        
                        if class_id in BED_LIKE_CLASSES:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            
                            # Gambar bounding box furniture
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(annotated_frame, f"{BED_LIKE_CLASSES[class_id]} ({conf:.2f})", 
                                      (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            
                            # Check jika orang berada di atas furniture
                            if x1 <= person_center_x <= x2 and y1 <= person_center_y <= y2:
                                on_furniture = True
                                furniture_name = BED_LIKE_CLASSES[class_id]
                                print(f"Orang di atas: {furniture_name}")
                                break
            except Exception as e:
                print(f"Peringatan: Error segmentasi: {e}")

        # Logika deteksi status
        if body_position == "Jatuh" and on_furniture:
            current_pose = f"Tidur di {furniture_name}"
        elif body_position == "Jatuh" and not on_furniture:
            current_pose = "Jatuh"
        elif body_position == "Duduk" and on_furniture:
            current_pose = f"Duduk di {furniture_name}"
        else:
            current_pose = body_position

        # --- Kondisi JATUH (dengan audio yang menyala selama jatuh) ---
        if current_pose == "Jatuh":
            cv2.putText(annotated_frame, "JATUH!", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)

            # Mulai timer konfirmasi jatuh ketika transisi ke state Jatuh
            if last_pose != "Jatuh":
                fall_start_time = time.time()
                recover_start_time = None

            # Jika jatuh terkonfirmasi selama lebih dari FALL_CONFIRM, nyalakan audio
            if fall_start_time and (time.time() - fall_start_time) >= FALL_CONFIRM:
                if not alert_playing and music_loaded:
                    try:
                        # main loop audio (ulang terus) agar terdengar selama masih jatuh
                        pygame.mixer.music.play(loops=-1)
                        alert_playing = True
                        print("JATUH TERDETEKSI - Audio ON")
                    except Exception as e:
                        print(f"[!] Gagal memutar audio: {e}")

            pose_start_time = time.time()  # reset timer

        elif "Tidur" in current_pose:
            # Jika tidur, jangan trigger alarm
            cv2.putText(annotated_frame, "TIDUR", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 5)
            
            # Stop alarm jika sedang aktif
            if alert_playing:
                try:
                    pygame.mixer.music.stop()
                    alert_playing = False
                    print("Tidur terdeteksi - Audio OFF")
                except Exception as e:
                    print(f"[!] Gagal stop audio: {e}")

        else:
            # --- Hitung durasi pose sama (untuk duduk/berdiri) ---
            if current_pose == last_pose:
                duration = time.time() - pose_start_time
                if duration >= 10:
                    if "Duduk" in current_pose:
                        cv2.putText(annotated_frame, f"Sudah {int(duration)} detik {current_pose}",
                                    (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 3)
                    elif "Berdiri" in current_pose:
                        cv2.putText(annotated_frame, f"Sudah {int(duration)} detik berdiri",
                                    (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 3)
            else:
                pose_start_time = time.time()

            # Jika sebelumnya Jatuh dan sekarang bukan, mulai timer pemulihan
            if last_pose == "Jatuh" and current_pose != "Jatuh":
                recover_start_time = time.time()

            # Jika recovery terkonfirmasi selama lebih dari RECOVER_CONFIRM, matikan audio
            if recover_start_time and (time.time() - recover_start_time) >= RECOVER_CONFIRM:
                if alert_playing:
                    try:
                        pygame.mixer.music.stop()
                    except Exception as e:
                        print(f"[!] Gagal menghentikan audio: {e}")
                    alert_playing = False

            # --- Tampilkan pose saat ini ---
            cv2.putText(annotated_frame, f"{current_pose}", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5)

        last_pose = current_pose

    cv2.imshow("AI Fall Detection", annotated_frame)

    # Tekan Q untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("🛑 Sistem berhenti.")
