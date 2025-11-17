from flask import Flask, render_template, Response
from ultralytics import YOLO
import cv2
import time
import os
import pygame

app = Flask(__name__)

# === Inisialisasi YOLO dan audio ===
try:
    model = YOLO("models/yolov8n-pose.pt")
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    exit(1)

alert_sound = os.path.join("static", "Efek suara jatuh.mp3")
if not os.path.exists(alert_sound):
    print(f"Warning: Audio file not found at {alert_sound}")
    print("Please make sure to place your audio file in the static folder")
    alert_sound = None

try:
    pygame.mixer.init()
    if alert_sound:
        pygame.mixer.music.load(alert_sound)
except Exception as e:
    print(f"Error initializing audio: {e}")
    alert_sound = None

alert_playing = False
last_pose = None
pose_start_time = time.time()
fall_start_time = None
recover_start_time = None
FALL_CONFIRM = 0.5
RECOVER_CONFIRM = 0.5

# === Fungsi untuk streaming kamera ===
def gen_frames():
    global alert_playing, last_pose, pose_start_time, fall_start_time, recover_start_time
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            results = model(frame)
            annotated_frame = results[0].plot()

            if len(results[0].keypoints) > 0:
                person = results[0].keypoints.xy[0]
                y_points = person[:, 1]

                head_y = y_points[0]
                foot_y = y_points[-1]
                frame_height = frame.shape[0]
                ratio = (foot_y - head_y) / frame_height

                if ratio > 0.45:
                    current_pose = "Berdiri"
                elif 0.25 <= ratio <= 0.45:
                    current_pose = "Duduk"
                else:
                    current_pose = "Jatuh"

                # === Logika jatuh ===
                if current_pose == "Jatuh":
                    cv2.putText(annotated_frame, "JATUH!", (50, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)

                    if last_pose != "Jatuh":
                        fall_start_time = time.time()
                        recover_start_time = None

                    if fall_start_time and (time.time() - fall_start_time) >= FALL_CONFIRM:
                        if not alert_playing and alert_sound:
                            try:
                                pygame.mixer.music.play(loops=-1)
                                alert_playing = True
                            except Exception as e:
                                print(f"Error playing alert sound: {e}")

                else:
                    # === Perilaku normal ===
                    if current_pose == last_pose:
                        duration = time.time() - pose_start_time
                        if duration >= 10:
                            if current_pose == "Duduk":
                                cv2.putText(annotated_frame, "Sudah 10 detik duduk, harap bergerak!",
                                            (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 3)
                            elif current_pose == "Berdiri":
                                cv2.putText(annotated_frame, "Sudah 10 detik berdiri, harap duduk!",
                                            (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 3)
                    else:
                        pose_start_time = time.time()

                    if last_pose == "Jatuh" and current_pose != "Jatuh":
                        recover_start_time = time.time()

                    if recover_start_time and (time.time() - recover_start_time) >= RECOVER_CONFIRM:
                        if alert_playing:
                            pygame.mixer.music.stop()
                            alert_playing = False

                    cv2.putText(annotated_frame, f"{current_pose}", (50, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5)

                last_pose = current_pose

            # Konversi frame ke format web (JPEG)
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

# === Route utama ===
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9000, debug=False)
