from flask import Flask, render_template, Response, jsonify
from ultralytics import YOLO
import cv2
import time
import os
import warnings
import shutil
import numpy as np

warnings.filterwarnings('ignore')

app = Flask(__name__)

# Load Pose Detection Model
try:
    if os.path.exists("models/yolov8n-pose.pt"):
        pose_model = YOLO("models/yolov8n-pose.pt")
    elif os.path.exists("yolov8n-pose.pt"):
        pose_model = YOLO("yolov8n-pose.pt")
    else:
        pose_model = YOLO("yolov8n-pose")
    print(" Pose Detection Model loaded")
except Exception as e:
    print(f" Error loading pose model: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Load Segmentation Model for detecting bed, chair, etc
try:
    seg_model_path = os.path.join("datasets", "coco8seg_test", "weights", "best.pt")
    if os.path.exists(seg_model_path):
        seg_model = YOLO(seg_model_path)
        print(" Segmentation Model (bed/chair detection) loaded")
    else:
        seg_model = None
        print(" Segmentation model not found, will use pose detection only")
except Exception as e:
    print(f" Error loading segmentation model: {e}")
    seg_model = None

# Setup audio file
alert_sound_source = os.path.join("assets", "Efek suara jatuh.mp3")
alert_sound_dest = os.path.join("static", "alert.mp3")
audio_active = False

# Copy audio file to static folder if it exists
if os.path.exists(alert_sound_source):
    os.makedirs("static", exist_ok=True)
    shutil.copy(alert_sound_source, alert_sound_dest)
    audio_active = True
    print(" Audio file setup berhasil")
else:
    print(" File audio tidak ditemukan di assets")

alert_playing = False
last_pose = None
pose_start_time = time.time()
fall_start_time = None
recover_start_time = None
FALL_CONFIRM = 0.5
RECOVER_CONFIRM = 0.5
fall_detected = False
sleep_detected = False
current_posture = "Berdiri"  # Track current posture: Berdiri, Duduk, Jatuh
current_furniture = ""  # Track furniture name if duduk/tidur
activity_duration = 0  # Track duration of current activity
warning_triggered = False  # Track if warning already triggered for long activity
last_frame = None  # store latest encoded JPEG frame for snapshot endpoint

def gen_frames():
    global alert_playing, last_pose, pose_start_time, fall_start_time, recover_start_time, fall_detected, sleep_detected, current_posture, current_furniture, activity_duration, warning_triggered, last_frame
    cap = cv2.VideoCapture(0)
    
    BED_LIKE_CLASSES = {
        56: 'chair',
        57: 'couch',
        59: 'bed',
        60: 'dining_table'
    }

    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            pose_results = pose_model(frame, verbose=False)
            annotated_frame = pose_results[0].plot()

            if len(pose_results[0].keypoints) > 0:
                person = pose_results[0].keypoints.xy[0].cpu().numpy()
                y_points = person[:, 1]

                head_y = y_points[0]
                foot_y = y_points[-1]
                frame_height = frame.shape[0]
                ratio = (foot_y - head_y) / frame_height

                if ratio > 0.45:
                    body_position = "Berdiri"
                elif 0.25 <= ratio <= 0.45:
                    body_position = "Duduk"
                else:
                    body_position = "Jatuh"

                on_furniture = False
                furniture_name = None
                person_center_x = np.mean(person[:, 0])
                person_center_y = np.mean(person[:, 1])

                if seg_model is not None:
                    try:
                        seg_results = seg_model(frame, conf=0.3, verbose=False)
                        
                        if len(seg_results[0].boxes) > 0:
                            for idx, box in enumerate(seg_results[0].boxes):
                                class_id = int(box.cls[0])
                                conf = float(box.conf[0])
                                
                                if class_id in BED_LIKE_CLASSES:
                                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                                    
                                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                    cv2.putText(annotated_frame, f"{BED_LIKE_CLASSES[class_id]} ({conf:.2f})", 
                                              (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                                    
                                    if x1 <= person_center_x <= x2 and y1 <= person_center_y <= y2:
                                        on_furniture = True
                                        furniture_name = BED_LIKE_CLASSES[class_id]
                                        break
                    except Exception as e:
                        pass

                if body_position == "Jatuh" and on_furniture:
                    current_pose = f"Tidur di {furniture_name}"
                    current_posture = "Tidur"
                    current_furniture = furniture_name
                    sleep_detected = True
                elif body_position == "Jatuh" and not on_furniture:
                    current_pose = "Jatuh"
                    current_posture = "Jatuh"
                    current_furniture = ""
                    sleep_detected = False
                elif body_position == "Duduk" and on_furniture:
                    current_pose = f"Duduk di {furniture_name}"
                    current_posture = "Duduk"
                    current_furniture = furniture_name
                    sleep_detected = False
                else:
                    current_pose = body_position
                    current_posture = body_position
                    current_furniture = ""
                    sleep_detected = False

                if current_pose == "Jatuh":
                    cv2.putText(annotated_frame, "JATUH!", (50, 100),
                              cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)

                    if last_pose != "Jatuh":
                        fall_start_time = time.time()
                        recover_start_time = None

                    if fall_start_time and (time.time() - fall_start_time) >= FALL_CONFIRM:
                        if not alert_playing:
                            alert_playing = True
                            fall_detected = True

                elif "Tidur" in current_pose:
                    cv2.putText(annotated_frame, "TIDUR", (50, 100),
                              cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 5)
                    
                    if alert_playing:
                        alert_playing = False
                        fall_detected = False
                    
                    sleep_detected = True
                    
                else:
                    if current_pose == last_pose:
                        duration = time.time() - pose_start_time
                        activity_duration = int(duration)
                        
                        if duration >= 10:
                            if "Duduk" in current_pose:
                                cv2.putText(annotated_frame, f"Sudah {int(duration)} detik duduk", (50, 150), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 3)
                            elif current_pose == "Berdiri":
                                cv2.putText(annotated_frame, f"Sudah {int(duration)} detik berdiri", (50, 150),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 3)
                    else:
                        pose_start_time = time.time()
                        activity_duration = 0
                        warning_triggered = False

                    if last_pose == "Jatuh" and current_pose != "Jatuh":
                        recover_start_time = time.time()

                    if recover_start_time and (time.time() - recover_start_time) >= RECOVER_CONFIRM:
                        if alert_playing:
                            alert_playing = False
                            fall_detected = False

                    cv2.putText(annotated_frame, f"{current_pose}", (50, 100),
                              cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5)

                last_pose = current_pose

            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame = buffer.tobytes()
            # update last_frame for snapshot endpoint (thread-safe enough for this simple app)
            try:
                last_frame = frame
            except Exception:
                pass
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html', audio_active=audio_active)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/fall_status')
def fall_status():
    return jsonify({
        'fall_detected': fall_detected, 
        'alert_playing': alert_playing,
        'sleep_detected': sleep_detected,
        'current_posture': current_posture,
        'current_furniture': current_furniture,
        'activity_duration': activity_duration
    })


@app.route('/snapshot')
def snapshot():
    """Return the latest encoded JPEG frame captured by the streaming loop.
    This avoids opening a second VideoCapture which can freeze the camera.
    """
    global last_frame
    if last_frame is None:
        return ("No frame available yet", 404)
    return Response(last_frame, mimetype='image/jpeg')

if __name__ == '__main__':
    print("Sistem deteksi jatuh dimulai")
    print("Akses: http://localhost:9000")
    app.run(host='0.0.0.0', port=9000, debug=False)
