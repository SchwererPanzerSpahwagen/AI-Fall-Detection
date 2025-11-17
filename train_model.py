from ultralytics import YOLO
import os

print("🎯 Memulai training model deteksi pose tidur...")
print("=" * 50)

# Path dataset
dataset_path = os.path.join("Datasets", "data.yaml")

if not os.path.exists(dataset_path):
    print(f"❌ File dataset tidak ditemukan: {dataset_path}")
    exit(1)

print(f"Dataset ditemukan: {dataset_path}")

# Load model YOLOv8 (gunakan model detection untuk klasifikasi pose)
model = YOLO("yolov8n.pt")

print("\n📚 Memulai training dengan dataset pose tidur...")
print("Kelas yang akan dideteksi:")
print("  1. Lateral (tidur miring)")
print("  2. Objects (ada benda)")
print("  3. Prone (tidur tengkurap)")
print("  4. Supine (tidur terlentang)")
print("\n")

# Training model
results = model.train(
    data=dataset_path,
    epochs=50,              # Jumlah epoch (iterasi training)
    imgsz=640,              # Ukuran gambar
    batch=16,               # Batch size
    patience=20,            # Early stopping
    device=0,               # GPU device (0 untuk GPU pertama, atau 'cpu' untuk CPU)
    save=True,              # Simpan model
    project="runs/detect",  # Folder untuk menyimpan hasil
    name="sleep_pose_model" # Nama proyek
)

print("\n" + "=" * 50)
print("Training selesai!")
print("Model tersimpan di: runs/detect/sleep_pose_model/weights/best.pt")
print("\nUntuk menggunakan model yang sudah dilatih, update app.py dengan:")
print('    model = YOLO("runs/detect/sleep_pose_model/weights/best.pt")')
