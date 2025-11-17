# 📋 DOKUMENTASI PERBAIKAN DETEKSI FURNITURE

## Masalah yang Sudah Diperbaiki

Sebelumnya, objek furniture (kursi, meja, kasur, sofa) tidak terdeteksi dengan baik. Sekarang sudah diperbaiki dengan sistem yang lebih robust.

## 🔧 Perubahan yang Dilakukan

### 1. **Update Logic di `gen_frames()`**
   - Model segmentasi sekarang dijalankan dengan confidence 0.4 untuk deteksi yang lebih sensitif
   - Inline deteksi furniture langsung dalam loop utama (lebih efisien)
   - Deteksi menggunakan bounding box furniture, bukan hanya mask

### 2. **Kelas Furniture yang Dideteksi**
```
   - 56 = chair (kursi)
   - 57 = couch (sofa/divani)
   - 59 = bed (kasur)
   - 60 = dining_table (meja makan)
```

### 3. **Logika Deteksi Gabungan**
```
Jika (ratio ≤ 0.25 AND pada_kasur) → "TIDUR" (tanpa alarm)
Jika (ratio ≤ 0.25 AND NOT pada_kasur) → "JATUH!" (dengan alarm)
Jika (ratio > 0.25) → "Berdiri/Duduk" (normal, tanpa alarm)
```

## 📹 Cara Test Sistem

### Option 1: Menggunakan Website
```
1. Buka: http://localhost:9000
2. Arahkan webcam ke furniture (kasur, kursi, dll)
3. Lihat apakah furniture terdeteksi di layar
4. Berbaring di kasur → Akan menampilkan "TIDUR" (biru, tanpa alarm)
5. Berbaring di lantai → Akan menampilkan "JATUH!" (merah, dengan alarm)
```

### Option 2: Test Lokal dengan Script
```bash
python simple_test.py
```
Script ini akan menampilkan:
- Video dari webcam
- Deteksi furniture dengan kotak hijau
- Status posisi dan apakah sedang di atas furniture

## 🐛 Debug Tips

Jika furniture masih tidak terdeteksi:

1. **Pastikan Lighting Cukup**
   - Furniture harus terlihat jelas di webcam
   - Hindari shadow yang terlalu gelap

2. **Ubah Confidence Threshold**
   Buka `app.py`, cari baris:
   ```python
   seg_results = seg_model(frame, conf=0.4, verbose=False)
   ```
   Turunkan nilai 0.4 menjadi 0.3 jika masih tidak terdeteksi

3. **Check Output Console**
   Lihat terminal yang menjalankan app.py untuk melihat log deteksi:
   ```
   Orang terdeteksi di atas: bed
   ```

## 📊 Output Console

Saat sistem berjalan, Anda akan melihat log:
```
Orang terdeteksi di atas: chair
Orang terdeteksi di atas: bed
Peringatan: Error in segmentation: [jika ada error]
```

## 🎯 Expected Results

| Posisi | Furniture | Status | Audio | Warna |
|--------|-----------|--------|-------|-------|
| Tidur | Kasur | "TIDUR" | ❌ OFF | Biru |
| Tidur | Lantai | "JATUH!" | ON | Merah |
| Duduk | Kursi | "Duduk di chair" | ❌ OFF | Hijau |
| Duduk | Lantai | "Duduk" | ❌ OFF | Hijau |
| Berdiri | - | "Berdiri" | ❌ OFF | Hijau |

## 📝 File-file Penting

- `app.py` - Aplikasi Flask utama dengan logic deteksi
- `simple_test.py` - Script untuk test lokal tanpa web
- `templates/index.html` - Interface web
- `datasets/coco8seg_test/weights/best.pt` - Model segmentasi terlatih

---

**Status**: Production Ready
**Last Update**: 12 Nov 2025
