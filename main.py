import cv2
import time
import mediapipe as mp
from pathlib import Path

# =====================================
# 1. LOKASI MODEL HAND LANDMARKER
# =====================================

model_path = Path("models/hand_landmarker.task")

if not model_path.exists():
    print("Model Hand Landmarker tidak ditemukan!")
    exit()

print("Model Hand Landmarker berhasil ditemukan!")


# =====================================
# 2. KONFIGURASI MEDIAPIPE
# =====================================

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(model_path)
    ),
    running_mode=RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

landmarker = HandLandmarker.create_from_options(options)

print("MediaPipe Hand Landmarker berhasil dijalankan!")


# =====================================
# 3. MEMBUKA WEBCAM
# =====================================

kamera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

kamera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
kamera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not kamera.isOpened():
    print("Webcam tidak dapat dibuka!")
    landmarker.close()
    exit()

print("Webcam berhasil dibuka!")


# =====================================
# 4. KONEKSI 21 TITIK LANDMARK
# =====================================

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]


# =====================================
# 5. PENGATURAN WARNA
# =====================================

cyan = (255, 220, 0)
putih = (255, 255, 255)
hijau = (0, 255, 120)
biru = (255, 100, 0)

waktu_sebelumnya = time.time()
timestamp_terakhir = 0


# =====================================
# 6. PROSES DETEKSI WEBCAM
# =====================================

try:

    while True:

        berhasil, frame = kamera.read()

        if not berhasil:
            print("Gagal mengambil gambar dari webcam.")
            break

        # Membalik gambar seperti cermin
        frame = cv2.flip(frame, 1)

        tinggi, lebar = frame.shape[:2]

        # Mengubah gambar BGR menjadi RGB
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # Membuat objek gambar MediaPipe
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # Timestamp harus selalu meningkat
        timestamp_sekarang = max(
            timestamp_terakhir + 1,
            int(time.perf_counter() * 1000)
        )

        timestamp_terakhir = timestamp_sekarang

        # Mendeteksi tangan
        hasil = landmarker.detect_for_video(
            mp_image,
            timestamp_sekarang
        )

        jumlah_tangan = len(hasil.hand_landmarks)

        # =================================
        # 7. MENGGAMBAR 21 TITIK LANDMARK
        # =================================

        for tangan in hasil.hand_landmarks:

            # Mengubah koordinat normalisasi
            # menjadi koordinat piksel webcam

            titik = []

            for landmark in tangan:

                x = int(landmark.x * lebar)
                y = int(landmark.y * tinggi)

                titik.append((x, y))

            # Menggambar garis penghubung
            for awal, akhir in HAND_CONNECTIONS:

                cv2.line(
                    frame,
                    titik[awal],
                    titik[akhir],
                    cyan,
                    2
                )

            # Menggambar 21 titik landmark
            for nomor, posisi in enumerate(titik):

                x, y = posisi

                # Titik ujung jari dibuat lebih besar
                if nomor in [4, 8, 12, 16, 20]:

                    cv2.circle(
                        frame,
                        (x, y),
                        8,
                        hijau,
                        -1
                    )

                else:

                    cv2.circle(
                        frame,
                        (x, y),
                        5,
                        putih,
                        -1
                    )

                # Lingkaran luar
                cv2.circle(
                    frame,
                    (x, y),
                    9,
                    cyan,
                    1
                )


        # =================================
        # 8. MENGHITUNG FPS
        # =================================

        waktu_sekarang = time.time()

        fps = 1 / max(
            waktu_sekarang - waktu_sebelumnya,
            0.001
        )

        waktu_sebelumnya = waktu_sekarang


        # =================================
        # 9. TAMPILAN FUTURISTIK
        # =================================

        # Panel atas
        overlay = frame.copy()

        cv2.rectangle(
            overlay,
            (0, 0),
            (lebar, 85),
            (15, 20, 30),
            -1
        )

        cv2.addWeighted(
            overlay, 0.75,
            frame, 0.25,
            0, frame
        )

        # Judul sistem
        cv2.putText(
            frame,
            "TOUCHLESS GESTURE CURSOR",
            (30, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            cyan,
            2
        )

        cv2.putText(
            frame,
            "HAND CONTROL SYSTEM",
            (32, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            putih,
            1
        )

        # Status kamera
        cv2.circle(
            frame,
            (lebar - 205, 38),
            7,
            hijau,
            -1
        )

        cv2.putText(
            frame,
            "CAMERA ACTIVE",
            (lebar - 185, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            putih,
            1
        )

        # Garis sudut futuristik
        ukuran = 45
        tebal = 3

        cv2.line(frame, (20, 105),
                 (20 + ukuran, 105), cyan, tebal)

        cv2.line(frame, (20, 105),
                 (20, 105 + ukuran), cyan, tebal)

        cv2.line(frame, (lebar - 20, 105),
                 (lebar - 20 - ukuran, 105), cyan, tebal)

        cv2.line(frame, (lebar - 20, 105),
                 (lebar - 20, 105 + ukuran), cyan, tebal)

        cv2.line(frame, (20, tinggi - 20),
                 (20 + ukuran, tinggi - 20), cyan, tebal)

        cv2.line(frame, (20, tinggi - 20),
                 (20, tinggi - 20 - ukuran), cyan, tebal)

        cv2.line(frame, (lebar - 20, tinggi - 20),
                 (lebar - 20 - ukuran, tinggi - 20), cyan, tebal)

        cv2.line(frame, (lebar - 20, tinggi - 20),
                 (lebar - 20, tinggi - 20 - ukuran), cyan, tebal)


        # =================================
        # 10. PANEL INFORMASI BAWAH
        # =================================

        overlay = frame.copy()

        cv2.rectangle(
            overlay,
            (0, tinggi - 75),
            (lebar, tinggi),
            (15, 20, 30),
            -1
        )

        cv2.addWeighted(
            overlay, 0.8,
            frame, 0.2,
            0, frame
        )

        # Informasi FPS
        cv2.putText(
            frame,
            "FPS: {:.1f}".format(fps),
            (30, tinggi - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            hijau,
            2
        )

        # Status deteksi tangan
        if jumlah_tangan > 0:

            status = "HAND DETECTED: " + str(jumlah_tangan)

            warna_status = hijau

        else:

            status = "HAND DETECTION: WAITING"

            warna_status = putih

        cv2.putText(
            frame,
            status,
            (190, tinggi - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            warna_status,
            1
        )

        # Informasi keluar
        cv2.putText(
            frame,
            "PRESS ESC TO EXIT",
            (lebar - 245, tinggi - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            cyan,
            1
        )


        # =================================
        # 11. MENAMPILKAN WEBCAM
        # =================================

        cv2.imshow(
            "TOUCHLESS GESTURE CURSOR",
            frame
        )

        # Keluar menggunakan ESC
        if cv2.waitKey(1) & 0xFF == 27:
            break


finally:

    # Menutup semua perangkat
    kamera.release()
    landmarker.close()

    cv2.destroyAllWindows()

    print("Webcam telah dihentikan.")