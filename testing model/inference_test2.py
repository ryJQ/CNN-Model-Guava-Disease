import numpy as np
import tensorflow as tf
from PIL import Image # Untuk memproses gambar

# --- PATHS KE FILE ANDA ---
MODEL_PATH = "D:\\KULIAH UNDIP\\SKRIPSI S NYA SANTUY\\python\\GuavaDisease_CNN_Model.tflite"
LABEL_PATH = 'Anthracnose', 'Fruit Flies', 'Healthy'
IMAGE_PATH = "D:\KULIAH UNDIP\SKRIPSI S NYA SANTUY\DATASET\GuavaDiseaseDataset\test\Anthracnose\29_unsharp_clahe_augmented_7.png" # Gambar yang akan Anda gunakan untuk menguji

try:
    # 1. Muat model TFLite
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    # Dapatkan detail input dan output model
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Dapatkan bentuk (shape) dan tipe data input yang diharapkan model
    input_shape = input_details[0]['shape']
    input_dtype = input_details[0]['dtype']

    print(f"Model input shape: {input_shape}")
    print(f"Model input dtype: {input_dtype}")

    # 2. Siapkan data masukan (gambar)
    # Asumsi model mengharapkan input [1, tinggi, lebar, 3] (batch, height, width, channels)
    # Sesuaikan ini jika model Anda berbeda (misalnya, grayscale, ukuran lain)
    target_height, target_width = input_shape[1], input_shape[2]

    img = Image.open(IMAGE_PATH).resize((target_width, target_height))
    input_data = np.array(img, dtype=input_dtype) # Ubah ke numpy array dengan dtype yang benar

    # Tambahkan dimensi batch (biasanya 1 untuk satu gambar)
    input_data = np.expand_dims(input_data, axis=0)

    # Normalisasi jika diperlukan (misalnya, dari 0-255 ke 0-1 atau -1 ke 1)
    # Contoh: input_data = input_data / 255.0
    # Contoh: input_data = (input_data - 127.5) / 127.5
    # Sesuaikan ini sesuai dengan bagaimana model Anda dilatih!

    # 3. Set tensor masukan
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # 4. Jalankan inferensi
    interpreter.invoke()

    # 5. Dapatkan dan interpretasikan hasil
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Baca label
    with open(LABEL_PATH, 'r') as f:
        labels = [line.strip() for line in f.readlines()]

    # Dapatkan indeks kelas dengan probabilitas tertinggi
    predicted_class_index = np.argmax(output_data)
    predicted_probability = output_data[0][predicted_class_index]
    predicted_label = labels[predicted_class_index]

    print(f"\nHasil Prediksi:")
    print(f"  Kelas Terprediksi: {predicted_label}")
    print(f"  Probabilitas: {predicted_probability:.4f}")

    # Anda bisa cetak semua probabilitas jika ingin
    # for i, prob in enumerate(output_data[0]):
    #     print(f"  {labels[i]}: {prob:.4f}")

    print("\nModel tampaknya berjalan dengan baik jika prediksi sesuai harapan!")

except Exception as e:
    print(f"Terjadi kesalahan saat mengecek model: {e}")
    print("Pastikan PATHS sudah benar dan model/data sesuai.")