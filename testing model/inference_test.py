import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input

# Path model dan gambar
MODEL_PATH = r"D:\\KULIAH UNDIP\\SKRIPSI S NYA SANTUY\\python\\GuavaDisease_CNN_Model.tflite"
IMAGE_PATH = r"D:\KULIAH UNDIP\SKRIPSI S NYA SANTUY\DATASET\GuavaDiseaseDataset\test\fruit_fly\22_unsharp_clahe_augmented_7.png"

# Harus cocok dengan urutan class_names saat training
CLASS_NAMES = ['anthracnose', 'fruit_fly', 'healthy_guava']

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Get input/output info
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load dan preprocess gambar
img = Image.open(IMAGE_PATH).convert('RGB').resize((224, 224))
img_array = np.array(img).astype(np.float32)
img_array = preprocess_input(img_array)  # PENTING untuk EfficientNet
img_array = np.expand_dims(img_array, axis=0)

# Set input
interpreter.set_tensor(input_details[0]['index'], img_array)

# Inference
interpreter.invoke()

# Get output
output = interpreter.get_tensor(output_details[0]['index'])
predicted_index = np.argmax(output)
confidence = output[0][predicted_index]

# Output
print(f"Prediksi: {CLASS_NAMES[predicted_index]}")
print(f"Akurasi: {confidence * 100:.2f}%")
print(f"Output raw: {output}")
