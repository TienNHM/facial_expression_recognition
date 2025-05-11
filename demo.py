import streamlit as st
import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
import torchvision.transforms as transforms

st.set_page_config(page_title="Facial Expression Recognition", layout="centered")
st.title("😃 Real-time Facial Expression Recognition")
st.markdown("Upload a face image (48x48 grayscale or larger) or use your webcam.")

# Nhãn cảm xúc (phải đúng thứ tự)
# class_names = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
class_names = ['Happy', 'Sleepy', 'Surprise']

# Load ONNX model
session = ort.InferenceSession("models/emotion_cnn.onnx", providers=['CPUExecutionProvider'])

# Hàm tiền xử lý
def preprocess_image(img: Image.Image):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    tensor = transform(img).unsqueeze(0).numpy()  # shape: (1, 1, 48, 48)
    return tensor

# Hàm dự đoán
def predict_emotion(img: Image.Image):
    input_tensor = preprocess_image(img)
    outputs = session.run(None, {"input": input_tensor})
    scores = outputs[0][0]
    predicted = np.argmax(scores)
    confidence = float(np.exp(scores[predicted]) / np.sum(np.exp(scores)))
    return class_names[predicted], confidence

# Giao diện
uploaded_file = st.file_uploader("📁 Upload ảnh khuôn mặt", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Ảnh đã tải lên", width=200)
    emotion, conf = predict_emotion(img)
    st.success(f"**Emotion:** {emotion} ({conf*100:.2f}%)")

# Webcam
if st.button("📷 Dùng webcam"):
    st.warning("⚠️ Vui lòng chạy file local vì Streamlit cloud không hỗ trợ webcam.")
