import cv2
import onnxruntime as ort
import numpy as np
import torchvision.transforms as transforms
import streamlit as st
from PIL import Image

# Nhãn cảm xúc
class_names = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Load ONNX model
session = ort.InferenceSession("models/emotion_cnn.onnx", providers=['CPUExecutionProvider'])

# Preprocessing
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    st.error("❌ Không mở được webcam.")
    exit()

st.title("Real-time Facial Expression Recognition")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml").detectMultiScale(gray)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        input_tensor = transform(face).unsqueeze(0).numpy()

        outputs = session.run(None, {"input": input_tensor})
        scores = outputs[0][0]
        pred = np.argmax(scores)
        label = f"{class_names[pred]} ({np.max(scores):.2f})"

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Chuyển frame OpenCV thành hình ảnh Streamlit
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    st.image(frame_rgb, channels="RGB", use_column_width=True)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
