import streamlit as st
import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model
from emotion_label import emotion_labels   # ✅ fixed import
from emoji_mapper import get_emoji
from PIL import Image

st.set_page_config(page_title="Photo to Emoji", layout="centered")

st.title("😊 Photo to Emoji")
st.write("Upload a face image and get emotion + emoji")

@st.cache_resource
def load_assets():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    cascade_path = os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml")
    model_path = os.path.join(BASE_DIR, "model.h5")

    model = load_model(model_path)
    face_cascade = cv2.CascadeClassifier(cascade_path)

    # 🔐 Safety check
    if face_cascade.empty():
        st.error("❌ Haar cascade file not loaded")
    else:
        st.success("✅ Haar cascade loaded successfully")

    return model, face_cascade

model, face_cascade = load_assets()

uploaded_file = st.file_uploader(
    "Upload an image", 
    type=["jpg", "png", "jpeg"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(30, 30)
    )

    if len(faces) == 0:
        st.warning("No face detected")
    else:
        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (48, 48))
            face = face / 255.0
            face = face.reshape(1, 48, 48, 1)

            prediction = model.predict(face)
            emotion_index = np.argmax(prediction)
            emotion = emotion_labels[emotion_index]

            emoji = get_emoji(emotion)

            st.subheader(f"Detected Emotion: {emotion.upper()}")
            st.image(emoji, width=150)

        st.image(image, caption="Uploaded Image", use_container_width=True)
