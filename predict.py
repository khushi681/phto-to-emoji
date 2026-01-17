import cv2
import numpy as np
from tensorflow.keras.models import load_model
from emotion_label import emotion_labels

model = load_model("model.h5")

def predict_emotion(face_img):
    face_img = cv2.resize(face_img, (48, 48))
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    face_img = face_img / 255.0 # normalize
    face_img = face_img.reshape(1, 48, 48, 1)

    prediction = model.predict(face_img)
    emotion_index = np.argmax(prediction) #picks the strongest emotion

    return emotion_labels[emotion_index]
