import numpy as np
import cv2

from keras.models import load_model
from model import WideResNet

from array_utility import scale_array
from image_ultility import snip_bounding_box, create_labeled_bounding_box

EMOTION_MODEL = "emotion_model.hdf5"
FACE_MODEL = "haarcascade_frontalface_default.xml"

emotion_classifier = load_model(EMOTION_MODEL)
emotion_target_size = emotion_classifier.input_shape[1:3]

face_cascade = cv2.CascadeClassifier(FACE_MODEL)

def predict_emotion(face_image, model):
    """
    Predict the emotion of the face in the image
    :param face_image: face image
    :param model: emotion classify model
    :return: face emotion as string
    """

    emotion_labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'sad', 5: 'surprise', 6: 'neutral'}

    gray_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    gray_image = scale_array(gray_image)
    gray_image = np.expand_dims(gray_image, 0)
    gray_image = np.expand_dims(gray_image, -1)

    prediction = model.predict(gray_image)
    emotion_probability = np.max(prediction)
    emotion_label_index = np.argmax(emotion_probability)
    return emotion_labels[emotion_label_index]

capture = cv2.VideoCapture(0)
while capture.isOpened():
    _, frame = capture.read()
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for f in faces:
        face_image, area_of_face = snip_bounding_box(frame, f)
        emotion = predict_emotion(face_image)
        label = f"{emotion}"
        create_labeled_bounding_box(frame, area_of_face)

    cv2.imshow('Emotion prediction', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

