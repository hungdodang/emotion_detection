import numpy as np
import cv2

from tensorflow.keras.models import load_model
from wrn_model import WideResNet

from array_utility import scale_array
from image_ultility import snip_bounding_box, create_labeled_bounding_box

# path to pretrained model
EMOTION_MODEL = "models/emotion_model.hdf5"
FACE_MODEL = "models/haarcascade_frontalface_default.xml"
GENDER_AGE_MODEL = "models/weights.18-4.06.hdf5"


def predict_age_gender(face_image, model):
    """
    Determine the age and gender of the face in the picture
    :param face_image: image of the face
    :return: (age, gender) of the image
    """
    face_imgs = np.empty((1, 64, 64, 3))
    face_imgs[0, :, :, :] = face_image
    result = model.predict(face_imgs)
    est_gender = "F" if result[0][0][0] > 0.5 else "M"
    est_age = int(result[1][0].dot(np.arange(0, 101).reshape(101, 1)).flatten()[0])
    return est_age, est_gender


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
    emotion_label_index = np.argmax(prediction)
    return emotion_labels[emotion_label_index]


def main():
    # emotion model
    emotion_classifier = load_model(EMOTION_MODEL)
    emotion_target_size = emotion_classifier.input_shape[1:3]
    # face detection model
    face_cascade = cv2.CascadeClassifier(FACE_MODEL)
    # age and gender model
    age_gender_model = WideResNet(64, 16, 8)()
    age_gender_model.load_weights(GENDER_AGE_MODEL)
    # capture video with webcam
    capture = cv2.VideoCapture(0)
    while capture.isOpened():
        _, frame = capture.read()
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # detect faces in picture
        faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for f in faces:
            face_image, area_of_face = snip_bounding_box(frame, f)
            try:
                face_image = cv2.resize(face_image, (emotion_target_size))
            except:
                continue
            age, gender = predict_age_gender(face_image, age_gender_model)
            emotion = predict_emotion(face_image, emotion_classifier)
            label = f"{age} {gender} {emotion}"
            create_labeled_bounding_box(frame, area_of_face, label)
        cv2.imshow('GAE Prediction', frame)
        # press Q to stop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    main()
