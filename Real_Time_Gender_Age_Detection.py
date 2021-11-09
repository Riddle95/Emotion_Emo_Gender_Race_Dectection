from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
import cvlib as cv

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

# load model
model_1 = load_model('Riddle_gender_detection.model')
model_2 = load_model('Riddle_race_detection.model')

# open webcam
webcam = cv2.VideoCapture(0)

classes_gender = ['Man', 'Woman']
classes_race = ['White', 'Black', 'Asian', 'Indian', 'Others']


# loop through frames
while webcam.isOpened():

    # read frame from webcam
    status, frame = webcam.read()


    # apply face detection
    face, confidence = cv.detect_face(frame)

    # loop through detected faces
    for idx, f in enumerate(face):

        # get corner points of face rectangle
        (startX, startY) = f[0], f[1]
        (endX, endY) = f[2], f[3]

        # draw rectangle over face
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

        # crop the detected face region
        face_crop = np.copy(frame[startY:endY, startX:endX])

        if (face_crop.shape[0]) < 10 or (face_crop.shape[1]) < 10:
            continue

        # preprocessing for gender detection model
        face_crop = cv2.resize(face_crop, (48, 48)) # resize the drop frame to (48,48)
        face_crop = face_crop.astype("float") / 255.0
        face_crop = img_to_array(face_crop) # convert image files into N-d array
        face_crop = np.expand_dims(face_crop, axis=0) # increase the dimension of the N-d array

        # apply gender detection on face
        conf_1 = model_1.predict(face_crop)[0] # predict gender
        conf_2 = model_2.predict(face_crop)[0] # predict race

        # get label with max accuracy
        idx_1 = np.argmax(conf_1) # get the gender prediction
        idx_2 = np.argmax(conf_2) # get the race prediction

        # convert the result from binary into classes for the visualization purpose

        label_gender = classes_gender[idx_1]
        label_race = classes_race[idx_2]

        # Gender, confidence level of gender prediction, race, confidence level of race prediction

        label = "{}: {:.2f}% : {}: {:.2f}%".format(label_gender, conf_1[idx_1] * 100, label_race, conf_2[idx_2] * 100)

        Y = startY - 10 if startY - 10 > 10 else startY + 10

        # write label and confidence above face rectangle
        cv2.putText(frame, label, (startX, Y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)

    # display output
    cv2.imshow("Gender & Race detection", frame)

    # press "Q" to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release resources
webcam.release()
cv2.destroyAllWindows()

