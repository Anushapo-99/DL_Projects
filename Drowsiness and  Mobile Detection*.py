from twilio.rest import Client
from scipy.spatial import distance
from imutils import face_utils
import imutils
import dlib
import cv2
import pygame
import numpy as np

def eye_aspect_ratio(eye):
    a = distance.euclidean(eye[1], eye[5])
    b = distance.euclidean(eye[2], eye[4])
    c = distance.euclidean(eye[0], eye[3])
    ear = (a + b) / (2.0 * c)
    return ear

def mouth_aspect_ratio(mouth):
    a = distance.euclidean(mouth[3], mouth[9])
    b = distance.euclidean(mouth[0], mouth[6])
    mar = a / b
    return mar

def play_alert_sound(sound):
    sound.play()

pygame.mixer.init()
pygame.mixer.music.load('alert-sound.wav')
yawn_alert_sound = pygame.mixer.Sound('take_a_break.wav')
mobile_alert_sound = pygame.mixer.Sound("severe-warning-alarm-98704.wav")

thresh_drowsiness = 0.25
frame_check_drowsiness = 7
thresh_yawn = 0.7
frame_check_yawn = 7

detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("shape_predictor_68_face_landmarks(1).dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["mouth"]

video = cv2.VideoCapture(0)

# Twilio credentials
account_sid = "ACec19f5a1abfeebfb8471b371f5536498"
auth_token = "e887b125da4e60e89c64b4beec663b22"
twilio_phone_number = "+15417274008"
your_phone_number = "+919961845031"

# Twilio client
client = Client(account_sid, auth_token)


#Mobile detetcion

whT = 320             #width and height of frame(pixel)
confThreshold = 0.5   #confidence threshold
nmsThreshold = 0.3    #Non_maximum_supression
classNames = ["mobile"]
modelConfiguration = "yolov3_custom.cfg"
modelWeights = "yolov3_custom_1000(1).weights"
net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


def findObjects(outputs, img):
    hT, wT, cT = img.shape
    boundingBoxes = []
    confidenceValues = []

    for output in outputs:
        for detection in output:
            probScores = detection[5:]
            classIndex = np.argmax(probScores)
            confidence = probScores[classIndex]

            if confidence >= confThreshold and classIndex == 0:
                w, h = int(detection[2]*wT), int(detection[3]*hT)
                x, y = int((detection[0]*wT)-w/2), int((detection[1]*hT)-h/2)

                boundingBoxes.append([x, y, w, h])
                confidenceValues.append(float(confidence))

    indices = cv2.dnn.NMSBoxes(boundingBoxes, confidenceValues, confThreshold, nmsThreshold)

    for i in indices:
        box = boundingBoxes[i]
        x, y, w, h = box[0], box[1], box[2], box[3]

        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.rectangle(img, (x - 1, y - 25), (x + w+1, y), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, f'{classNames[0].upper()} {int(confidenceValues[i] * 100)}%',
                    (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        play_alert_sound(mobile_alert_sound)

#Drowiness detection

flag_drowsiness = 0
flag_yawn = 0

while True:
    suc, frame = video.read()
    frame = imutils.resize(frame, width=600)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    subjects = detect(gray, 0)

    for subject in subjects:
        shape = predict(gray, subject)
        shape = face_utils.shape_to_np(shape)
        left_eye = shape[lStart:lEnd]
        right_eye = shape[rStart:rEnd]
        mouth = shape[mStart:mEnd]

        left_eye_aspect_ratio = eye_aspect_ratio(left_eye)
        right_eye_aspect_ratio = eye_aspect_ratio(right_eye)
        eye_aspect_ratio_avg = (left_eye_aspect_ratio + right_eye_aspect_ratio) / 2.0

        mouth_aspect_ratio_val = mouth_aspect_ratio(mouth)

        if eye_aspect_ratio_avg < thresh_drowsiness:
            flag_drowsiness += 1

            if flag_drowsiness >= frame_check_drowsiness:
                if not pygame.mixer.music.get_busy():
                    pygame.mixer.music.play(-1)
                
                # Send SMS alert for drowsiness using Twilio
                message = client.messages.create(
                    body="Your friend is feeling drowsy. Advise him to take a break.",
                    from_=twilio_phone_number,
                    to=your_phone_number
                )

                cv2.putText(frame, "****YOU ARE TIRED****", (10, 30),
                            cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 0), 2)
                cv2.putText(frame, "****Drowsiness Detected******", (10, 250),
                            cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)

        else:
            pygame.mixer.music.stop()
            flag_drowsiness = 0

        if mouth_aspect_ratio_val > thresh_yawn:
            flag_yawn += 1

            if flag_yawn >= frame_check_yawn:
                cv2.putText(frame, "****Yawn Detected!****", (20, 90),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)

                if pygame.mixer.get_busy() == 0:
                    yawn_alert_sound.play()

        else:
            flag_yawn = 0

    blob = cv2.dnn.blobFromImage(frame, 1 / 255, (whT, whT), [0, 0, 0], swapRB=True, crop=False)
    net.setInput(blob)

    outputLayersNames = net.getUnconnectedOutLayersNames()
    outputs = net.forward(outputLayersNames)

    findObjects(outputs, frame)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0XFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()
video.release()
