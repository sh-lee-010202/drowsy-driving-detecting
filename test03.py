# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 14:50:56 2024

@author: tjdgk
"""

import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

blink_count = 0
closed_eyes_frame = 0
#EAR = Eye Aspect Ratio 눈이 가로 세로 비율. 낮으면 closed
EAR_THRESHOLD = 0.2 #init 임계값
CLOSED_EYES_FRAMES_THRESHOLD = 1 #눈 감은 프레임 임계값

#사용자 마다 눈의 비율 먼저 계산 -> 이후 EAR_THRESHOLD 값 변경
calibration_count = 50
calibration_ear_values = []

LEFT_EYE_LANDMARKS = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144]

yaw_threshold = 0.02
pitch_threshold = 0.02

# Haar Cascade smile detector 로드
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

def calculate_eye_ratio(landmarks, eye_landmarks):
    left_right = np.linalg.norm(np.array([landmarks[eye_landmarks[0]].x, landmarks[eye_landmarks[0]].y]) -
                                np.array([landmarks[eye_landmarks[3]].x, landmarks[eye_landmarks[3]].y]))
    top_bottom_1 = np.linalg.norm(np.array([landmarks[eye_landmarks[1]].x, landmarks[eye_landmarks[1]].y]) -
                                  np.array([landmarks[eye_landmarks[5]].x, landmarks[eye_landmarks[5]].y]))
    top_bottom_2 = np.linalg.norm(np.array([landmarks[eye_landmarks[2]].x, landmarks[eye_landmarks[2]].y]) -
                                  np.array([landmarks[eye_landmarks[4]].x, landmarks[eye_landmarks[4]].y]))
    eye_ratio = (top_bottom_1 + top_bottom_2) / (2.0 * left_right)
    return eye_ratio

def detect_head_rotation(landmarks):
    nose_tip = landmarks[1]
    left_eye = landmarks[33]
    right_eye = landmarks[263]
    left_ear = landmarks[234]
    right_ear = landmarks[454]
    yaw = nose_tip.x - (left_eye.x + right_eye.x) / 2.0
    pitch = nose_tip.y - (left_ear.y + right_ear.y) / 2.0
    head_direction = ""
    if yaw > yaw_threshold:
        head_direction = "Looking Left"
    elif yaw < -yaw_threshold:
        head_direction = "Looking Right"
    elif pitch > pitch_threshold:
        head_direction = "Looking Down"
    elif pitch < -pitch_threshold:
        head_direction = "Looking Up"
    return head_direction

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = face_mesh.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            left_eye_ratio = calculate_eye_ratio(face_landmarks.landmark, LEFT_EYE_LANDMARKS)
            right_eye_ratio = calculate_eye_ratio(face_landmarks.landmark, RIGHT_EYE_LANDMARKS)
            eye_aspect_ratio = (left_eye_ratio + right_eye_ratio) / 2.0

            if calibration_count > 0:
                calibration_ear_values.append(eye_aspect_ratio)
                calibration_count -= 1
                cv2.putText(image, "Calibrating... Please keep eyes open", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                if calibration_count == 0:
                    EAR_THRESHOLD = np.mean(calibration_ear_values) * 0.7
                    print(f"Calibration complete. New EAR_THRESHOLD: {EAR_THRESHOLD}")
                    calibration_count -= 1

                if eye_aspect_ratio < EAR_THRESHOLD:
                    closed_eyes_frame += 1
                else:
                    if closed_eyes_frame >= CLOSED_EYES_FRAMES_THRESHOLD:
                        blink_count += 1
                        print(f"Blink detected. blink counts: {blink_count}")
                    closed_eyes_frame = 0

                head_direction = detect_head_rotation(face_landmarks.landmark)
                if head_direction:
                    cv2.putText(image, head_direction, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                cv2.putText(image, f"Blinks: {blink_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            mp_drawing.draw_landmarks(image, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION)


    smiles = smile_cascade.detectMultiScale(gray, scaleFactor=1.7, minNeighbors=22)
    for (sx, sy, sw, sh) in smiles:
        cv2.rectangle(frame, (sx, sy), (sx + sw, sy + sh), (0, 255, 0), 2)
        cv2.putText(frame, "Smile", (sx, sy - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Blink, Head Movement, and Smile Detection", frame)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(f"Total Blinks: {blink_count}")




