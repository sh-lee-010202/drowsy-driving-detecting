# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 22:44:25 2024

@author: tjdgk
"""
import cv2
import mediapipe as mp

class VideoStream:
    def __init__(self, face_detector, drowsiness_detector):
        self.cap = cv2.VideoCapture(0)
        self.face_detector = face_detector
        self.drowsiness_detector = drowsiness_detector
        self.pose = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5)

    def draw_landmarks(self, image, face_landmarks, pose_landmarks):
        key_points = self.face_detector.LEFT_EYE_LANDMARKS + self.face_detector.RIGHT_EYE_LANDMARKS + self.face_detector.MOUTH_LANDMARKS + [self.face_detector.NOSE_TIP]

        for point in key_points:
            x = int(face_landmarks[point].x * image.shape[1])
            y = int(face_landmarks[point].y * image.shape[0])
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

        left_shoulder = pose_landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = pose_landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value]

        for shoulder in [left_shoulder, right_shoulder]:
            x = int(shoulder.x * image.shape[1])
            y = int(shoulder.y * image.shape[0])
            cv2.circle(image, (x, y), 5, (0, 255, 255), -1)

    def start(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            face_results = self.face_detector.face_mesh.process(image)
            pose_results = self.pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if face_results.multi_face_landmarks and pose_results.pose_landmarks:
                for face_landmarks in face_results.multi_face_landmarks:
                    pitch = face_landmarks.landmark[self.face_detector.NOSE_TIP].y
                    yaw = face_landmarks.landmark[self.face_detector.NOSE_TIP].x
                    calibration_status = self.drowsiness_detector.calibrate(pitch, yaw)
                    head_movement = self.drowsiness_detector.detect_head_movement(pitch, yaw)
                    drowsiness_status = self.drowsiness_detector.detect_drowsiness(pitch, yaw)
                    blink_count = self.face_detector.detect_blink(face_landmarks.landmark)

                    self.draw_landmarks(image, face_landmarks.landmark, pose_results.pose_landmarks.landmark)

                    cv2.putText(image, calibration_status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.putText(image, f"Head Movement: {head_movement}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(image, f"Drowsiness Status: {drowsiness_status}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(image, f"Blink Count: {blink_count}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Drowsiness Detection with Landmarks", image)

            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()