# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 22:44:25 2024

@author: tjdgk
"""
import cv2
import mediapipe as mp
import pygame

class VideoStream:
    def __init__(self, face_detector, drowsiness_detector):
        self.cap = cv2.VideoCapture(0)  # 카메라 연결
        self.face_detector = face_detector
        self.drowsiness_detector = drowsiness_detector
        self.pose = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5)

        # FPS, 화면 크기 및 코덱 설정
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fourcc = cv2.VideoWriter_fourcc(*'DIVX')  # 영상 저장 코덱 설정
        self.save_brightness = cv2.VideoWriter("./preprocessing_video/brightness_adjust.avi", self.fourcc, self.fps, (self.width, self.height), True)
        self.save_reflection = cv2.VideoWriter("./preprocessing_video/reflection_remove.avi", self.fourcc, self.fps, (self.width, self.height), True)
        self.save_landmarks = cv2.VideoWriter("./preprocessing_video/landmarks.avi", self.fourcc, self.fps, (self.width, self.height), True)

    def adjust_brightness(self, frame):
        """조명 보정"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        adjusted_gray = clahe.apply(gray)
        return cv2.cvtColor(adjusted_gray, cv2.COLOR_GRAY2BGR)

    def remove_reflection(self, frame):
        """반사광 제거"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_val = (0, 0, 0)
        upper_val = (180, 255, 240)
        mask = cv2.inRange(hsv, lower_val, upper_val)
        return cv2.bitwise_and(frame, frame, mask=mask)

    def draw_landmarks(self, image, face_landmarks, pose_landmarks):
        """얼굴 및 어깨 랜드마크 표시"""
        key_points = self.face_detector.LEFT_EYE_LANDMARKS + self.face_detector.RIGHT_EYE_LANDMARKS + self.face_detector.MOUTH_LANDMARKS + [self.face_detector.NOSE_TIP]

        # 얼굴 랜드마크 표시
        for point in key_points:
            x = int(face_landmarks[point].x * image.shape[1])
            y = int(face_landmarks[point].y * image.shape[0])
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

        # 어깨 랜드마크 표시
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
            # 프레임 반전 및 조명 보정
            inversed = cv2.flip(frame, 1)
            brightness_adjusted_frame = self.adjust_brightness(inversed)
            reflection_removed_frame = self.remove_reflection(brightness_adjusted_frame)

            # Mediapipe 처리
            image = cv2.cvtColor(reflection_removed_frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            face_results = self.face_detector.face_mesh.process(image)
            pose_results = self.pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if face_results.multi_face_landmarks and pose_results.pose_landmarks:
                for face_landmarks in face_results.multi_face_landmarks:
                    # 얼굴 데이터 추출
                    pitch = face_landmarks.landmark[self.face_detector.NOSE_TIP].y
                    yaw = face_landmarks.landmark[self.face_detector.NOSE_TIP].x
                    head_movement = self.drowsiness_detector.detect_head_movement(pitch, yaw)
                    left_eye_ratio = self.face_detector.calculate_eye_ratio(face_landmarks.landmark, self.face_detector.LEFT_EYE_LANDMARKS)
                    right_eye_ratio = self.face_detector.calculate_eye_ratio(face_landmarks.landmark, self.face_detector.RIGHT_EYE_LANDMARKS)
                    ear = (left_eye_ratio + right_eye_ratio) / 2.0
                    blink_count = self.face_detector.detect_blink(face_landmarks.landmark)
                    threshold = self.face_detector.ear_threshold
                    


                    # 어깨 데이터 추출
                    left_shoulder = pose_results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value]
                    right_shoulder = pose_results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value]
                    shoulder_x = (left_shoulder.x + right_shoulder.x) / 2.0
                    shoulder_y = (left_shoulder.y + right_shoulder.y) / 2.0

                    # 캘리브레이션 및 상태 계산
                    calibration_status = self.drowsiness_detector.calibrate(pitch, yaw, ear, shoulder_x, shoulder_y)
                    drowsiness_status, state_duration = self.drowsiness_detector.process(pitch, yaw, shoulder_x, shoulder_y)


                    # 랜드마크 표시
                    self.draw_landmarks(image, face_landmarks.landmark, pose_results.pose_landmarks.landmark)

                    # 상태 텍스트 표시
                    cv2.putText(inversed, calibration_status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.putText(inversed, f"Head Movement: {head_movement}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(inversed, f"Drowsiness Status: {drowsiness_status}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(inversed, f"Blink Count: {blink_count}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(inversed, f"Threshold: {threshold}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.putText(inversed, f"Time: {state_duration:.2f}s", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    cv2.putText(inversed, f"Average Blink {self.drowsiness_detector.average_blink}", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                
                    
                    
                    cv2.putText(image, calibration_status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.putText(image, f"Head Movement: {head_movement}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(image, f"Drowsiness Status: {drowsiness_status}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(image, f"Blink Count: {blink_count}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(image, f"Threshold: {threshold}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.putText(image, f"Time: {state_duration:.2f}s", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    cv2.putText(image, f"Average Blink {self.drowsiness_detector.average_blink}", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            # 처리된 프레임 저장 및 출력
            self.save_brightness.write(brightness_adjusted_frame)
            self.save_reflection.write(reflection_removed_frame)
            self.save_landmarks.write(image)
            cv2.imshow("Drowsiness Detection with Landmarks", inversed)

            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()
        pygame.mixer.stop()