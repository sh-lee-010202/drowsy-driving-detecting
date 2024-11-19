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
        self.cap = cv2.VideoCapture(0)
        self.face_detector = face_detector
        self.drowsiness_detector = drowsiness_detector
        self.pose = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5)
        
        #fps not int
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fourcc = cv2.VideoWriter_fourcc(*'X264')  # 코덱 설정
        self.save_brightness = cv2.VideoWriter("./preprocessing_video/brightness_adjust.mp4", self.fourcc, self.fps, (self.width, self.height), True)
        self.save_reflection = cv2.VideoWriter("./preprocessing_video/reflection_remove.mp4", self.fourcc, self.fps, (self.width, self.height), True)
        self.save_landmarks = cv2.VideoWriter("./preprocessing_video/landmarks.mp4", self.fourcc, self.fps, (self.width, self.height), True)
    
    def adjust_brightness(self, frame): #조명 보정
        # RGB를 Grayscale로 변환
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # CLAHE 생성 (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        adjusted_gray = clahe.apply(gray)
    
        # 다시 BGR로 변환하여 반환
        return cv2.cvtColor(adjusted_gray, cv2.COLOR_GRAY2BGR)
    
    def remove_reflection(self, frame): #HSV 반사광 제거(안경반사광제거)
        # BGR 이미지를 HSV로 변환
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # 반사광 제거를 위한 HSV 범위 설정
        lower_val = (0, 0, 0)  # 최소 값
        upper_val = (180, 255, 240)  # 최대 값 (반사광)
        # 범위 내의 픽셀을 필터링
        mask = cv2.inRange(hsv, lower_val, upper_val)
        # 원본 이미지에서 반사광 영역 제거
        filtered_frame = cv2.bitwise_and(frame, frame, mask=mask)
        return filtered_frame

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
            # 프레임 반전
            inversed = cv2.flip(frame, 1)

            # 1. 조명 보정
            brightness_adjusted_frame = self.adjust_brightness(inversed)

            # 2. 반사광 제거
            reflection_removed_frame = self.remove_reflection(brightness_adjusted_frame)

            image = cv2.cvtColor(reflection_removed_frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            face_results = self.face_detector.face_mesh.process(image)
            pose_results = self.pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if face_results.multi_face_landmarks and pose_results.pose_landmarks:
                for face_landmarks in face_results.multi_face_landmarks:
                    pitch = face_landmarks.landmark[self.face_detector.NOSE_TIP].y
                    yaw = face_landmarks.landmark[self.face_detector.NOSE_TIP].x
                    left_eye_ratio = self.face_detector.calculate_eye_ratio(face_landmarks.landmark, self.face_detector.LEFT_EYE_LANDMARKS)
                    right_eye_ratio = self.face_detector.calculate_eye_ratio(face_landmarks.landmark, self.face_detector.RIGHT_EYE_LANDMARKS)
                    ear = (left_eye_ratio + right_eye_ratio) / 2.0
            
                    calibration_status = self.drowsiness_detector.calibrate(pitch, yaw, ear)
                    
                        
                        
                    head_movement = self.drowsiness_detector.detect_head_movement(pitch, yaw)
                    drowsiness_status = self.drowsiness_detector.detect_drowsiness(pitch, yaw)
                    blink_count = self.face_detector.detect_blink(face_landmarks.landmark)
                    threshold = self.face_detector.ear_threshold
                    
                    # drowsiness 상태에 따른 시간 계산
                    state_info = self.drowsiness_detector.process(pitch, yaw)
                    state_duration = state_info["duration"]
                    self.draw_landmarks(image, face_landmarks.landmark, pose_results.pose_landmarks.landmark)
            
                    cv2.putText(inversed, calibration_status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.putText(inversed, f"Head Movement: {head_movement}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(inversed, f"Drowsiness Status: {drowsiness_status}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(inversed, f"Blink Count: {blink_count}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(inversed, f"Threshold: {threshold}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.putText(inversed, f"Time: {state_duration:.2f}s", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    
                    cv2.putText(image, calibration_status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.putText(image, f"Head Movement: {head_movement}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(image, f"Drowsiness Status: {drowsiness_status}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(image, f"Blink Count: {blink_count}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(image, f"Threshold: {threshold}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.putText(image, f"Time: {state_duration:.2f}s", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                
            # 처리된 프레임 저장
            self.save_brightness.write(brightness_adjusted_frame)
            self.save_reflection.write(reflection_removed_frame)
            self.save_landmarks.write(image)
            cv2.imshow("Drowsiness Detection with Landmarks", inversed)

            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()
        pygame.mixer.stop()