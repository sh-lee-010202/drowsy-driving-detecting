# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 22:44:25 2024

@author: tjdgk
"""
import cv2
import mediapipe as mp
import pygame
import datetime

class VideoStream:
    def __init__(self, face_detector, drowsiness_detector):
        """
        VideoStream 초기화
        :param face_detector: FaceDetector 객체 (눈 깜빡임, EAR 계산 등 처리)
        :param drowsiness_detector: DrowsinessDetector 객체 (졸음 상태 감지 처리)
        """
        # 웹캠 연결
        self.cap = cv2.VideoCapture(0)
        self.face_detector = face_detector
        self.drowsiness_detector = drowsiness_detector

        # Mediapipe Pose 초기화 (어깨 데이터 감지)
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False,  # 실시간 모드
            model_complexity=1,  # 모델 복잡도 (기본값)
            min_detection_confidence=0.5  # 감지 신뢰도
        )

        # 전처리 영상 저장을 위한 설정
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.brightness_filename = f"./preprocessing_video/brightness_adjust_{self.timestamp}.avi"
        self.reflection_filename = f"./preprocessing_video/reflection_remove_{self.timestamp}.avi"
        self.landmarks_filename = f"./preprocessing_video/landmarks_{self.timestamp}.avi"

        # 웹캠 FPS 및 화면 크기 설정
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fourcc = cv2.VideoWriter_fourcc(*'DIVX')  # 영상 코덱 설정

        # 각 처리 단계의 영상 저장 객체 초기화
        self.save_brightness = cv2.VideoWriter(self.brightness_filename, self.fourcc, int(self.fps), (self.width, self.height), True)
        self.save_reflection = cv2.VideoWriter(self.reflection_filename, self.fourcc, int(self.fps), (self.width, self.height), True)
        self.save_landmarks = cv2.VideoWriter(self.landmarks_filename, self.fourcc, int(self.fps), (self.width, self.height), True)

    def adjust_brightness(self, frame):
        """
        조명 보정 (CLAHE를 사용한 밝기 조절)
        :param frame: 입력 프레임
        :return: 밝기 보정된 프레임
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # grayscale 변환
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # CLAHE 초기화
        adjusted_gray = clahe.apply(gray)  # 밝기 보정 적용
        return cv2.cvtColor(adjusted_gray, cv2.COLOR_GRAY2BGR)  # 다시 컬러로 변환

    def remove_reflection(self, frame):
        """
        반사광 제거 (HSV 마스크 적용)
        :param frame: 입력 프레임
        :return: 반사광이 제거된 프레임
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # HSV 색공간 변환
        lower_val = (0, 0, 0)  # HSV 범위 설정 (어두운 부분)
        upper_val = (180, 255, 240)  # HSV 범위 설정 (밝기 상한선)
        mask = cv2.inRange(hsv, lower_val, upper_val)  # 마스크 생성
        return cv2.bitwise_and(frame, frame, mask=mask)  # 마스크 적용 (결과: 명도 240 넘는 부분 0으로 변경)

    def draw_landmarks(self, image, face_landmarks, pose_landmarks):
        """
        얼굴 및 어깨 랜드마크 표시
        :param image: 입력 이미지
        :param face_landmarks: 얼굴 랜드마크 데이터
        :param pose_landmarks: 어깨 랜드마크 데이터
        """
        key_points = self.face_detector.LEFT_EYE_LANDMARKS + self.face_detector.RIGHT_EYE_LANDMARKS + \
                     [self.face_detector.NOSE_TIP]

        # 얼굴 랜드마크 표시
        for point in key_points:
            x = int(face_landmarks[point].x * image.shape[1])  # x 좌표 계산
            y = int(face_landmarks[point].y * image.shape[0])  # y 좌표 계산
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)  # 랜드마크 점 그리기

        # 어깨 랜드마크 표시
        left_shoulder = pose_landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = pose_landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value]
        for shoulder in [left_shoulder, right_shoulder]:
            x = int(shoulder.x * image.shape[1])  # x 좌표 계산
            y = int(shoulder.y * image.shape[0])  # y 좌표 계산
            cv2.circle(image, (x, y), 5, (0, 255, 255), -1)  # 랜드마크 점 그리기

    def start(self):
        """
        main 영상 처리 함수
        """
        while self.cap.isOpened():
            ret, frame = self.cap.read()  # 프레임 읽기
            if not ret:
                break

            # 프레임 반전 및 조명 보정
            inversed = cv2.flip(frame, 1)  # 좌우 반전(사용자가 보는 실제 영상)
            brightness_adjusted_frame = self.adjust_brightness(inversed)  # 밝기 조정
            reflection_removed_frame = self.remove_reflection(brightness_adjusted_frame)  # 반사광 제거

            # Mediapipe 처리
            image = cv2.cvtColor(reflection_removed_frame, cv2.COLOR_BGR2RGB)  # BGR -> RGB 변환
            """
            Mediapipe landmark를 사용하려면 기존 BGR이 아닌 RGB 순서의 채널 형태로 사용해야 함.
            """
            image.flags.writeable = False
            face_results = self.face_detector.face_mesh.process(image)  # 얼굴 랜드마크 처리
            pose_results = self.pose.process(image)  # 어깨 랜드마크 처리
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # RGB -> BGR 변환
            """
            brightness_adjust, reflection_removed, image 영상은 
            전처리에 사용한 영상이라 그레이스케일이기 때문에 사용자에게 보이지 않도록 함.
            """

            if face_results.multi_face_landmarks and pose_results.pose_landmarks:
                for face_landmarks in face_results.multi_face_landmarks:
                    # 얼굴 데이터 추출
                    pitch = face_landmarks.landmark[self.face_detector.NOSE_TIP].y
                    yaw = face_landmarks.landmark[self.face_detector.NOSE_TIP].x
                    head_movement = self.drowsiness_detector.detect_head_movement(pitch, yaw)

                    left_eye_ratio = self.face_detector.calculate_eye_ratio(face_landmarks.landmark, self.face_detector.LEFT_EYE_LANDMARKS)
                    right_eye_ratio = self.face_detector.calculate_eye_ratio(face_landmarks.landmark, self.face_detector.RIGHT_EYE_LANDMARKS)
                    ear = (left_eye_ratio + right_eye_ratio) / 2.0  # EAR 계산
                    blink_count = self.face_detector.detect_blink(face_landmarks.landmark)
                    threshold = self.face_detector.ear_threshold
                    

                    # 어깨 데이터 추출
                    left_shoulder = pose_results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value]
                    right_shoulder = pose_results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value]
                    shoulder_x = (left_shoulder.x + right_shoulder.x) / 2.0
                    shoulder_y = (left_shoulder.y + right_shoulder.y) / 2.0

                    # 캘리브레이션 및 상태 계산
                    calibration_status = self.drowsiness_detector.calibrate(pitch, yaw, ear, shoulder_x, shoulder_y)
                    average_blink, drowsiness_status, state_duration = self.drowsiness_detector.process(pitch, yaw, shoulder_x, shoulder_y)

                    # 랜드마크 표시
                    self.draw_landmarks(image, face_landmarks.landmark, pose_results.pose_landmarks.landmark)
                    
                    # 영상에 상태 표시        
                    # 전처리영상 "landmark"에 텍스트 표시
                    cv2.putText(image, calibration_status, (10, 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
                    cv2.putText(image, f"Head Movement: {head_movement}", (10, 40), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
                    cv2.putText(image, f"Drowsiness Status: {drowsiness_status}", (10, 60), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
                    cv2.putText(image, f"Blink Count: {blink_count}", (10, 80), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
                    cv2.putText(image, f"Threshold: {threshold}", (10, 100), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
                    cv2.putText(image, f"Time: {state_duration:.2f}s", (10, 120), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
                    cv2.putText(image, f"Average Blink: {average_blink}", (10, 140), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
                    
                    # 사용자에게 보이는 실제 영상(거울모드)에 텍스트 표시
                    cv2.putText(inversed, calibration_status, (10, 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
                    cv2.putText(inversed, f"Head Movement: {head_movement}", (10, 40), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
                    cv2.putText(inversed, f"Drowsiness Status: {drowsiness_status}", (10, 60), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
                    cv2.putText(inversed, f"Blink Count: {blink_count}", (10, 80), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
                    cv2.putText(inversed, f"Threshold: {threshold}", (10, 100), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
                    cv2.putText(inversed, f"Time: {state_duration:.2f}s", (10, 120), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
                    cv2.putText(inversed, f"Average Blink: {average_blink}", (10, 140), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)

            # 처리된 프레임 저장 및 출력
            self.save_brightness.write(brightness_adjusted_frame)
            self.save_reflection.write(reflection_removed_frame)
            self.save_landmarks.write(image)
            cv2.imshow("Drowsiness Detection with Landmarks", inversed)

            # 'q' 키를 누르면 종료
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

        # 종료 및 리소스 해제
        self.cap.release()
        cv2.destroyAllWindows()
        pygame.mixer.stop()