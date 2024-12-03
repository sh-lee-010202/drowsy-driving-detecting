# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 22:42:49 2024

@author: tjdgk
"""

import mediapipe as mp
import numpy as np

class FaceDetector:
    def __init__(self, ear_threshold=0.2):
        """
        FaceDetector init
        :param ear_threshold: EAR(Eye Aspect Ratio) 임계값, 눈이 감긴 상태로 간주되는 기준값
        """
        # Mediapipe의 FaceMesh 초기화
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,  # 실시간 모드로 작동 (고정이미지는 True)
            max_num_faces=1,  # 처리할 얼굴의 최대 수
            min_detection_confidence=0.5  # 얼굴 감지 신뢰도
        )

        # Mediapipe 얼굴 랜드마크 인덱스
        self.LEFT_EYE_LANDMARKS = [362, 385, 387, 263, 373, 380]  # 왼쪽 눈 랜드마크
        self.RIGHT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144]  # 오른쪽 눈 랜드마크
        self.NOSE_TIP = 1  # 코 끝 랜드마크

        # EAR 임계값 설정
        self.ear_threshold = ear_threshold

        # 눈 깜빡임 관련 상태
        self.blink_count = 0  # 총 깜빡임 횟수
        self.closed_eyes_frame = 0  # 눈을 감고 있는 연속 프레임 수
        self.ear = 0  # 현재 EAR 값

    def calculate_eye_ratio(self, landmarks, eye_landmarks):
        """
        눈의 EAR(Eye Aspect Ratio) 계산
        :param landmarks: 얼굴 랜드마크 좌표 리스트
        :param eye_landmarks: EAR 계산에 사용할 눈 랜드마크 인덱스 ex) LEFT_EYE_LANMARKS
        :return: 계산된 EAR 값
        """
        # 수평 길이 계산 (눈 좌우 거리)
        left_right = np.linalg.norm(
            np.array([landmarks[eye_landmarks[0]].x, landmarks[eye_landmarks[0]].y]) -
            np.array([landmarks[eye_landmarks[3]].x, landmarks[eye_landmarks[3]].y])
        )

        # 수직 길이 계산 (눈 위아래 거리)
        top_bottom_1 = np.linalg.norm(
            np.array([landmarks[eye_landmarks[1]].x, landmarks[eye_landmarks[1]].y]) -
            np.array([landmarks[eye_landmarks[5]].x, landmarks[eye_landmarks[5]].y])
        )
        top_bottom_2 = np.linalg.norm(
            np.array([landmarks[eye_landmarks[2]].x, landmarks[eye_landmarks[2]].y]) -
            np.array([landmarks[eye_landmarks[4]].x, landmarks[eye_landmarks[4]].y])
        )

        # EAR 계산 (수직 길이 평균 / 수평 길이)
        eye_ratio = (top_bottom_1 + top_bottom_2) / (2.0 * left_right)
        return eye_ratio

    def detect_blink(self, landmarks):
        """
        눈 깜빡임 감지
        :param landmarks: 얼굴 랜드마크 좌표 리스트
        :return: 현재까지 감지된 총 깜빡임 횟수
        """
        # 왼쪽 눈과 오른쪽 눈의 EAR 계산
        left_eye_ratio = self.calculate_eye_ratio(landmarks, self.LEFT_EYE_LANDMARKS)
        right_eye_ratio = self.calculate_eye_ratio(landmarks, self.RIGHT_EYE_LANDMARKS)

        # 양쪽 눈의 EAR 평균 계산
        self.ear = (left_eye_ratio + right_eye_ratio) / 2.0

        # EAR 값이 임계값보다 작으면 눈이 감긴 상태로 간주
        if self.ear < self.ear_threshold:
            self.closed_eyes_frame += 1  # 감긴 상태 프레임 수 증가
        else:
            # 눈을 다시 뜨면 깜빡임으로 간주
            if self.closed_eyes_frame >= 1:
                self.blink_count += 1  # 깜빡임 카운트 증가
            self.closed_eyes_frame = 0  # 감긴 상태 초기화

        return self.blink_count
    
