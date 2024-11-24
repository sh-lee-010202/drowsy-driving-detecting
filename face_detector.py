# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 22:42:49 2024

@author: tjdgk
"""

import mediapipe as mp
import numpy as np

class FaceDetector:
    def __init__(self, ear_threshold=0.2):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)
        self.LEFT_EYE_LANDMARKS = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144]
        self.NOSE_TIP = 1
        self.MOUTH_LANDMARKS = [61, 291, 78, 308, 13, 14]
        self.ear_threshold = ear_threshold
        self.blink_count = 0
        self.closed_eyes_frame = 0
        self.ear = 0

    def calculate_eye_ratio(self, landmarks, eye_landmarks):
        left_right = np.linalg.norm(np.array([landmarks[eye_landmarks[0]].x, landmarks[eye_landmarks[0]].y]) -
                                    np.array([landmarks[eye_landmarks[3]].x, landmarks[eye_landmarks[3]].y]))
        top_bottom_1 = np.linalg.norm(np.array([landmarks[eye_landmarks[1]].x, landmarks[eye_landmarks[1]].y]) -
                                      np.array([landmarks[eye_landmarks[5]].x, landmarks[eye_landmarks[5]].y]))
        top_bottom_2 = np.linalg.norm(np.array([landmarks[eye_landmarks[2]].x, landmarks[eye_landmarks[2]].y]) -
                                      np.array([landmarks[eye_landmarks[4]].x, landmarks[eye_landmarks[4]].y]))
        eye_ratio = (top_bottom_1 + top_bottom_2) / (2.0 * left_right)
        return eye_ratio

    def detect_blink(self, landmarks):
        left_eye_ratio = self.calculate_eye_ratio(landmarks, self.LEFT_EYE_LANDMARKS)
        right_eye_ratio = self.calculate_eye_ratio(landmarks, self.RIGHT_EYE_LANDMARKS)
        self.ear = (left_eye_ratio + right_eye_ratio) / 2.0

        if self.ear < self.ear_threshold:
            self.closed_eyes_frame += 1
        else:
            if self.closed_eyes_frame >= 1:
                self.blink_count += 1
            self.closed_eyes_frame = 0

        return self.blink_count
    
