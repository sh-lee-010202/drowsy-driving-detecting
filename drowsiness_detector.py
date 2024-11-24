# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 22:43:42 2024

@author: tjdgk
"""

import time
import pygame  # Pygame 라이브러리로 소리 출력

class DrowsinessDetector:
    def __init__(self, face_detector):
        self.calibrated_pitch = 0
        self.calibrated_yaw = 0
        self.calibration_count = 50
        self.calibration_complete = False
        self.ear_calibration_values = []
        self.face_detector = face_detector
        self.blink_count = self.face_detector.blink_count
        self.start_time = time.time()

        # 어깨 위치 초기화
        self.calibrated_shoulder_x = 0
        self.calibrated_shoulder_y = 0
        self.shoulder_threshold = 20  # 어깨 움직임 임계값 (픽셀 기준)

        # 알람
        self.last_state_time = None  # 상태가 변경된 시간을 추적
        self.state_duration = 0  # 상태가 지속된 시간
        self.alert_triggered = False  # 경고 소리가 이미 나온 경우

        # Pygame 초기화 및 소리 로딩
        pygame.mixer.init()
        self.alert_sound = pygame.mixer.Sound('./source/alert.wav')  # 졸음/분산 알람 소리

    def calibrate(self, pitch, yaw, ear, shoulder_x, shoulder_y):
        """
        캘리브레이션 단계 - 머리와 어깨 위치를 캘리브레이션
        """
        if self.calibration_count > 0:
            self.calibrated_pitch = pitch
            self.calibrated_yaw = yaw
            self.ear_calibration_values.append(ear)
            self.calibrated_shoulder_x = shoulder_x
            self.calibrated_shoulder_y = shoulder_y
            self.calibration_count -= 1
            return "Calibrating..."
        else:
            if not self.calibration_complete:
                self.calibration_complete = True
                # EAR 값의 평균을 계산하고 ear_threshold를 업데이트
                average_ear = sum(self.ear_calibration_values) / len(self.ear_calibration_values)
                self.face_detector.ear_threshold = average_ear * 0.8  # 80% 수준으로 설정 (사용자 조정 가능)
                return "Calibration Complete - EAR and Shoulder Threshold Set"
            return "Calibration Already Complete"

    def detect_head_movement(self, pitch, yaw):
        if yaw > self.calibrated_yaw + 0.02:
            return "Looking Left"
        elif yaw < self.calibrated_yaw - 0.02:
            return "Looking Right"
        elif pitch > self.calibrated_pitch + 0.02:
            return "Looking Down"
        elif pitch < self.calibrated_pitch - 0.02:
            return "Looking Up"
        return "Centered"

    def detect_drowsiness(self, pitch, yaw, ear, blink, shoulder_x, shoulder_y):
        """
        졸음 상태 및 어깨 움직임을 감지
        """
        # 어깨 위치와 캘리브레이션 기준 비교
        shoulder_movement = abs(shoulder_x - self.calibrated_shoulder_x) + abs(shoulder_y - self.calibrated_shoulder_y)
        
        
        # 기존 졸음 및 주의 산만 상태 감지
        if pitch > self.calibrated_pitch + 0.02:
            return "Drowsy"
        elif ear < self.face_detector.ear_threshold:
            return "Drowsy"
        elif blink > 30:
            return "Caution"
        elif yaw > self.calibrated_yaw + 0.02 or yaw < self.calibrated_yaw - 0.02:
            return "Distracted"
        elif shoulder_movement > self.shoulder_threshold:
            return "Distracted"
        return "Driving - Focused"

    def check_and_play_alert(self, state):
        current_time = time.time()
        
        # 상태가 변경되었을 때만 처리
        if state != getattr(self, "current_state", None):
            self.current_state = state  # 현재 상태 업데이트
            self.last_state_time = current_time  # 상태 변경 시간 기록
            self.state_duration = 0  # 상태 지속 시간 초기화
            self.alert_triggered = False
    
        # 상태별 처리
        if state == "Drowsy" or state == "Distracted":
            # "Drowsy" 상태 처리
            self.state_duration = current_time - self.last_state_time
            if self.state_duration >= 3 and not self.alert_triggered:
                if not pygame.mixer.get_busy():  # 소리가 재생 중인지 확인
                    self.alert_sound.play(loops=-1)  # 소리 무한 반복 재생
                self.alert_triggered = True
    
    
        elif state == "Caution":
            # "Caution" 상태에서 3번 소리를 재생
            if not self.alert_triggered:
                self.alert_sound.play(loops=2)  # 소리 3번 재생 (1번 + 추가 2번)
                self.alert_triggered = True
            elif not pygame.mixer.get_busy():  # 소리가 모두 재생 완료되면 상태 리셋
                self.last_state_time = None
                self.state_duration = 0
                self.alert_triggered = False

        else:
            # 상태가 "Drowsy", "Distracted", "Caution"이 아닐 때
            if pygame.mixer.get_busy():
                pygame.mixer.stop()  # 소리 중지
            self.last_state_time = None
            self.state_duration = 0
            self.alert_triggered = False

    def process(self, pitch, yaw, shoulder_x, shoulder_y):
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        self.blink_count = self.face_detector.blink_count
        self.average_blink = (self.blink_count / elapsed_time) * 60

        # 상태 감지 (어깨 움직임 포함)
        state = self.detect_drowsiness(
            pitch, yaw, self.face_detector.ear, self.average_blink, shoulder_x, shoulder_y
        )
        print(state)
        self.check_and_play_alert(state)
        return state, self.state_duration

