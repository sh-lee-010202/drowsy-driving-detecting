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
        
        #알람
        self.last_state_time = None  # 상태가 변경된 시간을 추적
        self.state_duration = 0  # 상태가 지속된 시간
        self.alert_triggered = False  # 경고 소리가 이미 나온 경우
        
        # Pygame 초기화 및 소리 로딩
        pygame.mixer.init()
        self.alert_sound = pygame.mixer.Sound('./source/alert.wav')  # 경고 소리 파일 경로 설정

    def calibrate(self, pitch, yaw, ear):
        # 캘리브레이션 단계
        if self.calibration_count > 0:
            self.calibrated_pitch = pitch
            self.calibrated_yaw = yaw
            self.ear_calibration_values.append(ear)
            self.calibration_count -= 1
            return "Calibrating..."
        else:
            if not self.calibration_complete:
                self.calibration_complete = True
                # EAR 값의 평균을 계산하고 ear_threshold를 업데이트
                average_ear = sum(self.ear_calibration_values) / len(self.ear_calibration_values)
                self.face_detector.ear_threshold = average_ear * 0.8  # 80% 수준으로 설정 (사용자 조정 가능)
                return "Calibration Complete - EAR Threshold Set"
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

    def detect_drowsiness(self, pitch, yaw):
        if pitch > self.calibrated_pitch + 0.02:
            return "Drowsy"
        elif yaw > self.calibrated_yaw + 0.02 or yaw < self.calibrated_yaw - 0.02:
            return "Distracted"
        return "Driving - Focused"
    
    def check_and_play_alert(self, state):
        current_time = time.time()

        if state == "Distracted" or state == "Drowsy":
            if self.last_state_time is None:
                self.last_state_time = current_time  # 상태가 처음 발생한 시간 기록
                self.state_duration = 0
                self.alert_triggered = False
            else:
                self.state_duration = current_time - self.last_state_time  # 상태 지속 시간 계산
                if self.state_duration >= 3 and not self.alert_triggered:
                    if not pygame.mixer.get_busy():  # 소리가 재생 중인지 확인
                        print("alarm")
                        self.alert_sound.play(loops=-1)  # 소리 3번 반복 재생
                    self.alert_triggered = True
        else:
            # 상태가 변하면 알림 소리 중지
            if pygame.mixer.get_busy():
                pygame.mixer.stop()  # 소리 중지
            self.last_state_time = None  # 상태가 변하면 시간 리셋
            self.state_duration = 0
            self.alert_triggered = False

    def process(self, pitch, yaw):
        state = self.detect_drowsiness(pitch, yaw)
        print(state)
        self.check_and_play_alert(state)
        return {"state": state, "duration": self.state_duration}