# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 22:43:42 2024

@author: tjdgk
"""

import time
import pygame  # Pygame 라이브러리로 소리 출력

class DrowsinessDetector:
    def __init__(self, face_detector):
        """
        DrowsinessDetector 초기화
        :param face_detector: 얼굴 감지를 위한 FaceDetector 객체
        """
        # 머리의 초기 캘리브레이션 값
        self.calibrated_pitch = 0
        self.calibrated_yaw = 0
        self.calibration_count = 50  # 캘리브레이션에 필요한 프레임 수
        self.calibration_complete = False  # 캘리브레이션 완료 여부
        self.ear_calibration_values = []  # 캘리브레이션 동안 EAR 값 저장
        self.face_detector = face_detector
        self.blink_count = self.face_detector.blink_count  # 초기 깜빡임 횟수
        self.start_time = time.time()  # 시작 시간 기록

        # 어깨 위치 초기화
        self.calibrated_shoulder_x = 0
        self.calibrated_shoulder_y = 0
        self.shoulder_threshold = 20  # 어깨 움직임 감지 임계값 (픽셀 기준)

        # 알람 관련 변수
        self.last_state_time = None  # 상태 변경 시점 기록
        self.state_duration = 0  # 현재 상태 지속 시간
        self.alert_triggered = False  # 경고 소리가 이미 울렸는지 여부

        # Pygame 초기화 및 경고음 설정
        pygame.mixer.init()
        self.alert_sound = pygame.mixer.Sound('./source/alert.wav')  # 경고음 파일 경로

    def calibrate(self, pitch, yaw, ear, shoulder_x, shoulder_y):
        """
        캘리브레이션 단계 - 머리의 초기 위치와 EAR 값을 설정
        :param pitch: 머리의 상하 움직임 값
        :param yaw: 머리의 좌우 움직임 값
        :param ear: EAR (Eye Aspect Ratio) 값
        :param shoulder_x: 어깨의 x 좌표
        :param shoulder_y: 어깨의 y 좌표
        :return: 캘리브레이션 상태 메시지
        """
        if self.calibration_count > 0:
            # 캘리브레이션 중 값 기록
            self.calibrated_pitch = pitch
            self.calibrated_yaw = yaw
            self.ear_calibration_values.append(ear)
            self.calibrated_shoulder_x = shoulder_x
            self.calibrated_shoulder_y = shoulder_y
            self.calibration_count -= 1
            return "Calibrating..."
        else:
            if not self.calibration_complete:
                # 캘리브레이션 완료 처리
                self.calibration_complete = True
                # EAR 임계값을 EAR 값의 평균에서 계산
                average_ear = sum(self.ear_calibration_values) / len(self.ear_calibration_values)
                self.face_detector.ear_threshold = average_ear * 0.8  # 80% 수준으로 설정
                return "Calibration Complete - EAR and Shoulder Threshold Set"
            return "Calibration Already Complete"

    def detect_head_movement(self, pitch, yaw):
        """
        머리의 움직임을 감지하여 상태 반환
        :param pitch: 머리의 상하 움직임 값
        :param yaw: 머리의 좌우 움직임 값
        :return: 머리의 상태 (예: Looking Left, Looking Right 등)
        """
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
        졸음 및 주의 산만 상태를 감지
        :param pitch: 머리의 상하 움직임 값
        :param yaw: 머리의 좌우 움직임 값
        :param ear: EAR (Eye Aspect Ratio) 값
        :param blink: 평균 깜빡임 횟수
        :param shoulder_x: 어깨의 x 좌표
        :param shoulder_y: 어깨의 y 좌표
        :return: 현재 상태 (예: Drowsy, Distracted 등)
        """
        # 어깨 움직임 감지
        shoulder_movement = abs(shoulder_x - self.calibrated_shoulder_x) + abs(shoulder_y - self.calibrated_shoulder_y)

        # 졸음 및 산만 상태 판단
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
        """
        상태에 따라 알람을 재생 또는 중지
        :param state: 감지된 상태
        """
        current_time = time.time()

        # 상태 변경 시 초기화
        if state != getattr(self, "current_state", None):
            self.current_state = state  # 상태 업데이트
            self.last_state_time = current_time  # 상태 변경 시간 기록
            self.state_duration = 0  # 상태 지속 시간 초기화
            self.alert_triggered = False 

        # 상태별 알람 처리
        if state == "Drowsy" or state == "Distracted":
            # 졸음 및 주의 산만 상태 처리
            self.state_duration = current_time - self.last_state_time
            if self.state_duration >= 3 and not self.alert_triggered:  # 3초 이상 지속 시 알람 재생
                if not pygame.mixer.get_busy():  # 현재 알람이 재생 중이 아니면
                    self.alert_sound.play(loops=-1)  # 무한 반복 재생
                self.alert_triggered = True

        elif state == "Caution":
            # 주의 상태에서 3번 알람 재생
            if not self.alert_triggered and not pygame.mixer.get_busy(): # 현재 알람이 재생 중이 아니면
                self.alert_sound.play(loops=2)  # 소리 3번 재생
                self.alert_triggered = True
            elif self.alert_triggered and not pygame.mixer.get_busy():
                # 소리 재생 완료 후 초기화
                self.alert_triggered = False
                self.last_state_time = current_time

        else:
            # 안전 상태에서 알람 중지
            if pygame.mixer.get_busy():
                pygame.mixer.stop()  # 소리 중지
            # 상태 변경에 필요한 시간 변수 초기화
            self.last_state_time = None 
            self.state_duration = 0
            self.alert_triggered = False

    def process(self, pitch, yaw, shoulder_x, shoulder_y):
        """
        상태를 감지하고 알람을 제어
        :param pitch: 머리의 상하 움직임 값
        :param yaw: 머리의 좌우 움직임 값
        :param shoulder_x: 어깨의 x 좌표
        :param shoulder_y: 어깨의 y 좌표
        :return: 상태 및 상태 지속 시간
        """
        current_time = time.time()
        elapsed_time = current_time - self.start_time  # 경과 시간 계산
        self.blink_count = self.face_detector.blink_count  # 깜빡임 횟수 업데이트
        self.average_blink = (self.blink_count / elapsed_time) * 60  # 평균 깜빡임 횟수 계산

        # 상태 감지 및 알람 처리
        state = self.detect_drowsiness(
            pitch, yaw, self.face_detector.ear, self.average_blink, shoulder_x, shoulder_y
        )
        print(state)  # 상태 출력(디버깅용)
        self.check_and_play_alert(state)  # 알람 처리
        return self.average_blink, state, self.state_duration  # 상태 및 상태 지속 시간 반환
    
    

