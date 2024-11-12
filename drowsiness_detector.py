# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 22:43:42 2024

@author: tjdgk
"""

class DrowsinessDetector:
    def __init__(self):
        self.calibrated_pitch = 0
        self.calibrated_yaw = 0
        self.calibration_count = 50
        self.calibration_complete = False

    def calibrate(self, pitch, yaw):
        if self.calibration_count > 0:
            self.calibrated_pitch = pitch
            self.calibrated_yaw = yaw
            self.calibration_count -= 1
            return "Calibrating..."
        else:
            self.calibration_complete = True
            return "Calibration Complete"

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