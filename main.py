# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 22:45:26 2024

@author: tjdgk
"""

from face_detector import FaceDetector
from drowsiness_detector import DrowsinessDetector
from video_stream import VideoStream

face_detector = FaceDetector()
drowsiness_detector = DrowsinessDetector()
video_stream = VideoStream(face_detector, drowsiness_detector)
video_stream.start()