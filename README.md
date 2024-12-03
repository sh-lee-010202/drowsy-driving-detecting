# ImportError: DLL load failed: mediapipe
MediaPipe를 사용 중 ImportError: DLL load failed: 오류가 발생하는 경우가 있습니다. 
이 문제는 Python 환경 설정이나 PC의 하드웨어/소프트웨어 구성에 따라 발생할 수 있으며, 인터넷에 있는 여러 해결법을 적용해도 문제가 해결되지 않을 수 있습니다.

원인
1) Python 환경과 MediaPipe 버전 불일치: MediaPipe가 Python 또는 종속 라이브러리(OpenCV 등)와 호환되지 않는 경우.
2) 누락된 DLL 파일: MediaPipe에서 사용하는 종속 DLL 파일이 누락되거나, 올바른 경로에 있지 않은 경우.
3) 하드웨어/OS 설정 문제: 시스템의 특정 구성(예: Windows 버전, 드라이버 상태)이 MediaPipe와 호환되지 않는 경우.

2)의 경우 mediapipe 버전을 4.7.x로 변경하면 해결될 수 있습니다.
3)의 경우 해결책이 없어서 HW 자체를 변경하여 작업하였습니다.

# library install
이 프로젝트는 spyder를 사용하였으며 anaconda를 통해 라이브러리를 설치하였습니다.
- pip install mediapipe opencv-python
mediapipe와 opencv를 동시에 최신버전으로 설치하여 작업했습니다.
또한 가상 환경은 사용하지 않았습니다.

# 참고 자료
- MediaPipe 공식 문서: [MediaPipe Solutions](https://ai.google.dev/edge/mediapipe/solutions/guide?hl=ko)
- OpenCV 공식 문서: [OpenCV Documentation](https://docs.opencv.org/4.x/index.html)
- EAR 알고리즘(블로그): [OpenCV를 이용한 졸음운전 방지 시스템](https://ultrakid.tistory.com/12)
- Mediapipe Face Mesh(블로그): [MediaPipe 얼굴 그물망(Face Mesh)](https://puleugo.tistory.com/5)

- **도서**:  
  ["이미지 처리 바이블"](https://github.com/Lilcob/imageprocessingbible)  
  저자: 류태선 외 3인 
  출판사: 길벗
  출판 연도: 2024

