# Driver Drowsiness Safety Monitoring System

A real-time computer vision project that detects driver drowsiness using OpenCV, Machine Learning, and Facial Landmark Detection. The system monitors the driver's eyes through a webcam feed, calculates the Eye Aspect Ratio (EAR), and triggers an alert if signs of drowsiness are detected to improve road safety.
# Features
Real-time driver monitoring using a webcam.
Facial landmark detection with Dlib.
Eye Aspect Ratio (EAR) calculation for blink and drowsiness detection.
Alerts the driver with sound when drowsiness is detected.
Lightweight, fast, and works on CPU (no GPU required).
# Tech Stack

Programming Language: Python

Libraries/Frameworks:

OpenCV (video processing, face/eye detection)

Dlib (facial landmarks)

NumPy (mathematical operations)

Scikit-learn (machine learning & classification model)

PyGame (alert system)
# Methodology

Face Detection → Detect driver’s face using OpenCV/Dlib.

Facial Landmark Detection → Extract eye coordinates.

Eye Aspect Ratio (EAR) → Calculate EAR to detect eye closure.

Thresholding → If EAR < 0.25 for continuous frames → driver is drowsy.

Alert System → Sound alarm to wake the driver.
