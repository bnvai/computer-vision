# 👁️ Computer Vision Projects
[![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](#)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](#)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-00FFFF?style=for-the-badge&logo=ultralytics&logoColor=black)](#)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-007fba?style=for-the-badge&logo=google&logoColor=white)](#)

A comprehensive repository for **Computer Vision** applications, featuring real-time object detection, traffic analysis, and human behavior monitoring using state-of-the-art frameworks.

---

## 🚀 Featured Modules

### 🚦 Traffic Detection & Management
* **Engine:** YOLOv8 (You Only Look Once)
* **Source:** `automotive/traffic-detection.py`
* **Capabilities:** Real-time vehicle detection, tracking, and classification (cars, trucks, motorcycles).
* **Data:** Configured to process high-definition streams from `dataset/traffic.mp4`.

### 💤 Driver Drowsiness Detection
* **Engine:** MediaPipe Face Landmarker / OpenCV
* **Source:** `automotive/drowsiness_detection.ipynb`
* **Capabilities:** Monitors Eye Aspect Ratio (EAR) and mouth opening (Yawning) to trigger fatigue alerts.
* **Logic:** Real-time facial landmark mapping for high-precision safety monitoring.

---

## 🛠️ Getting Started

### 1. Installation
Clone the repository and install the necessary dependencies:

```bash
# Clone the repo
git clone [https://github.com/your-username/computer-vision-projects.git](https://github.com/your-username/computer-vision-projects.git)
cd computer-vision-projects

# Install requirements
pip install opencv-python ultralytics mediapipe numpy