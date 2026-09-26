# Vision Target Tracker: Real-Time Object Detection & Centroid Localization

An end-to-end computer vision tracking pipeline built with YOLO object detection, OpenCV, and cloud inference to extract continuous 2D spatial centroids $(u, v)$ from video input.

![Tracker Demo](assets/demo.gif)

---

## 🎯 Overview

This project was built to demonstrate a modular vision tracking system suitable for robotic grasping, visual servoing, and closed-loop control pipelines. The system processes video streams frame-by-frame, performs low-latency bounding box inference, and calculates continuous pixel-space centroid coordinates $(u, v)$ for detected target items.

### Key Features
* **Automated Data Preparation:** Custom frame sampling script (`scripts/extract_frames.py`) with dynamic FPS interval extraction.
* **Trained Vision Model:** Custom-trained YOLO model hosted via Roboflow Serverless Inference.
* **Spatial Centroid Extraction:** Real-time geometric center calculation:
  $$u = x_{\min} + \frac{w}{2}, \quad v = y_{\min} + \frac{h}{2}$$
* **Visual Telemetry Overlay:** Real-time rendering of confidence scores, class labels, bounding boxes, and target centroid markers using OpenCV.
* **Production-Grade Secrets Management:** Cloud credentials decoupled via `.env` and excluded from version control.

---

## 📁 Repository Structure

```text
vision-target-tracker/
├── assets/                  # Media assets (sample outputs, demo.gif)
├── data/
│   └── raw/                 # Extracted frame sequences (git-ignored)
├── scripts/
│   └── extract_frames.py    # Automated dataset generation tool
├── .env.example             # Template for required environment variables
├── .gitignore               # Excludes large media, secrets, and temp caches
├── main.py                  # Primary tracking and centroid inference pipeline
├── requirements.txt         # Project dependencies
└── README.md                # Technical documentation