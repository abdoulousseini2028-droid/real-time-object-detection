# 🚀 Real-Time YouTube Object Detector (YOLOv5)
A high-performance real-time object detection system that processes YouTube streams using YOLOv5 and `yt-dlp`. 

![Project Demo](github_demo.gif)

## 🛠️ Features
- **Real-Time Stream Processing:** Extracts YouTube frames directly without downloading.
- **YOLOv5 Inference:** Optimized for both CPU and CUDA-enabled GPUs.
- **Automated Recording:** Automatically saves processed footage for analysis.

## 🚀 Quick Start
### 1. Clone the repo
```bash
git clone https://github.com/abdoulousseini2028-droid/real-time-object-detection
cd yolo_detection

# Install dependencies
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

dependencies: ultralytics/yolov5: Core detection model
yt-dlp: Stream extraction
opencv-python

# Run
python main.py
