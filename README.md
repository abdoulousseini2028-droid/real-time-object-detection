# 🚀 Real-Time YouTube Object Detector (YOLOv5)
A high-performance real-time object detection system that processes YouTube streams using YOLOv5 and `yt-dlp`. 

The full video demonstration could not be hosted directly in this repository due to GitHub's **100MB file size limit**.

Below is a screenshot of the detection in action.

<img width="858" height="479" alt="Screenshot 2026-01-09 at 17 54 29" src="https://github.com/user-attachments/assets/b81615f3-1c86-46df-8e36-f3113769ec44" />


##  Features
- **Real-Time Stream Processing:** Extracts YouTube frames directly without downloading.
- **YOLOv5 Inference:** Optimized for both CPU and CUDA-enabled GPUs.
- **Automated Recording:** Automatically saves processed footage for analysis.

## Quick Start
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
