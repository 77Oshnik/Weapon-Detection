# Weapon Detection System for Security Applications

## Project Overview
An AI-powered computer vision system for real-time weapon detection in surveillance footage, designed to enhance public safety by identifying potential security threats before incidents occur.

## 🚀 Quick Start - Video Detection

### Method 1: Simple Video Detection (Recommended)
```bash
# Process video file with trained model
python detect_video.py --model models/weights/best.pt --source video.mp4 --save-video --show




# Live webcam detection
python detect_video.py --model models/weights/best.pt --source 0 --show

# Process RTSP camera stream
python detect_video.py --model models/weights/best.pt --source "rtsp://camera_url" --show
```

### Method 2: Interactive Demo
```bash
# Run interactive demo with menu options
python demo.py

# Direct video processing
python demo.py --video video.mp4

# Live webcam demo
python demo.py --webcam

# Test on sample images
python demo.py --test
```

### Method 3: Advanced CLI Interface
```bash
# Full featured detection with logging
python main.py detect --model models/weights/best.pt --source video.mp4 --save-video --save-log --show

# Batch processing with custom settings
python main.py detect --model models/weights/best.pt --source video.mp4 --conf 0.3 --output results/batch1
```

## 📁 Video Detection Features

### Real-time Threat Detection
- **Threat Levels**: SAFE (Green), LOW (Yellow), MEDIUM (Orange), HIGH (Red)
- **Automatic Alerts**: Console alerts for weapon detections
- **Visual Indicators**: Color-coded bounding boxes and threat level display

### Output Options
- **Live Display**: Real-time video window with annotations
- **Video Recording**: Save processed video with detections
- **Screenshots**: Manual screenshot capture (press 's')
- **Detection Logs**: JSON format with timestamps and coordinates

### Threat Assessment
- **HIGH**: Firearms with >70% confidence
- **MEDIUM**: Firearms >50% confidence or knives >60% confidence
- **LOW**: Any weapon detection below medium thresholds
- **SAFE**: No weapons detected

## 💻 Installation and Setup

### Prerequisites
```bash
# Python 3.8+ required
python --version

# Install dependencies
pip install -r requirements.txt

# For GPU support (optional)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Quick Installation
```bash
# Clone repository
git clone https://github.com/yourusername/weapon-detection.git
cd weapon-detection

# Create virtual environment
python -m venv .venv
.venv\\Scripts\\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install package
pip install -e .
```

## 🎬 Usage Examples

### Basic Video Processing
```bash
# Process video with default settings
python detect_video.py --source video.mp4 --show

# Save output video
python detect_video.py --source video.mp4 --save-video --output results/

# Custom confidence threshold
python detect_video.py --source video.mp4 --conf 0.3 --show
```

### Live Detection
```bash
# Webcam detection
python detect_video.py --source 0 --show

# IP camera (replace with your camera URL)
python detect_video.py --source "rtsp://admin:password@192.168.1.100/stream" --show
```

### Batch Processing
```bash
# Process multiple videos
for video in videos/*.mp4; do
    python detect_video.py --source "$video" --save-video --output "results/$(basename "$video" .mp4)/"
done
```

### Advanced Features
```bash
# Full logging and analysis
python main.py detect \\
    --model models/weights/best.pt \\
    --source video.mp4 \\
    --save-video \\
    --save-log \\
    --output results/analysis \\
    --conf 0.25 \\
    --show

# Performance testing
python demo.py --video test_video.mp4  # Interactive with statistics
```
## 🔧 Technical Architecture

### Core Components
```
weapon_detection/
├── detector.py          # Main detection engine
├── video/              
│   ├── processor.py    # Video processing pipeline
│   └── detector.py     # Video-specific detection
├── utils/
│   ├── visualization.py # Drawing and annotations
│   ├── metrics.py      # Performance metrics
│   └── config.py       # Configuration management
└── cli/                # Command-line interface
    ├── detect_command.py
    ├── train_command.py
    └── ...
```

### Video Processing Pipeline
1. **Input Handling**: Video file, webcam, or stream
2. **Frame Extraction**: Real-time frame processing
3. **Detection**: YOLOv8 inference on each frame
4. **Post-processing**: Threat level assessment
5. **Visualization**: Bounding boxes and alerts
6. **Output**: Display, recording, logging

## 📱 Screenshots and Media

Below are real outputs from this repository under the `images/` folder.

### Inference Samples

![Validation Predictions](images/val_batch0_pred.jpg)
![Validation Labels](images/val_batch2_labels.jpg)

### Sample Frames

![Sample Frame 1](images/sample.png)
![Sample Frame 2](images/sample2.png)

### Training/Validation Metrics

![Confusion Matrix](images/confusion_matrix.png)
![PR Curve](images/BoxPR_curve.png)
![R Curve](images/BoxR_curve.png)

### Demo Video
https://github.com/user-attachments/assets/b9a71f12-34ad-45e9-bae1-46a9821d7032

### Sample Videos
Place test videos in `test_videos/` directory and run:
```bash
python demo.py --video test_videos/sample.mp4
```
