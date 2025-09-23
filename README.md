# Weapon Detection System for Security Applications

## Project Overview
An AI-powered computer vision system for real-time weapon detection in surveillance footage, designed to enhance public safety by identifying potential security threats.

## Assignment Objectives
- **Primary Weapon Classes**: Firearms (handguns, pistols, rifles)
- **Safety Class**: No weapon scenarios for comparison
- **Advanced Categories**: Improvised weapons (bonus)
- **False Positive Reduction**: Distinguish real weapons from similar objects

## Current Status
- ✅ Dataset collection and preparation
- ✅ COCO-based "no weapon" dataset generation
- ✅ Dataset splitting and standardization
- ✅ Unified dataset creation
- ✅ Professional project structure
- 🔄 Model training and optimization (ready to start)
- ⏳ Video processing pipeline
- ⏳ Real-time detection system

## Dataset Summary
| Dataset Type | Train | Val | Test | Total | Status |
|-------------|-------|-----|------|-------|--------|
| Pistols | 2,865 | 567 | 572 | 4,004 | ✅ Ready |
| Knives | 5,899 | 468 | 127 | 6,494 | ✅ Ready |
| Rifles | 4,325 | 372 | 300 | 4,997 | ✅ Ready |
| No Weapon | 640 | 80 | 80 | 800 | ✅ Ready |
| **Total** | **13,729** | **1,487** | **1,079** | **16,295** | ✅ Ready |

## Installation and Setup

### Quick Start
```bash
# Clone repository
git clone https://github.com/yourusername/weapon-detection.git
cd weapon-detection

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## Usage

### Main Entry Point
```bash
# Show all available commands
python main.py --help

# Train a model
python main.py train --data dataset/unified_weapon_dataset/data.yaml --epochs 50 --name my_model

# Detect weapons in video
python main.py detect --model runs/train/my_model/weights/best.pt --source video.mp4 --show --save-video

# Merge datasets
python main.py merge --config config/merge_config.yaml --output dataset/unified_weapon_dataset

# Download no-weapon dataset
python main.py download --output dataset/no_weapon_coco --max-images 800 --person-only
```

### Direct Script Usage (Legacy)
```bash
# Training
python scripts/train_model.py --data dataset/unified_weapon_dataset/data.yaml --epochs 50

# Video processing
python scripts/process_video.py --model weights/best.pt --source video.mp4 --show

# Dataset preparation
python scripts/download_no_weapon_coco.py --max-images 800 --person-only
python scripts/merge_datasets.py --config config/merge_config.yaml
```

## Project Structure
```
Weapon-Detection2/
├── main.py                    # Main entry point with CLI
├── setup.py                   # Package setup
├── requirements.txt           # Dependencies
├── README.md                  # This file
├── config/                    # Configuration files
│   └── merge_config.yaml     # Dataset merge configuration
├── src/                       # Source code
│   └── weapon_detection/      # Main package
│       ├── __init__.py
│       ├── detector.py        # Main detection class
│       ├── cli/              # Command line interface
│       │   ├── train_command.py
│       │   ├── detect_command.py
│       │   ├── merge_command.py
│       │   └── download_command.py
│       ├── data/             # Data processing
│       │   ├── dataset_merger.py
│       │   ├── coco_downloader.py
│       │   └── dataset_splitter.py
│       ├── models/           # Model training
│       │   ├── trainer.py
│       │   └── yolo_model.py
│       ├── video/            # Video processing
│       │   ├── processor.py
│       │   └── detector.py
│       └── utils/            # Utilities
│           ├── visualization.py
│           ├── metrics.py
│           └── config.py
├── scripts/                   # Legacy scripts (being phased out)
│   ├── train_model.py
│   ├── process_video.py
│   ├── download_no_weapon_coco.py
│   ├── merge_datasets.py
│   └── split_dataset.py
├── dataset/                   # Datasets (excluded from Git)
│   ├── README.md             # Dataset documentation
│   └── .gitkeep              # Keep directory in Git
└── tests/                     # Unit tests
    └── (test files)
```

## Technology Stack
- **Deep Learning**: YOLOv8 (Ultralytics)
- **Computer Vision**: OpenCV
- **Data Processing**: NumPy, Pandas
- **Visualization**: Matplotlib
- **Configuration**: PyYAML
- **CLI**: argparse

## Quick Training Example
```bash
# 1. Ensure datasets are ready (see dataset/README.md)
# 2. Start training
python main.py train \
  --data dataset/unified_weapon_dataset/data.yaml \
  --model yolov8n.pt \
  --epochs 50 \
  --batch 16 \
  --name weapon_detection_v1

# 3. Test trained model
python main.py detect \
  --model runs/train/weapon_detection_v1/weights/best.pt \
  --source 0 \
  --show
```

## Development

### Adding New Features
1. **Models**: Add to `src/weapon_detection/models/`
2. **Data processing**: Add to `src/weapon_detection/data/`
3. **Video processing**: Add to `src/weapon_detection/video/`
4. **CLI commands**: Add to `src/weapon_detection/cli/`

### Testing
```bash
# Install test dependencies
pip install -e .[dev]

# Run tests
pytest tests/

# Code formatting
black src/ tests/
flake8 src/ tests/
```

## Performance Targets
- **Accuracy**: > 95% on test set
- **Real-time Performance**: < 100ms per frame
- **False Positive Rate**: < 2%
- **Recall for Weapons**: > 98%

## Timeline
**Target Completion**: September 21, 2025, 11:59 PM IST

## License
This project is for educational and security research purposes.

## Contributing
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure code passes all tests and linting
5. Submit a pull request