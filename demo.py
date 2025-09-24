#!/usr/bin/env python3
"""
Weapon Detection System Demo
AISOLO Technologies Pvt. Ltd.

This script demonstrates the weapon detection system capabilities including:
- Video file processing
- Real-time webcam detection  
- Model performance metrics
- Threat detection alerts

Usage:
    python demo.py                          # Interactive demo
    python demo.py --video video.mp4        # Process specific video
    python demo.py --webcam                 # Use webcam
    python demo.py --test                   # Test model on sample images
"""

import argparse
import cv2
import time
import json
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO
import numpy as np


class WeaponDetectionDemo:
    """Demo class for weapon detection system."""
    
    def __init__(self, model_path="models/weights/best.pt"):
        """Initialize demo with trained model."""
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        print(f"[DEMO] Loading weapon detection model...")
        self.model = YOLO(model_path)
        
        # Configuration
        self.conf_threshold = 0.25
        self.class_names = {0: 'pistol', 1: 'rifle', 2: 'knife', 3: 'no_weapon'}
        self.weapon_classes = ['pistol', 'rifle', 'knife']
        
        # Colors for visualization
        self.colors = {
            'pistol': (0, 0, 255),      # Red
            'rifle': (0, 0, 255),       # Red  
            'knife': (0, 165, 255),     # Orange
            'no_weapon': (0, 255, 0)    # Green
        }
        
        self.threat_colors = {
            'SAFE': (0, 255, 0),
            'LOW': (0, 255, 255),
            'MEDIUM': (0, 165, 255),
            'HIGH': (0, 0, 255)
        }
        
        print(f"[DEMO] Model loaded successfully!")
        print(f"[DEMO] Weapon classes: {', '.join(self.weapon_classes)}")
        print(f"[DEMO] Confidence threshold: {self.conf_threshold}")
    
    def detect_frame(self, frame):
        """Detect weapons in a single frame."""
        results = self.model(frame, conf=self.conf_threshold, verbose=False)
        
        detections = []
        threat_level = "SAFE"
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    conf = float(box.conf[0])
                    class_id = int(box.cls[0])
                    
                    if class_id in self.class_names:
                        class_name = self.class_names[class_id]
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        
                        detection = {
                            'class': class_name,
                            'confidence': conf,
                            'bbox': [int(x1), int(y1), int(x2), int(y2)]
                        }
                        detections.append(detection)
                        
                        # Determine threat level
                        if class_name in self.weapon_classes:
                            if class_name in ['pistol', 'rifle'] and conf > 0.7:
                                threat_level = "HIGH"
                            elif class_name in ['pistol', 'rifle'] and conf > 0.5:
                                threat_level = "MEDIUM" if threat_level in ["SAFE", "LOW"] else threat_level
                            elif class_name == 'knife' and conf > 0.6:
                                threat_level = "MEDIUM" if threat_level in ["SAFE", "LOW"] else threat_level
                            elif threat_level == "SAFE":
                                threat_level = "LOW"
        
        return detections, threat_level
    
    def annotate_frame(self, frame, detections, threat_level, frame_num=None):
        """Add annotations to frame."""
        annotated = frame.copy()
        
        # Draw detections
        for det in detections:
            class_name = det['class']
            conf = det['confidence']
            x1, y1, x2, y2 = det['bbox']
            
            color = self.colors.get(class_name, (255, 255, 255))
            
            # Bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Label
            label = f"{class_name}: {conf:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(annotated, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(annotated, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Threat level indicator
        threat_color = self.threat_colors.get(threat_level, (255, 255, 255))
        cv2.rectangle(annotated, (10, 10), (220, 50), threat_color, -1)
        cv2.putText(annotated, f"THREAT: {threat_level}", (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # Frame info
        height = annotated.shape[0]
        info_text = f"Frame: {frame_num or 'N/A'} | {datetime.now().strftime('%H:%M:%S')}"
        cv2.putText(annotated, info_text, (10, height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Detection count
        weapon_count = len([d for d in detections if d['class'] in self.weapon_classes])
        if weapon_count > 0:
            count_text = f"Weapons detected: {weapon_count}"
            cv2.putText(annotated, count_text, (10, height - 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        return annotated
    
    def process_video(self, video_path, output_dir="results"):
        """Process video file for weapon detection."""
        print(f"\\n[DEMO] Processing video: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open video: {video_path}")
            return
        
        # Video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"[DEMO] Video: {width}x{height} @ {fps}fps, {total_frames} frames")
        
        # Setup output
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / f"detected_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        # Processing
        frame_count = 0
        weapon_detections = 0
        threat_frames = 0
        detection_log = []
        start_time = time.time()
        
        print(f"[DEMO] Processing frames... (Press 'q' to stop)")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Detect weapons
                detections, threat_level = self.detect_frame(frame)
                annotated_frame = self.annotate_frame(frame, detections, threat_level, frame_count)
                
                # Log results
                weapon_count = len([d for d in detections if d['class'] in self.weapon_classes])
                if weapon_count > 0:
                    weapon_detections += weapon_count
                    threat_frames += 1
                    
                    detection_log.append({
                        'frame': frame_count,
                        'timestamp': time.time(),
                        'threat_level': threat_level,
                        'weapons': [d['class'] for d in detections if d['class'] in self.weapon_classes],
                        'detections': detections
                    })
                    
                    print(f"[ALERT] Frame {frame_count}: {threat_level} threat - {weapon_count} weapon(s)")
                
                # Save and display
                out_writer.write(annotated_frame)
                cv2.imshow('Weapon Detection Demo', annotated_frame)
                
                # Progress
                if frame_count % 100 == 0:
                    progress = (frame_count / total_frames) * 100
                    print(f"[DEMO] Progress: {progress:.1f}% ({frame_count}/{total_frames})")
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        except KeyboardInterrupt:
            print("\\n[DEMO] Interrupted by user")
        
        finally:
            cap.release()
            out_writer.release()
            cv2.destroyAllWindows()
            
            # Results summary
            processing_time = time.time() - start_time
            
            summary = {
                'video_path': video_path,
                'frames_processed': frame_count,
                'total_frames': total_frames,
                'processing_time_seconds': processing_time,
                'average_fps': frame_count / processing_time,
                'weapon_detections': weapon_detections,
                'threat_frames': threat_frames,
                'threat_percentage': (threat_frames / max(frame_count, 1)) * 100,
                'output_video': str(output_path)
            }
            
            print(f"\\n[DEMO] Video processing complete!")
            print(f"  Output saved: {output_path}")
            print(f"  Frames processed: {frame_count}/{total_frames}")
            print(f"  Processing time: {processing_time:.2f} seconds")
            print(f"  Average FPS: {summary['average_fps']:.2f}")
            print(f"  Weapon detections: {weapon_detections}")
            print(f"  Threat frames: {threat_frames} ({summary['threat_percentage']:.1f}%)")
            
            # Save results
            results_path = output_dir / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(results_path, 'w') as f:
                json.dump({
                    'summary': summary,
                    'detections': detection_log
                }, f, indent=2, default=str)
            
            print(f"  Results saved: {results_path}")
    
    def webcam_demo(self):
        """Real-time webcam demonstration."""
        print(f"\\n[DEMO] Starting webcam demonstration...")
        print(f"[DEMO] Press 'q' to quit, 's' for screenshot")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print(f"[ERROR] Cannot access webcam")
            return
        
        frame_count = 0
        detection_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Detect and annotate
                detections, threat_level = self.detect_frame(frame)
                annotated_frame = self.annotate_frame(frame, detections, threat_level, frame_count)
                
                # Count weapons
                weapon_count = len([d for d in detections if d['class'] in self.weapon_classes])
                if weapon_count > 0:
                    detection_count += 1
                    weapons = [d['class'] for d in detections if d['class'] in self.weapon_classes]
                    print(f"[ALERT] Webcam frame {frame_count}: {threat_level} - {', '.join(weapons)}")
                
                cv2.imshow('Weapon Detection - Live Demo', annotated_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    screenshot_path = f"webcam_screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                    cv2.imwrite(screenshot_path, annotated_frame)
                    print(f"[DEMO] Screenshot saved: {screenshot_path}")
        
        except KeyboardInterrupt:
            print("\\n[DEMO] Webcam demo stopped")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            print(f"\\n[DEMO] Webcam session complete!")
            print(f"  Frames processed: {frame_count}")
            print(f"  Detection alerts: {detection_count}")
    
    def test_images(self, test_dir="dataset/test_images"):
        """Test model on sample images."""
        print(f"\\n[DEMO] Testing model on sample images...")
        
        test_path = Path(test_dir)
        if not test_path.exists():
            print(f"[DEMO] Test directory not found: {test_dir}")
            print(f"[DEMO] You can add test images to this directory")
            return
        
        image_files = list(test_path.glob("*.jpg")) + list(test_path.glob("*.png"))
        if not image_files:
            print(f"[DEMO] No test images found in {test_dir}")
            return
        
        print(f"[DEMO] Found {len(image_files)} test images")
        
        for img_path in image_files:
            print(f"\\n[DEMO] Testing: {img_path.name}")
            
            # Load and process image
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            
            detections, threat_level = self.detect_frame(image)
            annotated = self.annotate_frame(image, detections, threat_level)
            
            # Show results
            weapon_count = len([d for d in detections if d['class'] in self.weapon_classes])
            if weapon_count > 0:
                weapons = [d['class'] for d in detections if d['class'] in self.weapon_classes]
                print(f"  -> {threat_level} threat: {', '.join(weapons)}")
            else:
                print(f"  -> No weapons detected")
            
            cv2.imshow(f'Test Image: {img_path.name}', annotated)
            print(f"[DEMO] Press any key to continue, 'q' to skip remaining...")
            
            key = cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            if key == ord('q'):
                break
    
    def interactive_menu(self):
        """Interactive demo menu."""
        print(f"\\n" + "="*60)
        print(f"WEAPON DETECTION SYSTEM DEMO")
        print(f"AISOLO Technologies Pvt. Ltd.")
        print(f"="*60)
        print(f"Model: Loaded and ready")
        print(f"Classes: {', '.join(self.weapon_classes)}")
        print(f"="*60)
        
        while True:
            print(f"\\nSelect demo option:")
            print(f"1. Process video file")
            print(f"2. Live webcam demo")
            print(f"3. Test on sample images")
            print(f"4. Model information")
            print(f"5. Exit")
            
            choice = input(f"\\nEnter choice (1-5): ").strip()
            
            if choice == "1":
                video_path = input("Enter video file path: ").strip()
                if video_path and Path(video_path).exists():
                    self.process_video(video_path)
                else:
                    print(f"[ERROR] Video file not found: {video_path}")
            
            elif choice == "2":
                self.webcam_demo()
            
            elif choice == "3":
                self.test_images()
            
            elif choice == "4":
                self.show_model_info()
            
            elif choice == "5":
                print(f"[DEMO] Thank you for trying the weapon detection system!")
                break
            
            else:
                print(f"[ERROR] Invalid choice. Please select 1-5.")
    
    def show_model_info(self):
        """Display model information."""
        print(f"\\n" + "="*50)
        print(f"MODEL INFORMATION")
        print(f"="*50)
        print(f"Architecture: YOLOv8")
        print(f"Classes: {len(self.class_names)}")
        for i, name in self.class_names.items():
            print(f"  {i}: {name}")
        print(f"Weapon Classes: {', '.join(self.weapon_classes)}")
        print(f"Confidence Threshold: {self.conf_threshold}")
        print(f"Input Size: 640x640 (auto-resized)")
        print(f"="*50)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Weapon Detection System Demo")
    parser.add_argument("--model", type=str, default="models/weights/best.pt",
                        help="Path to trained model")
    parser.add_argument("--video", type=str, help="Process specific video file")
    parser.add_argument("--webcam", action="store_true", help="Use webcam")
    parser.add_argument("--test", action="store_true", help="Test on sample images")
    parser.add_argument("--output", type=str, default="results", help="Output directory")
    
    args = parser.parse_args()
    
    try:
        # Initialize demo
        demo = WeaponDetectionDemo(args.model)
        
        # Run specific demo
        if args.video:
            demo.process_video(args.video, args.output)
        elif args.webcam:
            demo.webcam_demo()
        elif args.test:
            demo.test_images()
        else:
            # Interactive menu
            demo.interactive_menu()
    
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        print(f"[ERROR] Please ensure the model file exists: {args.model}")
        return 1
    except KeyboardInterrupt:
        print(f"\\n[DEMO] Demo interrupted by user")
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())