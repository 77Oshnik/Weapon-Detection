#!/usr/bin/env python3
"""
Real-time weapon detection on video streams using trained YOLOv8 model.
Supports video files, webcam, and RTSP streams with threat detection alerts.

Usage:
  python scripts/process_video.py --model runs/train/weapon_detection/weights/best.pt --source video.mp4 --output results/
  python scripts/process_video.py --model best.pt --source 0 --show  # webcam
"""

import argparse
import cv2
import json
import time
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO
import numpy as np


class WeaponDetector:
    def __init__(self, model_path: str, conf_threshold: float = 0.25, device: str = ""):
        """Initialize weapon detection system."""
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.device = device
        
        # Class names (should match training data)
        self.class_names = ['pistol', 'rifle', 'knife', 'no_weapon']
        self.weapon_classes = ['pistol', 'rifle', 'knife']  # Exclude 'no_weapon'
        
        # Detection statistics
        self.total_frames = 0
        self.weapon_detections = 0
        self.detection_log = []
        
        print(f"[INFO] Loaded model: {model_path}")
        print(f"[INFO] Confidence threshold: {conf_threshold}")
        print(f"[INFO] Weapon classes: {self.weapon_classes}")
    
    def detect_weapons(self, frame: np.ndarray, frame_number: int, timestamp: float) -> tuple:
        """
        Detect weapons in a frame.
        Returns: (annotated_frame, detections, threat_level)
        """
        self.total_frames += 1
        
        # Run inference
        results = self.model(frame, conf=self.conf_threshold, device=self.device, verbose=False)
        
        # Process detections
        detections = []
        threat_level = "SAFE"
        weapon_detected = False
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get detection info
                    conf = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = self.class_names[class_id] if class_id < len(self.class_names) else "unknown"
                    
                    # Skip low-confidence detections
                    if conf < self.conf_threshold:
                        continue
                    
                    # Get bounding box
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    
                    detection_info = {
                        'frame_number': frame_number,
                        'timestamp': timestamp,
                        'class_name': class_name,
                        'confidence': conf,
                        'bbox': [x1, y1, x2, y2]
                    }
                    detections.append(detection_info)
                    
                    # Check if it's a weapon
                    if class_name in self.weapon_classes:
                        weapon_detected = True
                        self.weapon_detections += 1
                        
                        # Determine threat level based on weapon type and confidence
                        if class_name in ['pistol', 'rifle'] and conf > 0.7:
                            threat_level = "HIGH"
                        elif class_name in ['pistol', 'rifle'] and conf > 0.5:
                            threat_level = "MEDIUM"
                        elif class_name == 'knife' and conf > 0.6:
                            threat_level = "MEDIUM"
                        else:
                            threat_level = "LOW"
        
        # Annotate frame
        annotated_frame = self.annotate_frame(frame, detections, threat_level)
        
        # Log detection if weapon found
        if weapon_detected:
            self.detection_log.extend(detections)
        
        return annotated_frame, detections, threat_level
    
    def annotate_frame(self, frame: np.ndarray, detections: list, threat_level: str) -> np.ndarray:
        """Annotate frame with detection boxes and threat level."""
        annotated = frame.copy()
        
        # Define colors for different classes
        colors = {
            'pistol': (0, 0, 255),      # Red
            'rifle': (0, 0, 255),       # Red  
            'knife': (0, 165, 255),     # Orange
            'no_weapon': (0, 255, 0)    # Green
        }
        
        # Draw detections
        for det in detections:
            class_name = det['class_name']
            conf = det['confidence']
            x1, y1, x2, y2 = [int(x) for x in det['bbox']]
            
            # Get color for this class
            color = colors.get(class_name, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{class_name}: {conf:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(annotated, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(annotated, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Draw threat level indicator
        threat_colors = {
            'SAFE': (0, 255, 0),      # Green
            'LOW': (0, 255, 255),     # Yellow
            'MEDIUM': (0, 165, 255),  # Orange
            'HIGH': (0, 0, 255)       # Red
        }
        
        threat_color = threat_colors.get(threat_level, (255, 255, 255))
        cv2.rectangle(annotated, (10, 10), (200, 50), threat_color, -1)
        cv2.putText(annotated, f"THREAT: {threat_level}", (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # Add timestamp
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(annotated, timestamp_str, (10, annotated.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return annotated
    
    def get_statistics(self) -> dict:
        """Get detection statistics."""
        return {
            'total_frames': self.total_frames,
            'weapon_detections': self.weapon_detections,
            'detection_rate': self.weapon_detections / max(self.total_frames, 1),
            'total_weapon_instances': len(self.detection_log)
        }


def main():
    parser = argparse.ArgumentParser(description="Real-time weapon detection")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to trained YOLOv8 model (.pt file)")
    parser.add_argument("--source", type=str, default="0",
                        help="Video source (file path, webcam index, or RTSP URL)")
    parser.add_argument("--output", type=str, default="results",
                        help="Output directory for results")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold for detections")
    parser.add_argument("--device", type=str, default="",
                        help="Device to run inference on")
    parser.add_argument("--show", action="store_true",
                        help="Display video in real-time")
    parser.add_argument("--save-video", action="store_true",
                        help="Save annotated video")
    parser.add_argument("--save-log", action="store_true",
                        help="Save detection log to JSON")
    parser.add_argument("--fps-limit", type=int, default=30,
                        help="Limit processing FPS")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize detector
    detector = WeaponDetector(args.model, args.conf, args.device)
    
    # Open video source
    if args.source.isdigit():
        cap = cv2.VideoCapture(int(args.source))
    else:
        cap = cv2.VideoCapture(args.source)
    
    if not cap.isOpened():
        print(f"[ERROR] Could not open video source: {args.source}")
        return 1
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if args.source != "0" else 0
    
    print(f"[INFO] Video properties: {width}x{height} @ {fps} FPS")
    if total_frames > 0:
        print(f"[INFO] Total frames: {total_frames}")
    
    # Setup video writer if saving
    out_writer = None
    if args.save_video:
        output_path = output_dir / f"detected_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        print(f"[INFO] Saving video to: {output_path}")
    
    # Processing loop
    frame_number = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_number += 1
            current_time = time.time()
            
            # Process frame
            annotated_frame, detections, threat_level = detector.detect_weapons(
                frame, frame_number, current_time
            )
            
            # Show progress for video files
            if total_frames > 0 and frame_number % 100 == 0:
                progress = (frame_number / total_frames) * 100
                print(f"[INFO] Processing: {progress:.1f}% ({frame_number}/{total_frames})")
            
            # Display real-time alerts
            if threat_level != "SAFE":
                print(f"[ALERT] Frame {frame_number}: {threat_level} threat detected!")
                for det in detections:
                    if det['class_name'] in detector.weapon_classes:
                        print(f"  -> {det['class_name']} (conf: {det['confidence']:.3f})")
            
            # Save frame if video writer is active
            if out_writer:
                out_writer.write(annotated_frame)
            
            # Display frame if requested
            if args.show:
                cv2.imshow('Weapon Detection', annotated_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
            
            # FPS limiting
            if args.fps_limit > 0:
                time.sleep(max(0, 1/args.fps_limit - (time.time() - current_time)))
    
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        if out_writer:
            out_writer.release()
        cv2.destroyAllWindows()
        
        # Get final statistics
        stats = detector.get_statistics()
        processing_time = time.time() - start_time
        
        print(f"\n[SUMMARY] Processing completed")
        print(f"  Processed frames: {stats['total_frames']}")
        print(f"  Weapon detections: {stats['weapon_detections']}")
        print(f"  Detection rate: {stats['detection_rate']:.3f}")
        print(f"  Processing time: {processing_time:.2f} seconds")
        print(f"  Average FPS: {stats['total_frames'] / processing_time:.2f}")
        
        # Save detection log
        if args.save_log and detector.detection_log:
            log_path = output_dir / f"detections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(log_path, 'w') as f:
                json.dump({
                    'statistics': stats,
                    'processing_time': processing_time,
                    'detections': detector.detection_log
                }, f, indent=2)
            print(f"[INFO] Detection log saved to: {log_path}")
    
    return 0


if __name__ == "__main__":
    exit(main())