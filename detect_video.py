#!/usr/bin/env python3
"""
Simple video detection script for weapon detection.
Usage: python detect_video.py --model models/weights/best.pt --source video.mp4
"""

import argparse
import cv2
import numpy as np
import time
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Simple weapon detection on video")
    parser.add_argument("--model", type=str, default="models/weights/best.pt",
                        help="Path to trained model")
    parser.add_argument("--source", type=str, default="0", 
                        help="Video source (file path or webcam index)")
    parser.add_argument("--output", type=str, default="results",
                        help="Output directory")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold")
    parser.add_argument("--show", action="store_true",
                        help="Show video window")
    parser.add_argument("--save-video", action="store_true",
                        help="Save output video")
    
    args = parser.parse_args()
    
    # Check if model exists
    if not Path(args.model).exists():
        print(f"[ERROR] Model not found: {args.model}")
        return 1
    
    # Load model
    print(f"[INFO] Loading model: {args.model}")
    model = YOLO(args.model)
    
    # Class names (adjust based on your training)
    class_names = {0: 'pistol', 1: 'rifle', 2: 'knife', 3: 'no_weapon'}
    weapon_classes = ['pistol', 'rifle', 'knife']
    
    # Colors for different classes
    colors = {
        'pistol': (0, 0, 255),      # Red
        'rifle': (0, 0, 255),       # Red  
        'knife': (0, 165, 255),     # Orange
        'no_weapon': (0, 255, 0)    # Green
    }
    
    # Open video source
    if args.source.isdigit():
        cap = cv2.VideoCapture(int(args.source))
        print(f"[INFO] Using webcam: {args.source}")
    else:
        cap = cv2.VideoCapture(args.source)
        print(f"[INFO] Processing video: {args.source}")
    
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video source: {args.source}")
        return 1
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"[INFO] Video properties: {width}x{height} @ {fps} FPS")
    if total_frames > 0:
        print(f"[INFO] Total frames: {total_frames}")
    
    # Setup output
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    out_writer = None
    if args.save_video:
        output_path = output_dir / f"detected_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        print(f"[INFO] Output video: {output_path}")
    
    # Detection statistics
    frame_count = 0
    weapon_detections = 0
    start_time = time.time()
    
    print(f"[INFO] Starting detection...")
    print(f"[INFO] Press 'q' to quit, 's' to save screenshot")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Run detection
            results = model(frame, conf=args.conf, verbose=False)
            
            # Process results
            detections = []
            threat_level = "SAFE"
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        conf = float(box.conf[0])
                        class_id = int(box.cls[0])
                        
                        if class_id in class_names:
                            class_name = class_names[class_id]
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            
                            detections.append({
                                'class': class_name,
                                'conf': conf,
                                'bbox': [int(x1), int(y1), int(x2), int(y2)]
                            })
                            
                            if class_name in weapon_classes:
                                weapon_detections += 1
                                if class_name in ['pistol', 'rifle'] and conf > 0.7:
                                    threat_level = "HIGH"
                                elif class_name in ['pistol', 'rifle'] and conf > 0.5:
                                    threat_level = "MEDIUM"
                                elif class_name == 'knife' and conf > 0.6:
                                    threat_level = "MEDIUM"
                                elif threat_level == "SAFE":
                                    threat_level = "LOW"
            
            # Draw detections
            for det in detections:
                class_name = det['class']
                conf = det['conf']
                x1, y1, x2, y2 = det['bbox']
                
                color = colors.get(class_name, (255, 255, 255))
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw label
                label = f"{class_name}: {conf:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                             (x1 + label_size[0], y1), color, -1)
                cv2.putText(frame, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Draw threat level
            threat_colors = {
                'SAFE': (0, 255, 0),      # Green
                'LOW': (0, 255, 255),     # Yellow
                'MEDIUM': (0, 165, 255),  # Orange
                'HIGH': (0, 0, 255)       # Red
            }
            
            threat_color = threat_colors.get(threat_level, (255, 255, 255))
            cv2.rectangle(frame, (10, 10), (200, 50), threat_color, -1)
            cv2.putText(frame, f"THREAT: {threat_level}", (20, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            # Add timestamp and frame number
            timestamp_str = datetime.now().strftime("%H:%M:%S")
            cv2.putText(frame, f"Frame: {frame_count} | {timestamp_str}", 
                       (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Show progress for video files
            if total_frames > 0 and frame_count % 100 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"[INFO] Progress: {progress:.1f}% ({frame_count}/{total_frames})")
            
            # Print alerts
            if threat_level != "SAFE":
                weapon_list = [d['class'] for d in detections if d['class'] in weapon_classes]
                if weapon_list:
                    print(f"[ALERT] Frame {frame_count}: {threat_level} - {', '.join(weapon_list)}")
            
            # Save frame
            if out_writer:
                out_writer.write(frame)
            
            # Show frame
            if args.show or args.source == "0":
                cv2.imshow('Weapon Detection', frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    screenshot_path = output_dir / f"screenshot_{frame_count:06d}.jpg"
                    cv2.imwrite(str(screenshot_path), frame)
                    print(f"[INFO] Screenshot saved: {screenshot_path}")
    
    except KeyboardInterrupt:
        print("\\n[INFO] Interrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        if out_writer:
            out_writer.release()
        cv2.destroyAllWindows()
        
        # Print summary
        processing_time = time.time() - start_time
        avg_fps = frame_count / processing_time if processing_time > 0 else 0
        
        print(f"\\n[SUMMARY]")
        print(f"  Frames processed: {frame_count}")
        print(f"  Weapon detections: {weapon_detections}")
        print(f"  Processing time: {processing_time:.2f} seconds")
        print(f"  Average FPS: {avg_fps:.2f}")
        print(f"  Detection rate: {weapon_detections/max(frame_count,1):.3f}")
    
    return 0


if __name__ == "__main__":
    exit(main())