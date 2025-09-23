"""Main weapon detection class."""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Any
from ultralytics import YOLO
from pathlib import Path
import time
from datetime import datetime

from .utils.visualization import draw_detections, create_threat_overlay
from .utils.metrics import ThreatLevel


class WeaponDetector:
    """Main weapon detection system using YOLOv8."""
    
    def __init__(self, model_path: str, conf_threshold: float = 0.25, device: str = ""):
        """Initialize weapon detection system.
        
        Args:
            model_path: Path to trained YOLOv8 model
            conf_threshold: Confidence threshold for detections
            device: Device to run inference on (cuda/cpu)
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.device = device
        
        # Class configuration
        self.class_names = ['pistol', 'rifle', 'knife', 'no_weapon']
        self.weapon_classes = ['pistol', 'rifle', 'knife']
        
        # Statistics
        self.reset_statistics()
        
        print(f"[INFO] Loaded model: {model_path}")
        print(f"[INFO] Classes: {self.class_names}")
    
    def reset_statistics(self):
        """Reset detection statistics."""
        self.total_frames = 0
        self.weapon_detections = 0
        self.detection_log = []
    
    def detect(self, frame: np.ndarray, frame_number: int = None, 
              timestamp: float = None) -> Tuple[List[Dict], str]:
        """Detect weapons in a frame.
        
        Args:
            frame: Input image frame
            frame_number: Frame number for logging
            timestamp: Timestamp for logging
            
        Returns:
            Tuple of (detections, threat_level)
        """
        if timestamp is None:
            timestamp = time.time()
        if frame_number is None:
            frame_number = self.total_frames
            
        self.total_frames += 1
        
        # Run inference
        results = self.model(frame, conf=self.conf_threshold, 
                           device=self.device, verbose=False)
        
        detections = []
        threat_level = ThreatLevel.SAFE
        weapon_detected = False
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    conf = float(box.conf[0])
                    class_id = int(box.cls[0])
                    
                    if class_id >= len(self.class_names):
                        continue
                        
                    class_name = self.class_names[class_id]
                    
                    if conf < self.conf_threshold:
                        continue
                    
                    # Get bounding box
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    
                    detection = {
                        'frame_number': frame_number,
                        'timestamp': timestamp,
                        'class_name': class_name,
                        'confidence': conf,
                        'bbox': [x1, y1, x2, y2],
                        'class_id': class_id
                    }
                    detections.append(detection)
                    
                    # Check threat level
                    if class_name in self.weapon_classes:
                        weapon_detected = True
                        self.weapon_detections += 1
                        
                        # Determine threat level
                        if class_name in ['pistol', 'rifle'] and conf > 0.7:
                            threat_level = max(threat_level, ThreatLevel.HIGH)
                        elif class_name in ['pistol', 'rifle'] and conf > 0.5:
                            threat_level = max(threat_level, ThreatLevel.MEDIUM)
                        elif class_name == 'knife' and conf > 0.6:
                            threat_level = max(threat_level, ThreatLevel.MEDIUM)
                        else:
                            threat_level = max(threat_level, ThreatLevel.LOW)
        
        # Log weapon detections
        if weapon_detected:
            weapon_detections = [d for d in detections if d['class_name'] in self.weapon_classes]
            self.detection_log.extend(weapon_detections)
        
        return detections, threat_level
    
    def annotate_frame(self, frame: np.ndarray, detections: List[Dict], 
                      threat_level: str) -> np.ndarray:
        """Annotate frame with detections and threat level.
        
        Args:
            frame: Input frame
            detections: List of detection dictionaries
            threat_level: Current threat level
            
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        
        # Draw detections
        annotated = draw_detections(annotated, detections)
        
        # Add threat level overlay
        annotated = create_threat_overlay(annotated, threat_level)
        
        # Add timestamp
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(annotated, timestamp_str, (10, annotated.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return annotated
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get detection statistics."""
        return {
            'total_frames': self.total_frames,
            'weapon_detections': self.weapon_detections,
            'detection_rate': self.weapon_detections / max(self.total_frames, 1),
            'total_weapon_instances': len(self.detection_log),
            'class_counts': self._get_class_counts()
        }
    
    def _get_class_counts(self) -> Dict[str, int]:
        """Count detections by class."""
        counts = {cls: 0 for cls in self.weapon_classes}
        for detection in self.detection_log:
            class_name = detection['class_name']
            if class_name in counts:
                counts[class_name] += 1
        return counts