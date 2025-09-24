"""Real-time weapon detection for live streams."""

import cv2
import numpy as np
from typing import Optional, Callable, Dict, Any
import threading
import time
from queue import Queue, Empty
from datetime import datetime, timedelta

from ..detector import WeaponDetector
from ..utils.metrics import ThreatLevel


class RealTimeDetector:
    """Real-time weapon detection system for live streams."""
    
    def __init__(self, detector: WeaponDetector, alert_callback: Optional[Callable] = None,
                 buffer_size: int = 30, detection_interval: int = 3):
        """Initialize real-time detector.
        
        Args:
            detector: WeaponDetector instance
            alert_callback: Function to call when weapons detected
            buffer_size: Frame buffer size
            detection_interval: Process every N frames for performance
        """
        self.detector = detector
        self.alert_callback = alert_callback
        self.buffer_size = buffer_size
        self.detection_interval = detection_interval
        
        # Threading components
        self.frame_queue = Queue(maxsize=buffer_size)
        self.result_queue = Queue(maxsize=buffer_size)
        self.running = False
        
        # Detection thread
        self.detection_thread = None
        
        # Statistics
        self.frame_count = 0
        self.detections_count = 0
        self.current_threat_level = ThreatLevel.SAFE
        self.last_detection_time = None
        
        # Alert state
        self.alert_cooldown = 5.0  # seconds
        self.last_alert_time = 0
        
    def start(self):
        """Start the real-time detection system."""
        if self.running:
            return
        
        self.running = True
        self.detection_thread = threading.Thread(target=self._detection_worker)
        self.detection_thread.daemon = True
        self.detection_thread.start()
        
        print("[INFO] Real-time detection started")
    
    def stop(self):
        """Stop the real-time detection system."""
        self.running = False
        if self.detection_thread:
            self.detection_thread.join(timeout=2.0)
        
        # Clear queues
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except Empty:
                break
        
        while not self.result_queue.empty():
            try:
                self.result_queue.get_nowait()
            except Empty:
                break
        
        print("[INFO] Real-time detection stopped")
    
    def process_frame(self, frame: np.ndarray) -> tuple:
        """Process a single frame.
        
        Args:
            frame: Input frame
            
        Returns:
            Tuple of (annotated_frame, detections, threat_level)
        """
        self.frame_count += 1
        
        # Add frame to queue for processing
        if not self.frame_queue.full():
            timestamp = time.time()
            self.frame_queue.put((frame.copy(), self.frame_count, timestamp))
        
        # Get latest results
        detections = []
        threat_level = ThreatLevel.SAFE
        
        try:
            while not self.result_queue.empty():
                latest_result = self.result_queue.get_nowait()
                detections = latest_result['detections']
                threat_level = latest_result['threat_level']
                self.current_threat_level = threat_level
                
                if threat_level != ThreatLevel.SAFE:
                    self.last_detection_time = latest_result['timestamp']
                    self.detections_count += 1
                    
                    # Trigger alert if needed
                    self._check_alert(detections, threat_level)
                    
        except Empty:
            # Use cached results
            threat_level = self.current_threat_level
        
        # Annotate frame
        annotated_frame = self.detector.annotate_frame(frame, detections, threat_level)
        
        # Add real-time stats
        self._add_realtime_overlay(annotated_frame)
        
        return annotated_frame, detections, threat_level
    
    def _detection_worker(self):
        """Background thread for weapon detection."""
        print("[INFO] Detection worker thread started")
        
        while self.running:
            try:
                # Get frame from queue
                frame, frame_number, timestamp = self.frame_queue.get(timeout=1.0)
                
                # Skip frames for performance
                if frame_number % self.detection_interval != 0:
                    continue
                
                # Run detection
                detections, threat_level = self.detector.detect(
                    frame, frame_number, timestamp
                )
                
                # Store results
                result = {
                    'frame_number': frame_number,
                    'timestamp': timestamp,
                    'detections': detections,
                    'threat_level': threat_level
                }
                
                if not self.result_queue.full():
                    self.result_queue.put(result)
                else:
                    # Remove old result and add new one
                    try:
                        self.result_queue.get_nowait()
                        self.result_queue.put(result)
                    except Empty:
                        pass
                
            except Empty:
                continue
            except Exception as e:
                print(f"[ERROR] Detection worker error: {e}")
        
        print("[INFO] Detection worker thread stopped")
    
    def _check_alert(self, detections: list, threat_level: ThreatLevel):
        """Check if an alert should be triggered."""
        current_time = time.time()
        
        # Check cooldown
        if current_time - self.last_alert_time < self.alert_cooldown:
            return
        
        # Trigger alert for medium/high threats
        if threat_level.value >= ThreatLevel.MEDIUM.value and self.alert_callback:
            alert_data = {
                'timestamp': datetime.now(),
                'threat_level': threat_level.name,
                'detections': detections,
                'frame_number': self.frame_count
            }
            
            try:
                self.alert_callback(alert_data)
                self.last_alert_time = current_time
                print(f"[ALERT] {threat_level.name} threat detected!")
            except Exception as e:
                print(f"[ERROR] Alert callback failed: {e}")
    
    def _add_realtime_overlay(self, frame: np.ndarray):
        """Add real-time statistics overlay to frame."""
        height, width = frame.shape[:2]
        
        # System stats
        stats_text = [
            f"Frames: {self.frame_count}",
            f"Detections: {self.detections_count}",
            f"Current: {self.current_threat_level.name}",
        ]
        
        if self.last_detection_time:
            time_since = time.time() - self.last_detection_time
            stats_text.append(f"Last: {time_since:.1f}s ago")
        
        # Draw stats
        y_offset = 30
        for i, text in enumerate(stats_text):
            y_pos = y_offset + (i * 25)
            cv2.putText(frame, text, (width - 200, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get real-time detection statistics."""
        return {
            'frames_processed': self.frame_count,
            'total_detections': self.detections_count,
            'current_threat_level': self.current_threat_level.name,
            'detection_rate': self.detections_count / max(self.frame_count, 1),
            'last_detection_time': self.last_detection_time,
            'queue_sizes': {
                'frame_queue': self.frame_queue.qsize(),
                'result_queue': self.result_queue.qsize()
            },
            'detector_stats': self.detector.get_statistics()
        }


class AlertSystem:
    """Alert system for weapon detection events."""
    
    def __init__(self, email_config: Optional[Dict] = None, 
                 webhook_url: Optional[str] = None,
                 log_file: Optional[str] = None):
        """Initialize alert system.
        
        Args:
            email_config: Email configuration for alerts
            webhook_url: Webhook URL for alerts
            log_file: Log file for alert history
        """
        self.email_config = email_config
        self.webhook_url = webhook_url
        self.log_file = log_file
        
        # Alert history
        self.alert_history = []
    
    def send_alert(self, alert_data: Dict[str, Any]):
        """Send alert through configured channels.
        
        Args:
            alert_data: Alert information dictionary
        """
        # Log alert
        self.alert_history.append(alert_data)
        
        # Write to log file
        if self.log_file:
            self._log_to_file(alert_data)
        
        # Send email alert
        if self.email_config:
            self._send_email_alert(alert_data)
        
        # Send webhook alert
        if self.webhook_url:
            self._send_webhook_alert(alert_data)
        
        print(f"[ALERT] {alert_data['threat_level']} threat - {len(alert_data['detections'])} weapons detected")
    
    def _log_to_file(self, alert_data: Dict[str, Any]):
        """Log alert to file."""
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                timestamp = alert_data['timestamp'].strftime("%Y-%m-%d %H:%M:%S")
                threat_level = alert_data['threat_level']
                detection_count = len(alert_data['detections'])
                weapons = [d['class_name'] for d in alert_data['detections']]
                
                f.write(f"{timestamp} | {threat_level} | {detection_count} weapons | {', '.join(weapons)}\n")
        except Exception as e:
            print(f"[ERROR] Failed to log alert: {e}")
    
    def _send_email_alert(self, alert_data: Dict[str, Any]):
        """Send email alert (placeholder - requires email implementation)."""
        # TODO: Implement email sending
        print(f"[EMAIL] Would send email alert: {alert_data['threat_level']}")
    
    def _send_webhook_alert(self, alert_data: Dict[str, Any]):
        """Send webhook alert (placeholder - requires requests)."""
        # TODO: Implement webhook sending
        print(f"[WEBHOOK] Would send webhook alert: {alert_data['threat_level']}")
    
    def get_alert_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get alert summary for specified time period."""
        current_time = datetime.now()
        cutoff_time = current_time - timedelta(hours=hours)
        
        recent_alerts = [
            alert for alert in self.alert_history
            if alert['timestamp'] >= cutoff_time
        ]
        
        threat_counts = {}
        for alert in recent_alerts:
            level = alert['threat_level']
            threat_counts[level] = threat_counts.get(level, 0) + 1
        
        return {
            'total_alerts': len(recent_alerts),
            'threat_level_counts': threat_counts,
            'time_period_hours': hours,
            'most_recent': recent_alerts[-1] if recent_alerts else None
        }