import cv2
import numpy as np
from typing import Optional, Tuple, List
from collections import deque

class TargetSelector:
    
    def __init__(self, selection_mode='closest', history_size=10):
        self.selection_mode = selection_mode
        self.selected_track_id = None
        self.target_locked = False
        
        # Error smoothing history
        self.error_history_x = deque(maxlen=history_size)
        self.error_history_y = deque(maxlen=history_size)
        
        # Manual selection
        self.manual_click_pos = None
        
    def select_target(self, tracked_objects: List, frame_center: Tuple[int, int]) -> Optional[int]:
        if len(tracked_objects) == 0:
            return None
        
        # If target is locked and still exists, keep it
        if self.target_locked and self.selected_track_id is not None:
            for obj in tracked_objects:
                if obj[4] == self.selected_track_id:
                    return self.selected_track_id
            # Target lost, unlock
            self.target_locked = False
            self.selected_track_id = None
        
        # Select new target based on mode
        if self.selection_mode == 'closest':
            return self._select_closest_to_center(tracked_objects, frame_center)
        elif self.selection_mode == 'largest':
            return self._select_largest(tracked_objects)
        elif self.selection_mode == 'manual':
            return self._select_manual(tracked_objects)
        elif self.selection_mode == 'center':
            return self._select_closest_to_center(tracked_objects, frame_center)
        else:
            return tracked_objects[0][4]  # Default: first object
    
    def _select_closest_to_center(self, tracked_objects: List, frame_center: Tuple[int, int]) -> int:
        """Select object closest to frame center"""
        min_distance = float('inf')
        selected_id = None
        
        for obj in tracked_objects:
            x1, y1, x2, y2, track_id = obj[:5]
            
            # Calculate center of bbox
            bbox_center_x = (x1 + x2) // 2
            bbox_center_y = (y1 + y2) // 2
            
            # Distance to frame center
            distance = np.sqrt(
                (bbox_center_x - frame_center[0])**2 + 
                (bbox_center_y - frame_center[1])**2
            )
            
            if distance < min_distance:
                min_distance = distance
                selected_id = track_id
        
        return selected_id
    
    def _select_largest(self, tracked_objects: List) -> int:
        """Select largest object by area"""
        max_area = 0
        selected_id = None
        
        for obj in tracked_objects:
            x1, y1, x2, y2, track_id = obj[:5]
            area = (x2 - x1) * (y2 - y1)
            
            if area > max_area:
                max_area = area
                selected_id = track_id
        
        return selected_id
    
    def _select_manual(self, tracked_objects: List) -> Optional[int]:
        """Select object by manual click (set via mouse callback)"""
        if self.manual_click_pos is None:
            return None
        
        click_x, click_y = self.manual_click_pos
        
        # Find object containing click point
        for obj in tracked_objects:
            x1, y1, x2, y2, track_id = obj[:5]
            
            if x1 <= click_x <= x2 and y1 <= click_y <= y2:
                self.manual_click_pos = None  # Reset click
                return track_id
        
        return None
    
    def lock_target(self, track_id: int):
        """Lock onto specific target"""
        self.selected_track_id = track_id
        self.target_locked = True
    
    def unlock_target(self):
        """Unlock current target"""
        self.target_locked = False
        self.selected_track_id = None
        self.error_history_x.clear()
        self.error_history_y.clear()
    
    def calculate_pixel_error(self, target_obj: List, frame_center: Tuple[int, int]) -> Tuple[int, int]:
        x1, y1, x2, y2 = target_obj[:4]
        
        # Target center
        target_x = (x1 + x2) // 2
        target_y = (y1 + y2) // 2
        
        # Raw error
        error_x = target_x - frame_center[0]
        error_y = target_y - frame_center[1]
        
        # Add to history for smoothing
        self.error_history_x.append(error_x)
        self.error_history_y.append(error_y)
        
        return error_x, error_y
    
    def get_smoothed_error(self) -> Tuple[float, float]:
        if len(self.error_history_x) == 0:
            return 0.0, 0.0
        
        smooth_x = sum(self.error_history_x) / len(self.error_history_x)
        smooth_y = sum(self.error_history_y) / len(self.error_history_y)
        
        return smooth_x, smooth_y
    
    def get_velocity_estimate(self) -> Tuple[float, float]:
        if len(self.error_history_x) < 2:
            return 0.0, 0.0
        
        # Calculate velocity from last 5 frames
        window = min(5, len(self.error_history_x))
        
        vx = (self.error_history_x[-1] - self.error_history_x[-window]) / window
        vy = (self.error_history_y[-1] - self.error_history_y[-window]) / window
        
        return vx, vy
    
    def is_target_centered(self, threshold: int = 20) -> bool:
        if len(self.error_history_x) == 0:
            return False
        
        error_x = abs(self.error_history_x[-1])
        error_y = abs(self.error_history_y[-1])
        
        return error_x < threshold and error_y < threshold
    
    def set_manual_click(self, x: int, y: int):
        self.manual_click_pos = (x, y)
    
    def draw_target_info(self, frame: np.ndarray, target_obj: List, 
                        frame_center: Tuple[int, int],
                        color: Tuple[int, int, int] = (0, 255, 255)) -> np.ndarray:
        output = frame.copy()
        x1, y1, x2, y2, track_id = target_obj[:5]
        
        # Calculate target center
        target_x = (x1 + x2) // 2
        target_y = (y1 + y2) // 2
        
        # Draw crosshair brackets around target (lock indicator)
        size = 30
        thickness = 3
        
        # Top-left corner
        cv2.line(output, (x1, y1), (x1 + size, y1), color, thickness)
        cv2.line(output, (x1, y1), (x1, y1 + size), color, thickness)
        
        # Top-right corner
        cv2.line(output, (x2, y1), (x2 - size, y1), color, thickness)
        cv2.line(output, (x2, y1), (x2, y1 + size), color, thickness)
        
        # Bottom-left corner
        cv2.line(output, (x1, y2), (x1 + size, y2), color, thickness)
        cv2.line(output, (x1, y2), (x1, y2 - size), color, thickness)
        
        # Bottom-right corner
        cv2.line(output, (x2, y2), (x2 - size, y2), color, thickness)
        cv2.line(output, (x2, y2), (x2, y2 - size), color, thickness)
        
        # Draw line from center to target
        cv2.line(output, frame_center, (target_x, target_y), color, 2)
        
        # Get smoothed error
        error_x, error_y = self.calculate_pixel_error(target_obj, frame_center)
        smooth_x, smooth_y = self.get_smoothed_error()
        vx, vy = self.get_velocity_estimate()
        
        # Draw error text
        error_text = [
            f"TARGET LOCKED: ID {track_id}",
            f"Raw Error: X={error_x:+4d} Y={error_y:+4d}",
            f"Smooth Error: X={smooth_x:+6.1f} Y={smooth_y:+6.1f}",
            f"Velocity: X={vx:+5.1f} Y={vy:+5.1f} px/frame",
            f"Centered: {'YES' if self.is_target_centered() else 'NO'}"
        ]
        
        # Draw text box
        text_y = frame_center[1] - 120
        for i, text in enumerate(error_text):
            (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            # Background
            cv2.rectangle(output, 
                         (frame_center[0] - w//2 - 5, text_y + i*25 - 15),
                         (frame_center[0] + w//2 + 5, text_y + i*25 + 5),
                         (0, 0, 0), -1)
            
            # Text
            cv2.putText(output, text, 
                       (frame_center[0] - w//2, text_y + i*25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return output