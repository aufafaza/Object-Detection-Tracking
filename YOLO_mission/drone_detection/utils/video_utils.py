import cv2
import numpy as np
from typing import Tuple, Optional
import time

class VideoHandler: 
    '''
    VIDEO INPUT HANDLER 
    '''

    def __init__(self, source: int = 0, width: int = 640, height: int = 480,
                 use_clahe: bool = False, use_gpu: bool = False):
        self.cap = cv2.VideoCapture(source)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        self.actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))

        self.center_x = self.actual_width // 2
        self.center_y = self.actual_height // 2

        self.use_clahe = use_clahe
        self.use_gpu = use_gpu and hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0

        # CPU CLAHE fallback
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) if use_clahe else None

        # GPU CLAHE
        if self.use_clahe and self.use_gpu:
            self.gpu_clahe = cv2.cuda.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        self.frame_times = []
        self.current_fps = 0

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a frame and track FPS (optimized)."""
        start_time = time.time()
        ret, frame = self.cap.read()
        if not ret:
            return False, None

        if self.use_clahe:
            if self.use_gpu:
                gpu_frame = cv2.cuda_GpuMat()
                gpu_frame.upload(frame)
                gpu_lab = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.cuda.split(gpu_lab)
                l = self.gpu_clahe.apply(l)
                lab_merged = cv2.cuda.merge([l, a, b])
                frame = cv2.cuda.cvtColor(lab_merged, cv2.COLOR_LAB2BGR).download()
            else:
                frame = self.apply_clahe(frame)

        # FPS calculation
        elapsed = time.time() - start_time
        self.frame_times.append(elapsed)
        if len(self.frame_times) > 30:
            self.frame_times.pop(0)
        avg_time = sum(self.frame_times) / len(self.frame_times) if self.frame_times else 0
        self.current_fps = 1.0 / avg_time if avg_time > 0 else 0

        return True, frame

    def apply_clahe(self, frame: np.ndarray) -> np.ndarray:
        """Apply CLAHE to the L channel only (BGR->LAB->BGR)."""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = self.clahe.apply(l)
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    def draw_fps(self, frame: np.ndarray) -> np.ndarray:
        """Draw FPS counter on frame"""
        output = frame.copy()
        fps_text = f"FPS: {self.current_fps:.1f}"
        cv2.putText(output, fps_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return output

    def preprocess_frame(self, frame: np.ndarray, use_clahe: bool = True) -> np.ndarray:
        '''
        PREPROCESSING
        '''
        processed = frame.copy()

        if use_clahe and self.clahe is not None:
            # Convert to LAB color space
            lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
            
            # Apply CLAHE to L channel (lightness)
            l, a, b = cv2.split(lab)
            l = self.clahe.apply(l)
            
            # Merge channels and convert back to BGR
            lab = cv2.merge([l, a, b])
            processed = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return processed


    def calculate_pixel_error(self, bbox_center: Tuple[int, int]) -> Tuple[int, int]:
        '''
        offset from drone crosshair 
        '''

        error_x = bbox_center[0] - self.center_x
        error_y = bbox_center[1] - self.center_y
        
        return error_x, error_y
    
    def draw_crosshair(self, frame, color=(0,255,0)):
        cv2.line(frame, (self.center_x, 0), (self.center_x, self.actual_height), color, 2)
        cv2.line(frame, (0, self.center_y), (self.actual_width, self.center_y), color, 2)
        cv2.circle(frame, (self.center_x, self.center_y), 10, color, 2)
        return frame

    

    
    def draw_target_line(self, frame: np.ndarray, target_center: Tuple[int, int], color: Tuple[int, int, int] = (255, 0, 0)) -> np.ndarray:
        '''
        DRAW LINE FROM CENTER TO TARGET AND TYPE ERROR
        '''
        output = frame.copy()
        
        # Draw line from center to target
        cv2.line(output, (self.center_x, self.center_y), target_center, color, 2)
        
        # Calculate and display error
        error_x, error_y = self.calculate_pixel_error(target_center)
        
        # Display error text near center
        text = f"Error: X={error_x:+d} Y={error_y:+d}"
        cv2.putText(output, text, (self.center_x - 100, self.center_y - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw target circle
        cv2.circle(output, target_center, 15, color, 2)
        
        return output
    
    def resize_if_needed(self, frame: np.ndarray, max_width: int = 1920) -> np.ndarray:
        """
        resizing 
        """
        height, width = frame.shape[:2]
        
        if width > max_width:
            scale = max_width / width
            new_width = max_width
            new_height = int(height * scale)
            return cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        return frame
    
    def release(self):
        """Release video capture resources"""
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
    
    def __del__(self):
        """Cleanup on object destruction"""
        self.release()

        