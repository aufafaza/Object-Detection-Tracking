import cv2 
import numpy as np 
from typing import List, Tuple, Optional 
from ultralytics import YOLO 
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.cv import read_image 
import time 

class ObjectDetector:
    def __init__(self, model_path, str = "yolov8n.pt", confidence_threshold: float = 0.25, device: str = "cpu"): 
        '''
        INIT YOLO MODEL WITH SAHI 
        '''

        print(f"YOLO model: {model_path}")

        self.detection_model = AutoDetectionModel.from_pretrained(
            model_type='yolov8',
            model_path = model_path,
            confidence_threshold=confidence_threshold, 
            device=device,
        )

        self.yolo_model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.device = device 

        self.inference_times = [] 

        print(f"Model loaded on {device}")

    def detect_with_sahi(self, frame: np.ndarray, slice_height: int=640, slice_width: int=480, overlap_height_ratio: float = 0.2, overlap_width_ratio: float=0.2) -> Tuple[List, float]: 

        start_time = time.time() 

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result = get_sliced_prediction(
            frame_rgb, 
            self.detection_model,
            slice_height=slice_height,
            slice_width=slice_width,
            overlap_height_ratio=overlap_height_ratio,
            overlap_width_ratio=overlap_width_ratio,
            verbose=0
        )

        inference_time = time.time() - start_time 

        detections = [] 

        for obj in result.object_prediction_list:
            bbox = obj.bbox 

            detections.append([
                int(bbox.minx),  # Convert to int
                int(bbox.miny),
                int(bbox.maxx),
                int(bbox.maxy),
                float(obj.score.value),  # Extract .value from PredictionScore
                int(obj.category.id),
                str(obj.category.name)
            ])

        self.inference_times.append(inference_time)
        if len(self.inference_times) > 30:
            self.inference_times.pop(0)
        
        return detections, inference_time


    def detect_standard(self, frame: np.ndarray) -> Tuple[List, float]:
        """
        Run standard YOLO inference (no slicing) for comparison
        
        Returns:
            (detections, inference_time)
        """
        start_time = time.time()
        
        results = self.yolo_model(frame, verbose=False)[0]
        
        inference_time = time.time() - start_time
        
        # Convert to same format as SAHI
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = results.names[cls_id]
            
            detections.append([
                int(x1), int(y1), int(x2), int(y2),
                conf, cls_id, cls_name
            ])
        
        return detections, inference_time
    def filter_by_class(self, detections: List, target_class: str) -> List: 
        return [det for det in detections if det[6].lower() == target_class.lower()]

    def filter_by_area(self, detections: List, min_area: int = 100) -> List:
        """Filter out very small detections (likely false positives)"""
        filtered = []
        for det in detections:
            x1, y1, x2, y2 = det[:4]
            area = (x2 - x1) * (y2 - y1)
            if area >= min_area:
                filtered.append(det)
        return filtered
    
    def get_bbox_center(self, detection: List) -> Tuple[int, int]:
        """Calculate center point of bounding box"""
        x1, y1, x2, y2 = detection[:4]
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        return center_x, center_y
    

    def draw_slice_grid(self, frame: np.ndarray,
                       slice_height: int = 640,
                       slice_width: int = 640,
                       overlap_height_ratio: float = 0.2,
                       overlap_width_ratio: float = 0.2) -> np.ndarray:
        """Visualize SAHI slice grid (for debugging)"""
        output = frame.copy()
        h, w = frame.shape[:2]
        
        step_h = int(slice_height * (1 - overlap_height_ratio))
        step_w = int(slice_width * (1 - overlap_width_ratio))
        
        for x in range(0, w, step_w):
            cv2.line(output, (x, 0), (x, h), (255, 0, 0), 1)
        
        for y in range(0, h, step_h):
            cv2.line(output, (0, y), (w, y), (255, 0, 0), 1)
        
        cv2.putText(output, f"SAHI Grid: {slice_width}x{slice_height}",
                   (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        return output
    
    def get_avg_inference_time(self) -> float:
        """Get average inference time in milliseconds"""
        if not self.inference_times:
            return 0.0
        return (sum(self.inference_times) / len(self.inference_times)) * 1000
    
    def draw_detections(self, frame: np.ndarray, detections: List,
                       color: Tuple[int, int, int] = (0, 0, 255),
                       show_label: bool = True) -> np.ndarray:
        """Draw bounding boxes on frame"""
        output = frame.copy()
        
        for det in detections:
            # Ensure all coordinates are integers
            x1, y1, x2, y2, conf, cls_id, cls_name = det
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Draw bounding box

            if not isinstance(conf, float):
                conf = float(conf)
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            
            if show_label:
                label = f"{cls_name}: {conf:.2f}"
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(output, (x1, y1 - 20), (x1 + w, y1), color, -1)
                cv2.putText(output, label, (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Draw center point
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            cv2.circle(output, (center_x, center_y), 5, (0, 255, 0), -1)
        
        return output