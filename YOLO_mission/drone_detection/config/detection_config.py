import cv2
import numpy 
import ultralytics
import sahi 
import filterpy 
import torch 

class DetectionConfig: 
        # Video Input
    VIDEO_WIDTH = 640  # 1080p horizontal resolution
    VIDEO_HEIGHT = 480
    TARGET_FPS = 30
    
    # YOLO Model
    MODEL_PATH = "yolov8n.pt"  
    CONFIDENCE_THRESHOLD = 0.15
    IOU_THRESHOLD = 0.45  
    
    # SAHI Parameters
    SLICE_HEIGHT = 640  
    SLICE_WIDTH = 640
    OVERLAP_HEIGHT_RATIO = 0.2  
    OVERLAP_WIDTH_RATIO = 0.2
    
    USE_CLAHE = True  
    CLAHE_CLIP_LIMIT = 2.0
    CLAHE_TILE_SIZE = (8, 8)
    
    # Target Tracking
    TARGET_CLASS = "cell phone"  
    MIN_DETECTION_AREA = 50  
    
        # Display
    SHOW_SLICES = True  
    SHOW_DETECTIONS = True
    CROSSHAIR_COLOR = (0, 255, 0)  
    BBOX_COLOR = (0, 0, 255)  