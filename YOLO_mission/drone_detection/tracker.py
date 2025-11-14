import numpy as np 
from filterpy.kalman import KalmanFilter 
from typing import List, Tuple, Optional 
import cv2 

class KalmanBoxTracker: 
    count = 0 

    def __init__(self, bbox): 
        self.kf = KalmanFilter(dim_x=7, dim_z=4)

        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],  # x = x + vx
            [0, 1, 0, 0, 0, 1, 0],  # y = y + vy
            [0, 0, 1, 0, 0, 0, 1],  # s = s + vs
            [0, 0, 0, 1, 0, 0, 0],  # r = r
            [0, 0, 0, 0, 1, 0, 0],  # vx = vx
            [0, 0, 0, 0, 0, 1, 0],  # vy = vy
            [0, 0, 0, 0, 0, 0, 1]   # vs = vs
        ])

        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ])

        self.kf.R[2:, 2:] *= 10.0

        self.kf.P[4:, 4:] *= 1000.0  # High uncertainty in initial velocities
        self.kf.P *= 10.0
        
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        
        # Initialize with first detection
        self.kf.x[:4] = self._convert_bbox_to_z(bbox)
        
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, bbox): 
        self.time_since_update = 0 
        self.history = [] 
        self.hits += 1
        self.hit_streak += 1 
        self.kf.update(self._convert_bbox_to_z(bbox))

    def predict(self): 
        if (self.kf.x[6] + self.kf.x[2]) <= 0: 
            self.kf.x[6] *= 0.0 
        self.kf.predict() 
        self.age += 1 
        if self.time_since_update > 0: 
            self.hit_streak = 0 
        self.time_since_update += 1 
        self.history.append(self._convert_x_to_bbox(self.kf.x)) 
        return self.history[-1]
    
    def get_state(self): 
        return self._convert_x_to_bbox(self.kf.x)
    
    @staticmethod
    def _convert_bbox_to_z(bbox):
        """Convert [x1, y1, x2, y2] to [cx, cy, s, r]"""
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w / 2.0
        y = bbox[1] + h / 2.0
        s = w * h  # scale (area)
        r = w / float(h + 1e-6)  # aspect ratio
        return np.array([x, y, s, r]).reshape((4, 1))
    
    @staticmethod
    def _convert_x_to_bbox(x):
        """Convert [cx, cy, s, r] back to [x1, y1, x2, y2]"""
        w = np.sqrt(x[2] * x[3])
        h = x[2] / (w + 1e-6)
        return np.array([
            x[0] - w / 2.0,
            x[1] - h / 2.0,
            x[0] + w / 2.0,
            x[1] + h / 2.0
        ]).reshape((1, 4))
    
class ObjectTracker: 
    def __init__(self, max_age: int = 5, min_hits: int = 3, iou_threshold: float = 0.15): 
        self.max_age = max_age 
        self.min_hits = min_hits 
        self.iou_threshold = iou_threshold 
        self.trackers: List[KalmanBoxTracker] = [] 
        self.frame_count = 0

    def update(self, detections: List) -> List: 
        self.frame_count += 1 

        trks = np.zeros((len(self.trackers), 5))
        to_del = [] 

        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)

        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        if len(detections) > 0: 
            dets = np.array([det[:4] for det in detections])
        else: 
            dets = np.empty((0, 5))

        matched, unmatched_dets, unmatched_trks = self._associate_detections_to_trackers(dets, trks, self.iou_threshold)

        for m in matched: 
            self.trackers[m[1]].update(dets[m[0], :4])

        for i in unmatched_dets: 
            trk = KalmanBoxTracker(dets[i, :4])
            self.trackers.append(trk)

        ret = [] 

        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            
            # Only return confirmed tracks
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                # Find matching detection for class info
                for det in detections:
                    if self._iou(d, det[:4]) > 0.5:
                        ret.append([
                            int(d[0]), int(d[1]), int(d[2]), int(d[3]),  # bbox
                            trk.id,  # track_id
                            float(det[4]),  # confidence
                            int(det[5]),  # class_id
                            str(det[6])  # class_name
                        ])
                        break
            
            # Remove dead tracklet
            if trk.time_since_update > self.max_age:
                self.trackers.remove(trk)
        
        return ret

    @staticmethod 
    def _iou(bbox1, bbox2):
        """Calculate Intersection over Union"""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        
        bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        
        union_area = bbox1_area + bbox2_area - inter_area
        
        return inter_area / (union_area + 1e-6)
    
    def _associate_detections_to_trackers(self, detections, trackers, iou_threshold=0.3):
        """Match detections to existing tracks using IOU"""
        if len(trackers) == 0:
            return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)
        
        # Calculate IOU matrix
        iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
        for d, det in enumerate(detections):
            for t, trk in enumerate(trackers):
                iou_matrix[d, t] = self._iou(det[:4], trk[:4])
        
        # Hungarian algorithm for assignment (simple greedy for now)
        matched_indices = []
        
        if min(iou_matrix.shape) > 0:
            a = (iou_matrix > iou_threshold).astype(np.int32)
            if a.sum(1).max() == 1 and a.sum(0).max() == 1:
                matched_indices = np.stack(np.where(a), axis=1)
            else:
                # Greedy matching
                for _ in range(min(len(detections), len(trackers))):
                    if iou_matrix.max() < iou_threshold:
                        break
                    d, t = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
                    matched_indices.append([d, t])
                    iou_matrix[d, :] = 0
                    iou_matrix[:, t] = 0
        
        matched_indices = np.array(matched_indices)
        
        unmatched_detections = []
        for d in range(len(detections)):
            if len(matched_indices) == 0 or d not in matched_indices[:, 0]:
                unmatched_detections.append(d)
        
        unmatched_trackers = []
        for t in range(len(trackers)):
            if len(matched_indices) == 0 or t not in matched_indices[:, 1]:
                unmatched_trackers.append(t)
        
        return matched_indices, np.array(unmatched_detections), np.array(unmatched_trackers)