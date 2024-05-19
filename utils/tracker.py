import numpy as np
from byte_track_Utils.detections import draw
from utils.functions import *

tracking_id = None
initial_tracker_ids = {}
tracker_ids = {}

class Tracker:
    def __init__(self, roi, frame_height, frame_width, detections, detections_frame, rois):
        self.detections = detections
        self.detected_frame = detections_frame
        self.roi = roi
        self.rois = rois
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.heatmap_accumulator = np.zeros((frame_height, frame_width), dtype=np.float32)
        self.future_extraction = None
        self.frame_id = 0
        self.heatmap = Heatmap()

    def detect(self):
        self.frame_id += 1
    
        if self.heatmap_accumulator is None:
            self.heatmap_accumulator = np.zeros((self.frame_height, self.frame_width), dtype=np.float32)

        for detection in self.detections:
            x_center = detection['x'] + detection['width'] // 2
            y_center = detection['y'] + detection['height'] // 2

            self.heatmap_accumulator = self.heatmap.add_weighted_heat(self.heatmap_accumulator, x_center, y_center)
            self.heatmap.heatmap(self.heatmap_accumulator)

        self.detected_frame = draw(self.detected_frame, self.detections)
        return self.detected_frame


    def detect_single(self):
        global tracking_id
        
        if tracking_id:
            detection_to_track = [d for d in self.detections if d.get('id') == tracking_id]
            best_detection = detection_to_track[0] if detection_to_track else None
        else:
            best_detection = None 
            best_overlap = 0
            for detection in self.detections:
                print(self.roi)
                current_overlap = function.calculate_overlap(self.roi, detection)
                print(current_overlap)
                if (current_overlap > best_overlap):
                    best_overlap = current_overlap
                    best_detection = detection
            if best_detection:
                tracking_id = best_detection['id']
                print("İlk Nesnenin Takip ID'si:", tracking_id)
            
        if best_detection is not None:
            print(f"En iyi tespit bulundu. Takip ID'si: {tracking_id}")
        
            x_center = best_detection['x'] + best_detection['width'] / 2
            y_center = best_detection['y'] + best_detection['height'] / 2
            
            self.heatmap_accumulator = self.heatmap.add_weighted_heat(self.heatmap_accumulator, x_center, y_center)
            self.heatmap.heatmap(self.heatmap_accumulator)

            self.frame_id += 1
            detection_to_draw = [best_detection]
            print(best_detection)
            
            if detection_to_draw:
                self.detected_frame = draw(self.detected_frame, detection_to_draw, tracking_id=tracking_id)
                function.save_tracking_results(detection_to_draw, self.frame_id)

        return self.detected_frame

    def detect_multi(self):
        global initial_tracker_ids
        global tracker_ids

        if not tracker_ids:
            for roi_index, roi in enumerate(self.rois):
                best_overlap = 0
                best_detection = None
                for detection in self.detections:
                    overlap = function.calculate_overlap(roi, detection)
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_detection = detection
                        print(f"En iyi tespit bulundu. Takip ID'si: {best_detection['id']}")
                                        
                if best_detection:
                    multi_tracking_id = initial_tracker_ids.get(roi_index, None)
                    if multi_tracking_id is None:
                        multi_tracking_id = best_detection['id']
                        initial_tracker_ids[roi_index] = multi_tracking_id
                    else:
                        best_detection['id'] = multi_tracking_id

                    tracker_ids[roi_index] = best_detection

                    print(f"ROI index {roi_index} için takip ID'si: {multi_tracking_id}")

                    x_center = best_detection['x'] + best_detection['width'] / 2
                    y_center = best_detection['y'] + best_detection['height'] / 2

                    self.heatmap_accumulator = self.heatmap.add_weighted_heat(self.heatmap_accumulator, x_center, y_center)
                    self.heatmap.heatmap(self.heatmap_accumulator)

        self.frame_id += 1

        for roi_index, detection in tracker_ids.items():
            detection_to_track = next((d for d in self.detections if d.get('id') == detection['id']), None)
            if detection_to_track:
                tracker_ids[roi_index] = detection_to_track

                x_center = detection_to_track['x'] + detection_to_track['width'] / 2
                y_center = detection_to_track['y'] + detection_to_track['height'] / 2

                self.heatmap_accumulator = self.heatmap.add_weighted_heat(self.heatmap_accumulator, x_center, y_center)
                self.heatmap.heatmap(self.heatmap_accumulator)

        for detection in tracker_ids.values():
            self.detected_frame = draw(self.detected_frame, [detection], tracking_id=detection['id'])
            function.save_tracking_results([detection], self.frame_id, file_tracker=True)

        return self.detected_frame