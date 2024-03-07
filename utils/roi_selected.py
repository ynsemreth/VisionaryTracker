import cv2
import numpy as np

from utils.detections import draw
from utils.functions import add_weighted_heat,save_tracking_results,calculate_overlap

lines = {}
arrow_lines = []
arrow_line_length = 50
tracking_id = None
frame_id = 0
heatmap_accumulator = None


def detection_object(detections,detected_frame):
    global frame_id
    global heatmap_accumulator
    frame_width, frame_height = int(1280), int(720)
    if heatmap_accumulator is None:
        heatmap_accumulator = np.zeros((frame_height, frame_width), dtype=np.float32)
    for detection in detections:
        color = (np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255))
                
        if 'id' in detection:
            detection_id = detection['id']

            if detection_id not in lines:
                detection['color'] = color
                lines[detection_id] = {'points':[], 'arrows':[], 'color':color}
            else:
                detection['color'] = lines[detection_id]['color']
                    
            lines[detection_id]['points'].append(np.array([detection['x'] + detection['width']/2, detection['y'] + detection['height']/2], np.int32))
            points = lines[detection_id]['points']

            if len(points) >= 2:
                arrow_lines = lines[detection_id]['arrows']
                if len(arrow_lines) > 0:
                    distance = np.linalg.norm(points[-1] - arrow_lines[-1]['end'])
                    if distance >= arrow_line_length:
                        start = np.rint(arrow_lines[-1]['end'] - ((arrow_lines[-1]['end'] - points[-1])/distance)*10).astype(int)
                        arrow_lines.append({'start':start, 'end':points[-1]})
                else:
                    distance = 0
                    arrow_lines.append({'start':points[-2], 'end':points[-1]})
                        
        x_center = detection['x'] + detection['width'] // 2
        y_center = detection['y'] + detection['height'] // 2
        heatmap_accumulator = add_weighted_heat(heatmap_accumulator, x_center, y_center, weight=1, size=20)     

    for line in lines.values():
        arrow_lines = line['arrows']
        for arrow_line in arrow_lines:
            detected_frame = cv2.arrowedLine(detected_frame, arrow_line['start'], arrow_line['end'], line['color'], 2, line_type=cv2.LINE_AA)

    heatmap(heatmap_accumulator)
    frame_id += 1
    detected_frame = draw(detected_frame, detections)

    save_tracking_results(detections, frame_id,file_tracker=False)

    return detected_frame 

def detection_roi_single(detections,roi,detected_frame):
    global frame_id
    global tracking_id
    if tracking_id is not None:
        detection_to_track = [d for d in detections if d.get('id') == tracking_id]
        if detection_to_track:
            best_detection = detection_to_track[0]  
        else:
            best_detection = None 
    else:
        best_overlap = 0
        best_detection = None
        for detection in detections:
            current_overlap = calculate_overlap(roi, detection)
            if current_overlap > best_overlap:
                best_overlap = current_overlap
                best_detection = detection
        if best_detection:
            tracking_id = best_detection['id']
                
    if best_detection is not None:
        tracking_id = best_detection['id'] 
        print(tracking_id)
        if tracking_id not in lines:
            color = (np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255))
            best_detection['color'] = color
            lines[tracking_id] = {'points': [], 'arrows': [], 'color': color}
        else:
            best_detection['color'] = lines[tracking_id]['color']

        x_center = best_detection['x'] + best_detection['width'] / 2
        y_center = best_detection['y'] + best_detection['height'] / 2
        lines[tracking_id]['points'].append(np.array([x_center, y_center], np.int32))
            
        points = lines[tracking_id]['points']
        if len(points) >= 2:
            arrow_lines = lines[tracking_id]['arrows']
            if len(arrow_lines) > 0:
                distance = np.linalg.norm(points[-1] - arrow_lines[-1]['end'])
                if distance >= arrow_line_length:
                    start = np.rint(arrow_lines[-1]['end'] - ((arrow_lines[-1]['end'] - points[-1]) / distance) * 10).astype(int)
                    arrow_lines.append({'start': start, 'end': points[-1]})
            else:
                arrow_lines.append({'start': points[-2], 'end': points[-1]})
            
    for line_key, line_value in lines.items():
        if line_key == tracking_id: 
            arrow_lines = line_value['arrows']
            for arrow_line in arrow_lines:
                detected_frame = cv2.arrowedLine(detected_frame, tuple(arrow_line['start']), tuple(arrow_line['end']), line_value['color'], 2, line_type=cv2.LINE_AA)
    frame_id += 1
    detection_to_draw = [detection for detection in detections if detection.get('id') == tracking_id]
    if detection_to_draw:
        detected_frame = draw(detected_frame, detection_to_draw,tracking_id=tracking_id)
        save_tracking_results(detection_to_draw, frame_id)

    return detected_frame
    
def detection_roi_multi(detections , rois,detected_frame):

    initial_tracker_ids = {}  
    tracker_ids = {}

    for roi_index, roi in enumerate(rois):
        best_overlap = 0
        best_detection = None
        for detection in detections:
            overlap = calculate_overlap(roi, detection)
            if overlap > best_overlap:
                best_overlap = overlap
                best_detection = detection
    
        if best_detection:
            if roi_index not in initial_tracker_ids:
                initial_tracker_ids[roi_index] = best_detection['id']
            tracking_id = initial_tracker_ids[roi_index] 
            best_detection['id'] = tracking_id 
            tracker_ids[roi_index] = best_detection

            if tracking_id not in lines:
                color = (np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255))
                best_detection['color'] = color
                lines[tracking_id] = {'points': [], 'arrows': [], 'color': color}
            else:
                best_detection['color'] = lines[tracking_id]['color']

            x_center = best_detection['x'] + best_detection['width'] / 2
            y_center = best_detection['y'] + best_detection['height'] / 2
            lines[tracking_id]['points'].append(np.array([x_center, y_center], np.int32))
        
            points = lines[tracking_id]['points']
            if len(points) >= 2:
                arrow_lines = lines[tracking_id]['arrows']
                if len(arrow_lines) > 0:
                    distance = np.linalg.norm(points[-1] - arrow_lines[-1]['end'])
                    if distance >= arrow_line_length:
                        start = np.rint(arrow_lines[-1]['end'] - ((arrow_lines[-1]['end'] - points[-1]) / distance) * 10).astype(int)
                        arrow_lines.append({'start': start, 'end': points[-1]})
                else:
                    arrow_lines.append({'start': points[-2], 'end': points[-1]})

    for line_key, line_value in lines.items():
        arrow_lines = line_value['arrows']
        for arrow_line in arrow_lines:
            detected_frame = cv2.arrowedLine(detected_frame, tuple(arrow_line['start']), tuple(arrow_line['end']), line_value['color'], 2, line_type=cv2.LINE_AA)
    
    for detection in tracker_ids.values():
        detected_frame = draw(detected_frame, [detection], tracking_id=detection['id'])
        save_tracking_results([detection], frame_id, file_tracker=True)

    return detected_frame

def heatmap(heatmap_accumulator):

    heatmap_blurred = cv2.GaussianBlur(heatmap_accumulator, (51, 51), 0)
    heatmap_normalized = cv2.normalize(heatmap_blurred, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    heatmap_color = cv2.applyColorMap(heatmap_normalized.astype(np.uint8), cv2.COLORMAP_JET)

    cv2.imwrite('./result/isi_haritası.jpg', heatmap_color)