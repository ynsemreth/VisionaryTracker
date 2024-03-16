import cv2
import numpy as np

from byte_track_Utils.detections import draw
from utils.functions import add_weighted_heat,save_tracking_results,calculate_overlap,save_hog_features_and_image

lines = {}
arrow_lines = []
arrow_line_length = 50
tracking_id = None
frame_id = 0
heatmap_accumulator = None

def line_function(detection,detected_frame,frame_height,frame_width):
    global heatmap_accumulator
    if heatmap_accumulator is None:
        heatmap_accumulator = np.zeros((frame_height, frame_width), dtype=np.float32)
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
    heatmap(heatmap_accumulator)
    
    for line in lines.values():
        arrow_lines = line['arrows']
        for arrow_line in arrow_lines:
            detected_frame = cv2.arrowedLine(detected_frame, arrow_line['start'], arrow_line['end'], line['color'], 2, line_type=cv2.LINE_AA) 
    
    return detected_frame
    
    
def detection_object(detections,detected_frame,frame_height,frame_width):
    global frame_id
    for detection in detections:
        detected_frame = line_function(detection,detected_frame,frame_height,frame_width)
                
    frame_id += 1
    detected_frame = draw(detected_frame, detections)

    save_tracking_results(detections, frame_id,file_tracker=False)

    return detected_frame 

def detection_roi_single(detections,roi,detected_frame,frame_height,frame_width):
    global frame_id
    global tracking_id
    global heatmap_accumulator
    if heatmap_accumulator is None:
        heatmap_accumulator = np.zeros((frame_height, frame_width), dtype=np.float32)
    hog_features_path = './hog/single_track/hog_features/'
    cropped_images_path = './hog/single_track/cropped_images_hog/'
    hog_exract_path = './hog/single_track/hog_images/'
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
        
        heatmap_accumulator = add_weighted_heat(heatmap_accumulator, x_center, y_center, weight=1, size=20)   
        
    for line_key, line_value in lines.items():
        if line_key == tracking_id: 
            arrow_lines = line_value['arrows']
            for arrow_line in arrow_lines:
                detected_frame = cv2.arrowedLine(detected_frame, tuple(arrow_line['start']), tuple(arrow_line['end']), line_value['color'], 2, line_type=cv2.LINE_AA)
    frame_id += 1
    detection_to_draw = [detection for detection in detections if detection.get('id') == tracking_id]
    if detection_to_draw:
        for detection in detection_to_draw:
            x, y, w, h = detection['x'], detection['y'], detection['width'], detection['height']
            cropped_frame = detected_frame[y:y+h, x:x+w]

            save_hog_features_and_image(cropped_frame, hog_features_path, cropped_images_path,hog_exract_path)
    heatmap(heatmap_accumulator)
    if detection_to_draw:
        detected_frame = draw(detected_frame, detection_to_draw,tracking_id=tracking_id)
        save_tracking_results(detection_to_draw, frame_id)
    
    return detected_frame
    
def detection_roi_multi(detections, rois, detected_frame, frame_height, frame_width):
    global heatmap_accumulator
    if heatmap_accumulator is None:
        heatmap_accumulator = np.zeros((frame_height, frame_width), dtype=np.float32)
        
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
            tracking_id = initial_tracker_ids.get(roi_index, None)
            if tracking_id is None:
                tracking_id = best_detection['id']
                initial_tracker_ids[roi_index] = tracking_id
            else:
                best_detection['id'] = tracking_id
            
            tracker_ids[roi_index] = best_detection

            if tracking_id not in lines:
                color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
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

    cv2.imwrite('./result/heatmap.jpg', heatmap_color)
    
    
    
# Kodunuzda initial_tracker_ids ve tracker_ids adında iki farklı sözlük kullanıyorsunuz.
# initial_tracker_ids, başlangıçta takip edilen nesnelerin kimliklerini saklar, tracker_ids ise mevcut karedeki tespit edilen nesnelerin kimliklerini saklar.
# Kodumuza bakarak, her bir roi için bir tane tracker kimliği atanmasını beklersiniz. 
# Ancak, kodunuzda bu işlevsellik yerine, her bir roi için en iyi örtüşen tespiti seçip bir initial_tracker_ids sözlüğüne ekleyip, 
# sonra her bir tespiti ayrı ayrı tracker_ids sözlüğüne ekliyorsunuz. Bu nedenle, aynı nesneyi takip eden farklı tespitler için farklı kimlikler atanabilir.

# tracker_ids sözlüğüne yeni bir tespit eklerken, tüm mevcut tespitleri yazdırıyorsunuz. 
# Bu, her döngü adımında tracker_ids sözlüğünün tamamını yazdırmanıza neden olur. 
# Her döngü adımında yalnızca bir tespitin eklenmesi gerektiğini varsayarsak, bu çıktı gereksiz olabilir.

# Döngü içinde, her bir roi için en iyi tespit seçilirken, bu tespitin bir kimliği olup olmadığını kontrol ediyorsunuz. 
# Eğer yoksa, best_detection['id'] değeri None olarak atanır ve daha sonra tracker_ids sözlüğüne eklenir. 
# Ancak, initial_tracker_ids sözlüğüne eklenmez. Bu, initial_tracker_ids sözlüğünün her zaman boş kalmasına neden olabilir.
