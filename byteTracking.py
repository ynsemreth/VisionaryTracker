import argparse
from algorithm.object_detector import YOLOv7
import cv2
from utils.detections import draw
from collections import Counter
import numpy as np

parser = argparse.ArgumentParser(description='YOLOv7 ile nesne tespiti ve takibi')
parser.add_argument('--video', type=str, default='', help='Video dosyasının yolu. Boş bırakılırsa, varsayılan kamera kullanılır.')
args = parser.parse_args()

yolov7 = YOLOv7()
yolov7.load('coco.weights', classes='coco.yaml', device='cpu')

video = cv2.VideoCapture(args.video if args.video else 0)

if not video.isOpened():
    print('[!] Video açılırken bir hata oluştu.')
    exit()

frame_width, frame_height = int(video.get(3)), int(video.get(4))
heatmap_accumulator = np.zeros((frame_height, frame_width), dtype=np.float32)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (int(video.get(3)), int(video.get(4))))

print('[+] Video takip ediliyor...\n')

def calculate_overlap(roi, detection):
    x1, y1, w1, h1 = roi
    x2, y2, w2, h2 = detection['x'], detection['y'], detection['width'], detection['height']
    
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1+w1, x2+w2)
    y_bottom = min(y1+h1, y2+h2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    roi_area = w1 * h1
    detection_area = w2 * h2
    union_area = roi_area + detection_area - intersection_area
    iou = intersection_area / union_area
    return iou

def save_tracking_results(detections, frame_id, file_path='./result/tracking_results_byte.txt', file_tracker=False):
    with open(file_path, 'a') as file:
        if file_tracker == False:
            class_counts = Counter([det['class'] for det in detections])
            summary = f"Frame {frame_id}: " + ", ".join([f"{cls}: {count}" for cls, count in class_counts.items()])
            file.write(summary + "\n")
        else:
            for detection in detections:
                class_name = detection['class']
                tracking_id = detection['id']
                bbox = (detection['x'], detection['y'], detection['width'], detection['height'])
                bbox_str = f"{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}"
                line = f"Frame {frame_id}, ID: {tracking_id}, Class: {class_name}, BBox: {bbox_str}\n"
                file.write(line)
            file.write("Takip Edilen Nesnenin Tracker Bilgileri:\n")
            
def add_weighted_heat(heatmap, center_x, center_y, weight=1, size=10):
    d = np.dstack(np.mgrid[-size//2:size//2, -size//2:size//2])
    g = np.exp(-((d ** 2).sum(axis=2)) / (2.0 * size))
    g /= g.max()

    start_x = max(center_x - size//2, 0)
    end_x = min(center_x + size//2, heatmap.shape[1])
    start_y = max(center_y - size//2, 0)
    end_y = min(center_y + size//2, heatmap.shape[0])

    heatmap[start_y:end_y, start_x:end_x] += g[:end_y-start_y, :end_x-start_x] * weight

    return heatmap

roi_selected = False
multi_roi_selection = False 
tracking_id = None
track_mode = False
frame_id = 0  
lines = {}
arrow_lines = []
arrow_line_length = 50
rois = []  

while True:
    ret, frame = video.read()
    if not ret:
        print("[!] Video akışından kare alınamadı.")
        break

    if not track_mode:
        detections = yolov7.detect(frame, track=True)
        detected_frame = frame
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
            
        frame = draw(detected_frame, detections)
        save_tracking_results(detections, frame_id,file_tracker=False)
    
    elif roi_selected and track_mode:
        detections = yolov7.detect(frame, track=True)
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
                    frame = cv2.arrowedLine(frame, tuple(arrow_line['start']), tuple(arrow_line['end']), line_value['color'], 2, line_type=cv2.LINE_AA)

        detection_to_draw = [detection for detection in detections if detection.get('id') == tracking_id]
        if detection_to_draw:
            frame = draw(frame, detection_to_draw,tracking_id=tracking_id)
            save_tracking_results(detection_to_draw, frame_id)

    elif multi_roi_selection and track_mode:
        detections = yolov7.detect(frame, track=True)
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
                tracker_ids[roi_index] = best_detection
    
        for detection in tracker_ids.values():
            frame = draw(frame, [detection], tracking_id=detection['id'])
            save_tracking_results([detection], frame_id, file_tracker=True)

    heatmap_blurred = cv2.GaussianBlur(heatmap_accumulator, (51, 51), 0)
    heatmap_normalized = cv2.normalize(heatmap_blurred, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    heatmap_color = cv2.applyColorMap(heatmap_normalized.astype(np.uint8), cv2.COLORMAP_JET)

    cv2.imwrite('final_heatmap.jpg', heatmap_color)
    cv2.imshow('ByteTracker', frame)
    out.write(frame)

    frame_id += 1 

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    
    elif key == ord('s') and not roi_selected:
        roi = cv2.selectROI("ByteTracker", frame, False)
        print(f"Çizilen Roi'nin Koordinatları: {roi}")
        if roi[2] > 0 and roi[3] > 0: 
            roi_selected = True
            track_mode = True
    elif key == ord('f'):  
        while True:  
            roi = cv2.selectROI("ByteTracker", frame, False)
            if roi[2] > 0 and roi[3] > 0:  
                rois.append(roi)
                print(f"Çizilen Roi'nin Koordinatları: {roi}")
                multi_roi_selection = True
                track_mode = True
            else:
                print("[!] Geçersiz ROI, atlanıyor.")
                break 
            key = cv2.waitKey(0) & 0xFF  
            if key == ord('f'):  
                print(rois)
                continue
            elif key == ord('g'):  
                break
    elif key == ord('d'):
        track_mode = False
        roi_selected = False
        print("[+] Takip modu durduruldu.")

video.release()
out.release()
cv2.destroyAllWindows()
yolov7.unload()