import numpy as np
import cv2 
from collections import Counter
from skimage.feature import hog


def add_weighted_heat(heatmap, center_x, center_y, weight=1, size=10):
    d = np.dstack(np.mgrid[-size//2:size//2, -size//2:size//2])
    g = np.exp(-((d ** 2).sum(axis=2)) / (2.0 * size))
    g /= g.max()

    start_x = int(max(center_x - size//2, 0))
    end_x = int(min(center_x + size//2, heatmap.shape[1]))
    start_y = int(max(center_y - size//2, 0))
    end_y = int(min(center_y + size//2, heatmap.shape[0]))

    heatmap[start_y:end_y, start_x:end_x] += g[:end_y-start_y, :end_x-start_x] * weight

    return heatmap

frame_id = 0

def save_hog_features_and_image(frame, hog_features_path, cropped_images_path,hog_exract_path):
    global frame_id 
    fd, hog_image = hog(frame, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualize=True, channel_axis=-1)
    
    np.savetxt(f"{hog_features_path}/frame_{frame_id}.txt", fd, fmt='%f')
    cv2.imwrite(f"{cropped_images_path}/frame_{frame_id}.jpg", frame)
    cv2.imwrite(f"{hog_exract_path}/hog_frame_{frame_id}.jpg", hog_image)

    frame_id += 1

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
