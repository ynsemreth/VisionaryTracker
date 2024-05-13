import numpy as np
import cv2 
from collections import Counter

import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image

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
    
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    win_size = (16,64) 
    cell_size = (8, 8)
    block_size = (16, 16)
    block_stride = (8, 8)
    num_bins = 9

    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, num_bins)


    hog_descriptor = hog.compute(img_gray)

    print('HOG Descriptor:', hog_descriptor)
    print('HOG Descriptor shape:', hog_descriptor.shape)
    
    # Save HOG features to a text file
    np.savetxt(f"{hog_features_path}/frame_{frame_id}.txt", hog_descriptor.squeeze())

    # Save the cropped frame as an image
    cv2.imwrite(f"{cropped_images_path}/frame_{frame_id}.jpg", frame)

    # Save the HOG descriptor visualization as an image (optional)
    hog_image = hog.compute(img_gray, (1, 1))  # Compute the HOG image
    cv2.imwrite(f"{hog_exract_path}/hog_frame_{frame_id}.jpg", hog_image)

    frame_id += 1
    
    return hog_descriptor

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

def extract_resnet50_features(image):
    resnet50 = models.resnet50(pretrained=True)

    resnet50.eval()
    
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(Image.fromarray(image))
    input_batch = input_tensor.unsqueeze(0)  
    if torch.cuda.is_available():
        input_batch = input_batch.to('cpu')
        resnet50.to('cpu')
    with torch.no_grad():
        features = resnet50(input_batch)
    features = features.squeeze().cpu().numpy()

    return features
def calculate_feature_distance(feature1, feature2):
    return np.linalg.norm(feature1 - feature2)