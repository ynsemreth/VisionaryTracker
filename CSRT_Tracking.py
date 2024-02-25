import argparse
from algorithm.object_detector import YOLOv7
import cv2
from collections import Counter


parser = argparse.ArgumentParser(description='YOLOv7 ile nesne tespiti ve takibi')
parser.add_argument('--video', type=str, default='', help='Video dosyasının yolu. Boş bırakılırsa, varsayılan kamera kullanılır.')
args = parser.parse_args()

yolov7 = YOLOv7()
yolov7.load('coco.weights', classes='coco.yaml', device='cpu')

video = cv2.VideoCapture(args.video if args.video else 0)

if not video.isOpened():
    print('[!] Video açılırken bir hata oluştu.')
    exit()

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (int(video.get(3)), int(video.get(4))))

print('[+] Video takip ediliyor...\n')

def roi_intersects_detection(roi, detection):
    x, y, w, h = roi 
    dx, dy, dw, dh = detection['x'], detection['y'], detection['width'], detection['height'] 
    if (x < dx + dw) and (x + w > dx) and (y < dy + dh) and (y + h > dy):
        return True 
    return False  

def save_tracking_results(detections, frame_id, file_path='./result/tracking_result_kfc.txt'):
    with open(file_path, 'a') as file:
        for det in detections:
            class_name = det['class']
            bbox = (det['x'], det['y'], det['width'], det['height'])
            bbox_str = f"{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}"
            line = f"Frame {frame_id}, Class: {class_name}, BBox: {bbox_str}\n"
            file.write(line)
            
def summarize_detections(detections, frame_id, file_path='./result/tracking_result_kfc_total.txt'):
    class_counts = Counter([det['class'] for det in detections])
    with open(file_path, 'a') as file:
        summary = f"Frame {frame_id}: " + ", ".join([f"{cls}: {count}" for cls, count in class_counts.items()])
        file.write(summary + "\n")

frame_id = 0
roi_selected = False
tracking_mode = False 
tracker = cv2.TrackerKCF_create() 

while True:
    ret, frame = video.read()
    if not ret:
        print("[!] Video akışından kare alınamadı.")
        break

    if not tracking_mode:
        detections = yolov7.detect(frame)
        for detection in detections:
            x, y, w, h = detection['x'], detection['y'], detection['width'], detection['height']
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            if 'class' in detection:
                cv2.putText(frame, detection['class'], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (36,255,12), 2)
    
    summarize_detections(detections, frame_id)
    
    if roi_selected and tracking_mode:
        success, box = tracker.update(frame)
        if success:
            p1 = (int(box[0]), int(box[1]))
            p2 = (int(box[0] + box[2]), int(box[1] + box[3]))
            tracked_region = frame[p1[1]:p2[1], p1[0]:p2[0]]
        
            if tracked_region.size != 0:
                detections = yolov7.detect(tracked_region)
                if detections:
                    detected_object = max(detections, key=lambda x: x['width'] * x['height'])
                    save_tracking_results([detected_object], frame_id)

                    bx, by, bw, bh = detected_object['x'], detected_object['y'], detected_object['width'], detected_object['height']
                    cv2.rectangle(frame, (p1[0] + bx, p1[1] + by), (p1[0] + bx + bw, p1[1] + by + bh), (0, 0, 255), 2)
                    cv2.putText(frame, detected_object['class'], (p1[0] + bx, p1[1] + by - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
            
    cv2.imshow('Tespit Edilen Video', frame)
    out.write(frame)
    frame_id += 1

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s') and not roi_selected:
        roi = cv2.selectROI("Tespit Edilen Video", frame, False)
        if roi[2] > 0 and roi[3] > 0:  
            tracker.init(frame, roi)
            roi_selected = True
            tracking_mode = True 
    elif key == ord('d'):
        tracking_mode = False
        roi_selected = False

video.release()
out.release()
cv2.destroyAllWindows()
yolov7.unload()
