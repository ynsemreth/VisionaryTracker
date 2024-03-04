import argparse
from algorithm.object_detector import YOLOv7
import cv2

from utils.roi_selected import detection_object,detection_roi_single,detection_roi_multi

roi_selected = False
multi_roi_selection = False 
track_mode = False 
rois = []  

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

while True:
    ret, frame = video.read()
    if not ret:
        print("[!] Video akışından kare alınamadı.")
        break
    
    detections = yolov7.detect(frame, track=True)
    detected_frame = frame

    if not track_mode:
        frame = detection_object(detections,detected_frame)

    elif roi_selected and track_mode:
        frame = detection_roi_single(detections,roi,detected_frame)

    elif multi_roi_selection and track_mode:
        frame = detection_roi_multi(detections,rois,detected_frame)

    cv2.imshow('ByteTracker', frame)
    out.write(frame)

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