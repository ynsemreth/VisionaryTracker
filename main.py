import cv2
import argparse
from models.object_detector import YOLOv9
from utils.roi_selected import detection_object, detection_roi_single, detection_roi_multi
from utils.logo import print_logo

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def main(video_path):
    print_logo()
    roi_selected = False
    multi_roi_selection = False 
    track_mode = False 
    rois = []  

    yolov9 = YOLOv9()
    yolov9.load('./model/carandperson/best.pt', classes='./model/carandperson/best.yaml', device='cpu')

    video = cv2.VideoCapture(video_path if video_path else 0)

    if not video.isOpened():
        print('[!] Video açılırken bir hata oluştu.')
        exit()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('./result/output.mp4', fourcc, 20.0, (int(video.get(3)), int(video.get(4))))

    print('[+] Video takip ediliyor...\n')

    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    
    while True:
        ret, frame = video.read()
        if not ret:
            print("[!] Video akışından kare alınamadı.")
            break
        
        detections = yolov9.detect(frame, track=True)
        detected_frame = frame

        if not track_mode:
            frame = detection_object(detections,detected_frame,frame_height,frame_width)

        elif roi_selected and track_mode:
            frame = detection_roi_single(detections,roi,detected_frame,frame_height,frame_width)
            

        elif multi_roi_selection and track_mode:
            frame = detection_roi_multi(detections,rois,detected_frame,frame_height,frame_width)

        cv2.imshow('VisionaryTrack', frame)
        out.write(frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        
        elif key == ord('s') and not roi_selected:
            roi = cv2.selectROI("VisionaryTrack", frame, False)
            print(f"Çizilen Roi'nin Koordinatları: {roi}")
            if roi[2] > 0 and roi[3] > 0: 
                roi_selected = True
                track_mode = True
        elif key == ord('f'):  
            while True:  
                roi = cv2.selectROI("VisionaryTrack", frame, False)
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
    yolov9.unload()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='YOLOv7 ile nesne tespiti ve takibi')
    parser.add_argument('--video', type=str, default='', help='Video dosyasının yolu. Boş bırakılırsa, varsayılan kamera kullanılır.')
    args = parser.parse_args()

    main(args.video)
