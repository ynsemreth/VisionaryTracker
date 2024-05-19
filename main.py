import cv2
import argparse
from models.object_detector import YOLOv9
from utils.tracker import *
from utils.logo import print_logo
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def main(video):
    print_logo()
    roi_selected = False
    multi_roi_selection = False 
    track_mode = False 
    rois = []  
    roi = None

    yolov9 = YOLOv9()
    yolov9.load('./model/carandperson/best.pt', classes='./model/carandperson/best.yaml', device='cpu')

    video = cv2.VideoCapture(video if video else 0)

    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('./result/output.mp4', fourcc, 20.0, (frame_width, frame_height))

    print('[+] Video takip ediliyor...\n')

    while True:
        ret, frame = video.read()
        if not ret:
            print("[!] Video akışından kare alınamadı.")
            break
        
        detections = yolov9.detect(frame, track=True)
        detected_frame = frame

        tracker = Tracker(detections=detections, roi=roi, frame_height=frame_height,
                                           frame_width=frame_width, detections_frame=detected_frame,rois=rois)
        
        if not track_mode:
            frame = tracker.detect()
        elif roi_selected and track_mode:
            frame = tracker.detect_single()
        elif multi_roi_selection and track_mode:
            frame = tracker.detect_multi()

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
    parser.add_argument('--video', type=str, required=True, help='Birinci video dosyasının yolu.')
    args = parser.parse_args()

    main(args.video)
