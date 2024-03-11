# VISIONARY TRACKER

This project leverages the YOLOv9 algorithm for real-time object detection and tracking in video streams or from a webcam. It includes functionalities for drawing detection bounding boxes, tracking object movements, generating heatmaps for tracking visualization, and saving tracking results.

## Training 

```sh
python train.py --workers 2 --device 'cpu' --batch 4 --data C:\Users\cypoi\Masaüstü\VisionaryTracker\data\carandperson\data.yaml --img 640 --cfg C:\Users\cypoi\Masaüstü\VisionaryTracker\models\detect\gelan-c.yaml --weights 'C:\Users\cypoi\Masaüstü\VisionaryTracker\gelan-c.pt' --name kisi --hyp C:\Users\cypoi\Masaüstü\VisionaryTracker\data\hyps\hyp.scratch-high.yaml --min-items 0 --epochs 10 --close-mosaic 15
```

## Features

- **Object Detection**: Utilizes YOLOv9 for detecting objects in each frame of the video.
- **Object Tracking**: Tracks objects across frames, visualizing their paths.
- **Heatmap Generation**: Generates a heatmap based on the tracked paths of objects.
- **Customizable Video Input**: Supports processing video files or live webcam streams.
- **Result Saving**: Saves a summary of detected objects and their tracking information.

## How It Works

### Video Input Handling
- The script accepts a video file path through the `--video` argument. If no path is provided, it defaults to using the webcam (`0`).

### Object Detection and Tracking
- YOLOv9 is loaded with pre-trained weights and configured to detect objects defined in `data.yaml`.
- Detected objects are tracked across frames, with their movement paths visualized.

### Single ROI Selection

To select a single ROI, run the project and wait for the video stream to start. Once the stream is live:

1. Press the `s` key to enter single ROI selection mode.
2. Click and drag the mouse over the area you wish to track.
3. Release the mouse button to finalize the ROI.
4. The system will now focus on detecting and tracking objects within this specified region.

This feature is particularly useful for isolating the tracking to a specific object or area in the scene, enhancing tracking accuracy and efficiency.

### Multiple ROI Selection

For scenarios requiring attention to several areas simultaneously, our system allows for multiple ROI selections:

1. Press the `f` key to initiate multiple ROI selection.
2. For each ROI, click and drag the mouse over the desired area and release to finalize it.
3. After selecting an ROI, press the `f` key again to continue selecting additional ROIs.
4. To complete the selection process and start tracking, press the `g` key.

Each selected ROI will be individually monitored, enabling the detection and tracking of objects across multiple areas of interest within the same frame.

### Tips for Effective ROI Selection

- Ensure clear boundaries for each ROI to avoid overlap, which could affect tracking performance.
- Use the multiple ROI feature judiciously, as tracking numerous areas simultaneously may increase computational load and impact performance.
- ROI selection can be adjusted or reset at any time by re-initiating the selection process.

By utilizing ROI selection, users can tailor the object detection and tracking process to specific needs, focusing computational resources on areas of interest and improving overall system efficiency.


### Heatmap Visualization
- A heatmap is generated and updated in real-time to visualize the frequency of object movements across different areas of the frame.
![Heatmap Visualization](https://github.com/ynsemreth/VisionaryTracker/blob/main/result/heatmap.jpg)

### Saving Results
- Detected objects and their tracking information are saved to a file, providing a summary of object counts per frame and detailed tracking data.

## Implementation Details

### Dependencies
- argparse, cv2 (OpenCV), numpy, and custom modules like `object_detector` and `utils.detections`.

### Key Functions
- `calculate_overlap`: Calculates the Intersection over Union (IoU) to assist in tracking.
- `save_tracking_results`: Saves tracking information to a text file.
- `add_weighted_heat`: Updates the heatmap based on object locations.
- Keyboard interactions (`q`, `s`, `f`, `d`) to quit, start tracking, select multiple ROIs, and disable tracking mode.

## Setup and Usage

1. Install dependencies: Ensure Python 3.x is installed along with required packages (OpenCV, NumPy).
2. Place your YOLOv7 model weights and configuration files in the project directory.
3. Run the script with the desired video input or leave it for webcam input:
```sh
python main.py

python main.py --video ./videos/examples.mp4
```

4. Use keyboard commands during execution to interact with the tracking process.

## Conclusion

This project showcases the power of YOLOv9 for real-time object detection and tracking, enhanced with heatmap visualization for movement analysis. It's adaptable for various applications, from surveillance to sports analytics.

# ALGORITMALAR : 

## HOG (Histogram of Oriented Gradients)

### Amaç: 
HOG, görüntülerdeki şekil ve doku bilgilerini yakalamak için tasarlanmış bir özellik çıkartma yöntemidir.
    
### Çalışma Prensibi:
Görüntüdeki her bir pikselin gradyan yönünü ve büyüklüğünü hesaplar. Daha sonra, bu gradyanlar belirli bir pencere içerisindeki hücrelere ayrılır ve her bir hücre için gradyan yönlerinin histogramı oluşturulur. Bu histogramlar, görüntünün yerel gradyan yapısını özetleyen güçlü ve açıklayıcı özellikler üretir.

### Kullanım Alanları: 
Yaya tespiti, araç tanıma ve insan tanıma gibi görevlerde yaygın olarak kullanılır.

## SIFT (Scale-Invariant Feature Transform)

### Amaç: 
SIFT, görüntülerdeki anahtar noktaları bulmak ve bunların özelliklerini çıkarmak için kullanılır. Bu algoritma, ölçek ve dönüşüme karşı dayanıklı özellikler sağlar.
### Çalışma Prensibi:
Görüntü üzerinde ölçek uzayı aşamaları uygulanır, potansiyel ilgi noktaları tespit edilir, ve bu noktaların çevresindeki gradyan bilgileri kullanılarak her bir nokta için benzersiz bir tanımlayıcı (descriptor) oluşturulur.
### Kullanım Alanları: 
Nesne tanıma, panoramik görüntü birleştirme ve 3D modelleme gibi çeşitli alanlarda kullanılır.

## SURF (Speeded Up Robust Features)

### Amaç: 
SURF, SIFT'in hedeflediği sorunları çözmek için tasarlanmıştır ancak daha hızlı hesaplama ve benzer veya daha iyi performans sunar.
### Çalışma Prensibi: 
Hızlı bir şekilde ilgi noktalarını tespit etmek için Hessian matris tabanlı bir yaklaşım kullanır. Bu noktalar için özellik tanımlayıcıları, ilgi noktalarının çevresindeki basit, hızlı ve etkili bir şekilde hesaplanabilir şekilde üretilir.
### Kullanım Alanları: 
SIFT'e benzer şekilde, nesne tanıma, görüntü eşleştirme ve 3D rekonstrüksiyon gibi alanlarda kullanılır.
