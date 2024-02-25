# VISIONARY TRACKER

This project leverages the YOLOv7 algorithm for real-time object detection and tracking in video streams or from a webcam. It includes functionalities for drawing detection bounding boxes, tracking object movements, generating heatmaps for tracking visualization, and saving tracking results.

## Features

- **Object Detection**: Utilizes YOLOv7 for detecting objects in each frame of the video.
- **Object Tracking**: Tracks objects across frames, visualizing their paths.
- **Heatmap Generation**: Generates a heatmap based on the tracked paths of objects.
- **Customizable Video Input**: Supports processing video files or live webcam streams.
- **Result Saving**: Saves a summary of detected objects and their tracking information.

## How It Works

### Video Input Handling
- The script accepts a video file path through the `--video` argument. If no path is provided, it defaults to using the webcam (`0`).

### Object Detection and Tracking
- YOLOv7 is loaded with pre-trained weights and configured to detect objects defined in `coco.yaml`.
- Detected objects are tracked across frames, with their movement paths visualized.

### Heatmap Visualization
- A heatmap is generated and updated in real-time to visualize the frequency of object movements across different areas of the frame.

![Heatmap Visualization](https://github.com/ynsemreth/VisionaryTracker/blob/main/final_heatmap.jpg)


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
python byteTracking.py

python byteTracking.py --video ./videos/examples.mp4
```

4. Use keyboard commands during execution to interact with the tracking process.

## Conclusion

This project showcases the power of YOLOv7 for real-time object detection and tracking, enhanced with heatmap visualization for movement analysis. It's adaptable for various applications, from surveillance to sports analytics.

