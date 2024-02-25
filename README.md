# Byte Tracking User Guide

This application tracks objects of your choice in real time or from a video, using Byte Tracking and YOLOv7, and updates the object's ID.

Byte Tracking, also known as Multiple Object Tracking (MOT), is a task in computer vision that involves tracking the movements of multiple objects over time within a video sequence. The aim is to determine the identity, location, and trajectory of each object in the video, even in cases where objects are partially or fully occluded by other objects in the scene.

Byte Tracking typically proceeds in two stages:

## Object Detection

- This stage involves identifying all potential objects of interest in the current frame using object detectors like YOLOv7.

## Object Association

- This stage involves linking objects detected in the current frame with their corresponding objects from previous frames, referred to as tracklets. Object or instance association is usually achieved by predicting the object’s location in the current frame based on previous frames' tracklets using the Kalman Filter, followed by one-to-one linear assignment typically using the Hungarian Algorithm to minimize the total differences between the matching results.

ByteTrack is a specific implementation of MOT that also considers low-accuracy bounding boxes. It uses a motion model that manages a queue called tracklets to store objects being tracked and performs tracking and matching between bounding boxes with low confidence values. The matching process utilizes an algorithm called BYTE.

This technology has numerous practical applications, such as surveillance, robotics, sports analytics, and medical imaging. For example, in surveillance, MOT can be used to detect and track suspicious behavior in a crowd or monitor the movement of vehicles in a parking lot. In robotics, it can assist in guiding autonomous vehicles, identifying obstacles, and planning safe routes through a dynamic environment.

### To Work With REALTIME And Made-Video

```sh
python tracking.py

python tracking.py --video ./videos/cat.mp4
