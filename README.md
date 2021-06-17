# Spot: Spot: A Computer Vision System for Detecting Directional Changes of Pedestrians 

- Rotation and scaling
	- Interpolation of small resolution video, rotation of video
- Preprocessing
	- Foreground and background segmentation using KNN algorithm
	- FG preprocessing: CLAHE, Median filtering, Morphological transformation, Detail enhancement
	- BG preprocesing: Edge preserving filter
- Object Detection and tracking
	- [YOLO v4, Deep SORT, Tensorflow](https://github.com/theAIGuysCode/yolov4-deepsort)
- Direction labeling
	- Centroid comparison
- Directional change spotting
	- Mode 0: Spotting objects that are significantly changing directions
	- Mode 1: Spotting objects moving in a different direction from the general direction of objects in each frame
	- Mode 2: Spotting objects going against the direction set by the user

Refer to spot.py for the frags. Sample run:

    python spot.py --rotate True --scale True --preprocess True --output "output.mp4" --mode 1 --video "input.mp4"
