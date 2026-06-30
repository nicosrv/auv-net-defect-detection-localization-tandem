# AUV Net Defect Detection and 3D Localization

This repository contains the ROS packages developed for the Final Degree Project:

Autonomous 3D Localization of Fish Farm Net Defects for AUV-Based Inspection and Intervention

The system detects defects in fish-farm nets using deep learning, segments the damaged region, estimates its depth from stereo vision, and provides the information required for the AUV to approach the defect safely.

## Repository structure

- net_hole_detector/
  - launch/
  - scripts/
  - msg/
  - srv/
  - config/
  - envs/
  - weights/
  - CMakeLists.txt
  - package.xml
  - setup.py

- tandem/
  - launch/
  - src/
  - CMakeLists.txt
  - package.xml

## Main packages

### net_hole_detector

This package contains the perception pipeline:

- stereo image preprocessing
- YOLO-based 2D defect detection
- FastSAM-based segmentation
- masked stereo image generation
- stereo disparity computation
- 3D depth estimation of the detected defect

### tandem

This package contains the robot approach and inspection logic used to guide the AUV according to the visual detections and the estimated distance to the defect.

## Main launch files

The final pipeline used in the TFG is executed with the following launch files.

### Simulation environment

Command:

roslaunch cola2_stonefish girona500_windturbine.launch

This launch starts the simulation environment and the Girona500 AUV. It belongs to the external Stonefish/COLA2 simulation setup and is not part of this repository.

### Stereo preprocessing

Command:

roslaunch net_hole_detector stereo_processing.launch

Main script:

corrector_camera_info.py

This launch prepares the stereo camera information required by the rest of the perception pipeline.

### YOLO 2D detection

Command:

roslaunch net_hole_detector stereo_bboxes.launch

This launch internally uses:

yolo_inference.launch
bbox_detector.py

It runs the YOLO-based detector on the left and right stereo images and publishes the detected bounding boxes.

### FastSAM segmentation

Command:

roslaunch net_hole_detector fastsam_stereo_dual.launch

Main script:

hole_segmenter_fastsam.py

This launch applies FastSAM to the stereo detections in order to obtain masks of the damaged region.

### Masked stereo image generation

Command:

roslaunch net_hole_detector masked_stereo_images.launch

Main script:

masked_stereo_image_builder.py

This launch generates masked stereo images using the segmentation masks.

### Masked stereo disparity

Command:

roslaunch net_hole_detector masked_stereo_processing.launch

This launch runs the stereo disparity computation on the masked stereo images.

### 3D depth estimation

Command:

roslaunch net_hole_detector masked_stereo_z.launch

Main script:

stereo_distance_estimator.py

This launch estimates the depth of the detected defect from the masked stereo disparity.

### AUV approach and inspection logic

Command:

roslaunch tandem girona500.launch

Main script:

ri.py

This launch runs the control logic used for the visual approach and inspection behavior of the AUV.

## Final execution order

A typical execution order is:

1. Start the simulation environment:

roslaunch cola2_stonefish girona500_windturbine.launch

2. Start the stereo preprocessing:

roslaunch net_hole_detector stereo_processing.launch

3. Start the YOLO stereo detections:

roslaunch net_hole_detector stereo_bboxes.launch

4. Start the FastSAM stereo segmentation:

roslaunch net_hole_detector fastsam_stereo_dual.launch

5. Start the masked stereo image generation:

roslaunch net_hole_detector masked_stereo_images.launch

6. Start the masked stereo disparity computation:

roslaunch net_hole_detector masked_stereo_processing.launch

7. Start the 3D depth estimation:

roslaunch net_hole_detector masked_stereo_z.launch

8. Start the AUV approach and inspection logic:

roslaunch tandem girona500.launch

## Main scripts used in the final system

- net_hole_detector/scripts/corrector_camera_info.py
- net_hole_detector/scripts/bbox_detector.py
- net_hole_detector/scripts/hole_segmenter_fastsam.py
- net_hole_detector/scripts/masked_stereo_image_builder.py
- net_hole_detector/scripts/stereo_distance_estimator.py
- tandem/src/ri.py

## Model weights

The trained and required model weights are stored in:

net_hole_detector/weights/

This folder includes the YOLO weights used for defect detection and the FastSAM model used for segmentation.

## Notes

This repository contains the TFG-specific ROS packages. External simulation and robot packages, such as Stonefish, COLA2, and the Girona500 simulation setup, must be available in the ROS workspace for the complete system to run.

The final submitted version is marked with the tag:

v1.0-tfg-submission

