# AUV Net Defect Detection and 3D Localization

This repository contains the ROS packages developed for the Final Degree Project:

**Autonomous 3D Localization of Fish Farm Net Defects for AUV-Based Inspection and Intervention**

The objective of the system is to detect holes in underwater fish-farm nets, segment the damaged region, estimate its 3D position from stereo vision, and use this information to guide the Girona500 AUV during the final approach.

The perception pipeline estimates the distance between the stereo camera and the detected defect. This distance is then used by the control node to move the vehicle towards the hole and stop at a predefined safety distance of approximately 1 meter.

### Description

First, YOLO detects the defect in the left and right rectified stereo images. The resulting bounding boxes are used as prompts for FastSAM, which generates segmentation masks of the damaged region. These masks are applied to the stereo pair so that disparity is computed mainly around the relevant area.

The disparity information is then used to estimate the 3D position of the defect with respect to the camera. Finally, the 2D detections and the estimated distance to the defect are used to center the vehicle and approach the defect until the desired safety distance is reached.


## Repository structure

auv-net-defect-detection-localization-tandem/

├── net_hole_detector/   
├── tandem/             
├── .gitignore

└── README.md

Each package (net_hole_detector and tandem) contains its own README file with a more detailed description of its internal structure, launch files, scripts and ROS topics.


#### net_hole_detector directory

Contains the perception pipeline used to detect and localize the defect.

Main functions:

YOLO-based 2D defect detection.
FastSAM-based segmentation of the damaged region.
Masked stereo image generation.
Stereo disparity computation.
3D position estimation of the detected defect.

Main output topic:

/net_hole_detector/stereo_detections_3d

More details are provided in:

net_hole_detector/README.md



#### tandem directory

Contains the control node used for the Girona500 AUV movement and approach behavior.

The controller uses the perception output to:

Align the vehicle with the detected defect.
Follow a predefined inspection path.
Use the estimated 3D distance and YOLO detections to regulate the final approach.
Stop the vehicle at a predefined safety distance from the net.

Main output topic:

/girona500/controller/body_velocity_req

More details are provided in:

tandem/README.md



## Main execution order

The complete system is launched step by step:

1. Simulation environment
roslaunch cola2_stonefish girona500_windturbine.launch

2. Stereo processing
roslaunch net_hole_detector stereo_processing.launch

3. YOLO 2D detection
roslaunch net_hole_detector stereo_bboxes.launch

4. FastSAM segmentation
roslaunch net_hole_detector fastsam_stereo_dual.launch

5. Masked stereo image generation
roslaunch net_hole_detector masked_stereo_images.launch

6. Masked stereo disparity
roslaunch net_hole_detector masked_stereo_processing.launch

7. 3D defect localization
roslaunch net_hole_detector masked_stereo_z.launch

8. AUV final approach control
roslaunch tandem girona500.launch



This repository contains the TFG-specific ROS packages used for perception and control. Generated catkin folders such as build/, devel/, install/ and logs/ are intentionally excluded from the repository.
