# net_hole_detector
ROS Noetic Package to detect automatically broken holes in nets and estimate the real distance between the camera and the holes.

This system combines **YOLO-World** for object detections (broken holes) and **Classical Computer Vision** for the escale estimation of the net (meter/pixel) based on known shape and real size of the net.

## Project Structure

```text
.
├── config/                 # Calibration file and tuning parameters
│   ├── UJI_tuning_params.yaml  <-- ¡IMPORTANT! Main Configuration
│   └── right_laser.yaml        <-- Intrinsec Camera Calibration
├── launch/                 # ROS launches
├── msg/                    # Messages definition (NetStats, Detection3D...)
├── scripts/                # Executable Nodes (Python)
├── src/                    # Internal Libraries (Estimators, Geometry)
├── weights/                # YOLO models weights (.pt)
└── envs/                   # Conda environment .yaml info

```

## ROS Dependencies

This package has been tested in **Ubuntu 20.04** and **ROS Noetic** make sure to have standard packages installed:

```bash
sudo apt install ros-noetic-vision-opencv ros-noetic-cv-bridge ros-noetic-image-transport
```

## Conda environment

Due to dependency conflicts between ultralytics and ROS, is strongly recommended to use the provided environment to avoid system issues. Assuming Conda or MiniConda is installed:

```bash
# Create environment from the yaml file (Execute in the root of the package)
conda env create -f envs/tandem_yolo.yaml
```

## How to use?

Due to this conflicts, we will have this setup:

In one terminal, we will execute:

```bash
roscore
```

and in another window of the same terminal we will launch the Preprocessing nodes, in charge of decompressing the images and aplying CLAHE if we desire so:

```bash
roslaunch net_hole_detector preprocessing.launch is_compressed:=true
```

The different arguments can be modified in the .launch file or in the terminal. We must specifie if the images we are expecting are compressed or not, to know if we must decompress them or not.

In a **different terminal**, we activate the conda environment

```bash
conda activate tandem_yolo
```

and we launch the Detection and Inference nodes, in charge of managing YOLO detections, infering scale and distance and publishing the results.

```bash
roslaunch net_hole_detector detection.launch image_topic:=/your/camera/topic is_rectified:=true
```

We must specifie if the images we are expecting are rectified or not, since we will use matrix K or matrix P (both are callibration matrixes) depending on the case. 

The commands specified above will execute the nodes, but they will not activate the inference and hole information gathering. There are two services to enable and disable that, they are the service net_hole_detector/inference_activation_srv and net_hole_detector/inference_deactivation_srv.

## Configuration/Tuning

All numeric parameters are centered in a key yaml file in the *config/UJI_tuning_params.yaml* path. So, if we desire to adjust any of them, they will be more accessible, and modification of Python code will not be needed.

## Camera Calibration

This system needs the calibration matrix of the camera (otherwise it will not work). We have a two steps system: in *config/right_laser.yaml* (we can add another configuration yaml file for another camera if we want to) we store the backup calibration, so when the process starts we have some matrix to default to; additionally we will try to subsribe to the CameraInfo topic (if it is published by the robot) if we are able to subscribe to this topic we will overwrite the previous calibration info.

## Recorded Case

This is meant to be used live, but if we want to try it with already recorded bagfiles, we must do as follows:

```bash
roscore
```

```bash
roslaunch net_hole_detector preprocessing.launch sim:=true is_compressed:=true
```

(again in the conda environment terminal):
```bash
roslaunch net_hole_detector detection.launch sim:=true is_rectified:=true
```

And also (in the 'normal' terminal, in another window)
```bash
rosbag play --clock your/desired/bagfile.bag
```

put the *sim* argument as *True* inthe roslaunches and the *--clock* flag in the rosbag play.

## system.launch

Additionally we have a launch that can trigger the whole process without CLAHE. We will not use this launch anyway.

## Inputs

In order to work, this pipeline will need the *image_topic* (defined in the preprocessing.launch file, can be modified there if needed) and, as we discussed before, the *camera_info_topic* or otherwise its associated yaml file with ALL the calibration info.

## Publications (Outputs)

We will publish different information abot the detections. Some of the info will be visual (different topics to see what are we detecting) and some of the info will be numerical (via ROS messages) so the robot can act accodingly.

The 'informative' topics (Custom messages of type *BoundingBox(Array).msg*, *Detection3D(Array).msg*, *NetStats.msg*) will be:

**/net_hole_detector/bounding_boxes** will publish the info of the boundingboxes detected by YOLO (the score/probability of the detection, x of boxcenter, y of boxcenter, width, height) normalized (the rest of the pipeline will interpret this results).

**/net_hole_detector/detections_3d** will publish the X,Y,Z coordinates (in meters) of the center of the box according to **Camera Coordinates**, a further transform will be needed to have it in World Coordinates.

We will provide **ALL** detections and its coordinates, so the robot can choose what to do with this information.

**/net_hole_detector/net_stat** will publish the scale (meters/pixel) and the calculated median area (in pixels and meters) of the net square grid of every image.

The main 'visual' topics (messages of type *sensor_msgs/Image*) will be:

*Note*: In this context, we are refering to *blobs* as **ALL** the detected net squares of the grid in every image (not only the broken ones). And as *valid* we mean that we apply some filters to remove as much noise or missdetected blobs as possible, we do it by requiring a minimum and maximum area, and a certain aspect ratio height/height to the detected contours. This allows us to calculate the average size in pixels of the net square grid and the meters/pixel scale of every image.

**/hole_detector/image_processed** we will see the original image decompressed and with CLAHE applied (if asked so).

**/net_hole_detector/blobs/binary** an image with the 'valid' detected blobs.

**/net_hole_detector/blobs/overlay** the original image with the same 'valid' blobs painted over it

**/net_hole_detector/debug_image** the original image with the detected bounding boxes painted over it and the Z distance and width written.

**/net_hole_detector/net_mask_yolo_fused** A fusion of the blobs and the yolo: we will see only the valid blobs inside the detected bounding box.

**/net_hole_detector/blob_axis_orientation** the minor and major axis of the detected hole given by two points for each axis and the orientation of the hole major axis, and the length of both axis in meters.
