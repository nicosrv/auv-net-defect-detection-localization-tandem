# net_hole_detector

ROS Noetic package for detecting holes in underwater net images and estimating their 3D position using stereo vision.

The package combines YOLO-based 2D detection, FastSAM mask generation, masked stereo processing and disparity-based depth estimation. The final output is a custom `Detection3DArray` message containing the estimated 3D position of the detected hole.

## Package structure

```text
net_hole_detector/
├── envs/        Conda environment files
├── launch/      ROS launch files
├── msg/         Custom ROS messages
├── scripts/     Python ROS nodes
├── src/         Internal Python modules
└── weights/     Trained YOLO weights
```

## Main nodes

| Script | Purpose |
|---|---|
| `bbox_detector.py` | Runs YOLO on rectified stereo images and publishes 2D bounding boxes. |
| `corrector_camera_info.py` | Republishes the right camera `CameraInfo` with the corrected stereo projection term. |
| `hole_segmenter_fastsam.py` | Generates binary masks from YOLO bounding boxes using FastSAM. |
| `masked_stereo_image_builder.py` | Applies the FastSAM masks to the rectified stereo pair. |
| `stereo_distance_estimator.py` | Estimates the 3D position of the detected hole from masked stereo disparity. |

## Launch files

| Launch file | Description |
|---|---|
| `stereo_processing.launch` | Runs the stereo rectification and CameraInfo correction stage. |
| `stereo_bboxes.launch` | Starts the YOLO detection nodes for the left and right stereo images. |
| `fastsam_stereo_dual.launch` | Runs FastSAM segmentation for both stereo images. |
| `masked_stereo_images.launch` | Builds the masked stereo image pair. |
| `masked_stereo_processing.launch` | Runs `stereo_image_proc` on the masked stereo pair. |
| `masked_stereo_z.launch` | Estimates the 3D position of the detected hole. |
| `yolo_inference.launch` | Generic YOLO inference launch used by the stereo detection launch. |

## Main input topics

The perception pipeline uses the rectified stereo images and the corresponding camera information from the Girona500 stereo camera:

```text
/girona500/xiroi/stereo_ch3/left_optical/image_rect_color
/girona500/xiroi/stereo_ch3/right_optical/image_rect_color
/girona500/xiroi/stereo_ch3/left_optical/camera_info
/girona500/xiroi/stereo_ch3/right_optical/camera_info
```

## Main output topics

```text
/net_hole_detector/stereo_left/bounding_boxes
/net_hole_detector/stereo_right/bounding_boxes
/net_hole_detector/stereo_left/hole_mask
/net_hole_detector/stereo_right/hole_mask
/masked_stereo/left/image_raw
/masked_stereo/right/image_raw
/masked_stereo/disparity
/net_hole_detector/stereo_detections_3d
```

The final topic used by the control package is:

```text
/net_hole_detector/stereo_detections_3d
```

## Typical execution

The complete perception pipeline can be launched step by step:

```bash
roslaunch net_hole_detector stereo_processing.launch
roslaunch net_hole_detector stereo_bboxes.launch
roslaunch net_hole_detector fastsam_stereo_dual.launch
roslaunch net_hole_detector masked_stereo_images.launch
roslaunch net_hole_detector masked_stereo_processing.launch
roslaunch net_hole_detector masked_stereo_z.launch
```

The YOLO and FastSAM nodes require the Python environment containing the corresponding deep learning dependencies.

## Custom messages

The package defines the following custom ROS messages:

```text
BoundingBox.msg
BoundingBoxArray.msg
Detection3D.msg
Detection3DArray.msg
```

`BoundingBoxArray` is used to publish the 2D detections obtained from YOLO.

`Detection3DArray` is used to publish the final 3D position estimated from the masked stereo disparity.