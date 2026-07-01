# tandem

ROS Noetic package containing the control node used for the final vehicle approach during the net inspection task.

The package launches the `vertical_inspector` controller for the Girona500 AUV. The controller follows an inspection pattern and, when a hole is detected, uses the perception output from `net_hole_detector` to center the vehicle and approach the defect.

## Package structure

```text
tandem/
├── cfg/       Dynamic reconfigure parameters
├── launch/    Controller launch file
└── src/       Python control node
```

## Main files

| File | Purpose |
|---|---|
| `src/ri.py` | Main controller node. Publishes body velocity commands for the Girona500. |
| `cfg/VerticalInspector.cfg` | Runtime tuning parameters for the Y and Z control loops. |
| `launch/girona500.launch` | Launch file for the `vertical_inspector` node. |

## Control inputs

The controller uses the following perception topics:

```text
/net_hole_detector/stereo_left/bounding_boxes
/net_hole_detector/stereo_detections_3d
```

The bounding boxes are used for visual centering.

The 3D detections are used to control the final distance to the detected hole.

## Control output

The controller publishes body velocity commands to:

```text
/girona500/controller/body_velocity_req
```

These commands are used to move the vehicle in the body frame during the inspection and final approach.

## Launch

```bash
roslaunch tandem girona500.launch
```

The launch file defines the inspection limits, velocity limits, visual servoing gains, detection timeouts and safety distance to the net-defect.

## Dynamic reconfigure

Some control parameters can be adjusted at runtime using:

```bash
rosrun rqt_reconfigure rqt_reconfigure
```

The available parameters are defined in:

```text
cfg/VerticalInspector.cfg
```

This is mainly used to tune the proportional, integral and derivative gains, the velocity limits and the filtering parameters for the Y and Z axes.