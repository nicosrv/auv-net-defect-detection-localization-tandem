# tandem

ROS Noetic package containing the control node used for the final vehicle approach during the net inspection task.

The package launches the `vertical_inspector` controller for the Girona500 AUV. The controller follows an inspection pattern and, when a hole is detected, uses the perception output from `net_hole_detector` to center the vehicle and approach the defect.

## Package structure

```text
tandem/
├── cfg/       Parameters
├── launch/    Controller launch file
└── src/       Python control node
```

## Main files

| File | Purpose |
|---|---|
| `src/ri.py` | Main controller node. Publishes body velocity commands for the Girona500. |
| `launch/girona500.launch` | Launch file for the `vertical_inspector` node. |

## Control inputs

The controller uses the following perception topics:

```text
/net_hole_detector/stereo_left/bounding_boxes
/net_hole_detector/stereo_detections_3d
```

The bounding boxes are used for visual centering.

The 3D detections are used to control the stopping final distance to the detected hole.

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

The launch file defines the inspection movement, velocity limits, visual servoing gains, detection timeouts and safety distance to the net-defect.
