This repository contains the ROS nodes for the `TANDEM` project, designed for Girona500 AUV inspection tasks. 

## ⚠️ Required Environment

This software has been developed and tested under a **very specific** set of dependencies. It is **not guaranteed** to work with other versions (e.g., newer Stonefish versions).

For stable operation, your system **MUST** match the following configuration:

* **Operating System:** Ubuntu 20.04 
* **ROS Distribution:** ROS Noetic 

### Stonefish Simulator Configuration

This project is validated against a specific build of the Stonefish simulator. Using versions other than those listed below will likely result in compilation errors or simulation-time failures.

* **[Stonefish ](https://github.com/srv/stonefish):** Core Library - Version `v1.3`
* **[`stonefish_ros` ](https://github.com/srv/stonefish_ros):** ROS Wrapper - Version`v1.3`
* **[`cola2_stonefish`](https://github.com/srv/cola2_stonefish):** COLA2 Integration - Branch `master`

> **Note:** The `cola2_stonefish` repository's `master` branch is configured to work specifically with the `v1.3` releases of `stonefish` and `stonefish_ros`.

## 📦 Installation

1.  Ensure the **Required Environment** (see above) is correctly set up.
2.  Clone this repository into your Catkin workspace's `src` directory:
    ```bash
    cd ~/catkin_ws/src
    git clone https://github.com/srv/tandem.git
    ```
4.  Build the workspace:
    ```bash
    cd ~/catkin_ws
    catkin build
    ```
5.  Source the workspace:
    ```bash
    source ~/catkin_ws/devel/setup.bash
    ```
