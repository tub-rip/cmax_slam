## Installation

CMax-SLAM is built upon [ROS](http://www.ros.org/).
The installation instructions of ROS can be found [here](http://wiki.ros.org/noetic/Installation/Ubuntu).
We have tested this software on Ubuntu 20.04 and ROS noetic.

Install [catkin tools](http://catkin-tools.readthedocs.org/en/latest/installing.html), [vcstool](https://github.com/dirk-thomas/vcstool):

    sudo apt install python3-catkin-tools python3-vcstool

Install additional libraries:

    sudo apt install ros-noetic-image-geometry ros-noetic-camera-info-manager ros-noetic-image-view

The [GSL library](http://www.gnu.org/software/gsl/) is a scientific library that can be installed with the command:

    sudo apt install libgsl-dev

Create a new catkin workspace (e.g., `cmax_slam_ws`) if needed:

    mkdir -p ~/cmax_slam_ws/src && cd ~/cmax_slam_ws/
    catkin config --init --mkdirs --extend /opt/ros/noetic --merge-devel --cmake-args -DCMAKE_BUILD_TYPE=Release

Download the source code:

    cd ~/cmax_slam_ws/src/
    git clone https://github.com/tub-rip/cmax_slam

Clone dependencies:

    vcs-import < cmax_slam/dependencies.yaml

Build the ROS package:

    cd ~/cmax_slam_ws
    catkin build cmax_slam
    source devel/setup.bash
