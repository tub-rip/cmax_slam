## Live demo with a DAVIS camera

### Start the DAVIS

First, calibrate your camera, and place the calibration file in the appropriate folder, `~/.ros/camera_info/`, for example [DAVIS-00000254.yaml](https://github.com/tub-rip/cmax_slam/blob/main/docs/DAVIS-00000254.yaml) (for DAVIS with serial number S/N 0254).

Launch the ROS core:

    roscore

In a new terminal, use [davis_ros_driver](https://github.com/uzh-rpg/rpg_dvs_ros/) to start your DAVIS camera driver:

    cd ~/cmax_slam_ws
    catkin build davis_ros_driver
    source devel/setup.bash
    rosrun davis_ros_driver davis_ros_driver

This should read the calibration file corresponding to the serial number of your DAVIS camera and publish its content in topic `/dvs/camera_info`. 
It should also publish event data in the topic `/dvs/events`.

### Run CMax-SLAM

In a new terminal, launch the node of CMax-SLAM:

    cd ~/cmax_slam_ws
    source devel/setup.bash
    roslaunch cmax_slam live_davis.launch

See the generated panoramic edge map in `rqt_image_view`, which is published under the topic of `/pano_map`.
See the [live_davis](https://github.com/tub-rip/cmax_slam/blob/main/launch/live_davis.launch) launch file. 
