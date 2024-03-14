## Parameter Guide

Here we introduce the main parameters of CMax-SLAM, and explain how to tune them according to the task requirement.
These parameters are specified (and briefly explained) in launch files; see an [example launch file](https://github.com/tub-rip/cmax_slam/blob/main/launch/ijrr.launch).

### Topics

- `events_topic`: The topic name for the event data. Note that the message type should be `dvs_msgs/EventArray`.
- `camera_info_topic`: The topic name for the camera calibration. If it is not included in your rosbag, you can publish it like we did in the [launch files](https://github.com/tub-rip/cmax_slam/blob/main/launch/ecrot_handheld.launch).

### CMax Parameters

- `contrast_measure`: Objective function to be optimized (0=Variance, 1=Mean Square).
- `event_batch_size`: Number of events in each event mini-batch (just for speeding-up).
- `frontend_blur_sigma, backend_blur_sigma`: Gaussian blur (in pixels) to make the image of warped events smoother, so that the landscape of the objective function becomoes smoother.

### Event Sub-sampling

- `frontend_event_sample_rate`: systematic event sampling rate for the front-end. For example, `frontend_event_sample_rate = 10` means the front-end processes one event out of ten (10% events).
- `backend_event_sample_rate`: Further event sampling rate on the top of the sampled events from the front-end. For example, `frontend_event_sample_rate = 10 && backend_event_sample_rate = 2`, means the back-end processes 5% of all events.

### Front-end Parameters

- `num_events_per_packet`: Number of events used for estimating each angular velocity (after sub-sampling, if enabled). It is the main parameter that needs to be tuned according to different scene textures and camera motion. The values used our experiments are given in the comments of the corresponding launch files.
- `dt_ang_vel`: Frequency of the front-end angular velocity estimates: `f = 1/dt_ang_vel`.
- `show_local_iwe`: Whether to publish the local motion-compensated IWEs. This is useful for tuning the `num_events_per_packet`, by observing the effect of motion compensation. It is published under the topic of `/local_iwe`.


### Back-end Parameters

**Sliding-window Settings**
- `backend_time_window_size`: Size of the sliding time window [s].
- `backend_sliding_window_stride`: Stride of the sliding time window [s].

**Camera Trajectory Settings**
- `spline_degree`: Degree of the camera spline trajectory (1=Linear, 3=Cubic).
- `dt_knots`: Time interval between the knots/control poses of the spline trajectory.

**Panoramic Map Settings**
- `pano_height`: Height of the panoramic map (pano_width = 2*pano_height). The bigger the map, the higher the time and memory consumption.
- `backend_min_ev_rate`: Minimal event rate to launch the backend [ev/s]. When the event rate is too low, we assume the camera stays still, thus do not update the panoramic map.
- `max_update_times`: Maximal updating times for the panoramic map pixels. We check the location of camera FOV on the panorama every 0.05s, the observed pixels (in the FOV) are marked to be observed (updated). We stop updating a panorama pixel, if it is observed (updated) more than a threshold times (to keep the map clean).
- `Y_angle`: Set the starting point of the camera FOV (initial yaw) [deg].
- `gamma`: Gamma correction coefficient, for better visualization.
- `show_pano_map`: Whether to publish global panoramic map. It is published under the topic of `/pano_map`.
- `draw_FOV`: Whether to draw the camera FOV on the panoramic map.
