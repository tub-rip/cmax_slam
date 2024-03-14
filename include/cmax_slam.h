#pragma once

#include <thread>
#include <fstream>

#include "frontend/ang_vel_estimator.h"
#include "backend/pose_graph_optimizer.h"

#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <dvs_msgs/Event.h>
#include <dvs_msgs/EventArray.h>

namespace cmax_slam {

class CMaxSLAM
{
public:
    // Constructor
    CMaxSLAM(ros::NodeHandle& nh);
    // Deconstructor
    ~CMaxSLAM();

    // Precomputed bearing vectors for each image pixel
    std::vector<cv::Point3d> precomputed_bearing_vectors; // Share with the back-end
    image_geometry::PinholeCameraModel cam;

private:
    // Node handle used to subscribe to ROS topics
    ros::NodeHandle nh_;
    // Private node handle for reading parameters
    ros::NodeHandle pnh_;

    // Subscribers and callbacks
    ros::Subscriber event_sub_, camera_info_sub_;
    void eventsCallback(const dvs_msgs::EventArray::ConstPtr& msg);
    void cameraInfoCallback(const sensor_msgs::CameraInfo::ConstPtr& camera_info);
    bool got_camera_info_;

    // Precompute bearing vectors and share with the front-end and the back-end
    void precomputeBearingVectors();

    // Parameters
    AngVelEstParams front_end_params_;
    PoseGraphParams back_end_params_;

    // Multi-thread
    AngVelEstimator* ang_vel_estimator_;       // Front-end
    PoseGraphOptimizer* pose_graph_optimizer_; // Back-end

    std::thread* pose_graph_optim_; // Thread for the back-end
};
}
