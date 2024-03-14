#pragma once

#include "frontend/ang_vel_estimator.h"
#include "backend/event_pano_warper.h"
#include "utils/parameters.h"

#include <ros/ros.h>
#include <opencv2/core.hpp>

#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>

#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/Imu.h>

#include <dvs_msgs/Event.h>
#include <dvs_msgs/EventArray.h>

#include <deque>
#include <vector>
#include <mutex>
#include <fstream>

namespace cmax_slam
{

class AngVelEstimator;

typedef std::pair<ros::Time, Eigen::Vector3d> AngVelEntry;
typedef std::map<ros::Time, Eigen::Vector3d> AngVelMap;
typedef std::pair<int, int> EvIdxPair;

class PoseGraphOptimizer {
public:
    // Constructor
    PoseGraphOptimizer(ros::NodeHandle* nh);

    // Deconstructor
    ~PoseGraphOptimizer();

    // Initialize the backend
    void initialize(int camera_width, int camera_height,
                    const PoseGraphParams& val,
                    std::vector<dvs_msgs::Event>* ptr,
                    std::vector<cv::Point3d>* precomputed_bearing_vectors_ptr);

    // Set the pointer to the frontend
    void setFrontend(AngVelEstimator* ptr) { ang_vel_estimator_ = ptr; }

    // Feed frontend angular velocity and the corresponding events into the backend
    void pushAngVel(const ros::Time& ts, const Eigen::Vector3d& ang_vel);

    // Main function of the backend
    void Run();

    // Get the number of control poses of the trajectory
    int getNumControlPoses() const { return traj_->size(); }

    // Get the number of control poses involved in the optimization
    int getNumOptControlPoses() const { return num_cp_opt_; }

    // Get the number of fixed control poses in the current time window
    int getNumFixedCtrlPoses() const { return idx_cp_opt_beg_-idx_cp_traj_beg_; }

    // Call event_warper to compute IWE
    void computeImageOfWarpedEvents(Trajectory* traj,
                                    cv::Mat* iwe,
                                    std::vector<cv::Mat>* iwe_deriv);

    // Update trajectory using incremental rotation vector
    void increUpdateTraj(const std::vector<Eigen::Vector3d>& drotv);

    // Return a copy of the trajectory that is updated by given incremental rotation
    Trajectory* copyAndUpdateTraj(const std::vector<Eigen::Vector3d>& drotv);

    // Visualize the refined panoramic IWEs
    void publishEventImage();

    // Update the updating times of each pixel in IG
    void setUpdateTimesIG();

    // Parameters
    PoseGraphParams params;

    // Mutex for multi-thread
    std::mutex mutex_events, mutex_ang_vel;

    // For debugging & visualization
    cv::Mat evoluting_sharp_iwe_;

    // Event subset look-up table
    std::map<ros::Time,int> ev_subset_ts_map_;

private:
    // Node handle used to subscribe to ROS topics
    ros::NodeHandle* nh_;

    // Publishers
    image_transport::ImageTransport it_;
    image_transport::Publisher image_pub_;

    // Timestamp for the display IWE
    ros::Time ts_event_image_;

    // Sliding window
    void slideWindow();
    // Begin/end time of current window
    ros::Time t_win_beg_, t_win_end_;
    ros::Time t_ang_vel_beg_, t_ang_vel_end_;
    ros::Duration win_size_, win_stride_;
    // Index of the first control pose within the current timw window
    int idx_cp_traj_beg_;
    // Index of the first control pose involved in the optimization
    int idx_cp_opt_beg_;
    // Stride of the control pose index
    int cp_stride_;
    // Number of control poses involved in the optimization
    int num_cp_opt_;
    // Flag for the time window initialization
    bool time_window_initialized_;
    // Flag to show if we are optimizing the first time window
    bool first_time_window_;
    // Counter of the time window
    int count_window_;

    // Check if all frontend poses are collected for the current time window
    bool isReadyFrontendPoses();

    // Events
    std::vector<dvs_msgs::Event> event_subset_;
    void getEventSubset(const ros::Time& t_beg, const ros::Time& t_end);

    // Front-end angular velocity & integration
    PoseEntry pose_latest_;
    AngVelMap frontend_ang_vel_;
    AngVelEntry ang_vel_prev_;
    AngVelMap getAngVelSubset(const ros::Time& t_beg, const ros::Time& t_end);
    // Integrate frontend angular velocities into absolute poses
    PoseMap integrateAngVel(const PoseEntry& pose_init, const AngVelMap& ang_vel_subset);

    // Perform CMax-based BA
    void processTimeWindow(const AngVelMap& ang_vel_subset);
    void setupProblemAndOptimize_gsl();

    // Pointer to the total event vector (stored in the frontend)
    std::vector<dvs_msgs::Event>* events_ptr_;

    // Event warper: compute IWE and deriv
    EventWarper* event_warper_;

    // Trajectory
    Trajectory* traj_;

    // Pointer to the angular velocity estimator (front-end)
    AngVelEstimator* ang_vel_estimator_;

    // Stop updating IG at some point
    size_t min_num_ev_per_win_;
};

} // namespace
