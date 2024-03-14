#include "backend/pose_graph_optimizer.h"
#include "backend/trajectory.h"
#include "backend/global_optim_contrast_gsl.h"

#include <camera_info_manager/camera_info_manager.h>
#include <geometry_msgs/TwistStamped.h>
#include <sensor_msgs/Image.h>

#include <opencv2/highgui.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <glog/logging.h>

#include <sstream>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <sys/stat.h>
#include <unistd.h>

namespace cmax_slam {

PoseGraphOptimizer::PoseGraphOptimizer(ros::NodeHandle* nh): nh_(nh), it_(*nh)
{
    image_pub_ = it_.advertise("pano_map", 1);
}

PoseGraphOptimizer::~PoseGraphOptimizer()
{
    delete traj_;
    delete event_warper_;
    image_pub_.shutdown();
}

void PoseGraphOptimizer::initialize(int camera_width, int camera_height,
                                    const PoseGraphParams &opt,
                                    std::vector<dvs_msgs::Event>* ptr,
                                    std::vector<cv::Point3d>* precomputed_bearing_vectors_ptr)
{
    // Load params
    params = opt;

    // Initialize event_warper
    event_warper_ = new EventWarper(opt.warp_opt, opt.map_opt);
    event_warper_->initialize(camera_width, camera_height,
                              precomputed_bearing_vectors_ptr);

    // Initialize time cursors for sliding window
    win_size_ = ros::Duration(params.sliding_window_opt.time_window_size);
    win_stride_ = ros::Duration(params.sliding_window_opt.sliding_window_stride);
    time_window_initialized_ = false;
    first_time_window_ = true;

    // Initialize the counter for the time window
    count_window_ = 0;

    // Stride of the sliding window, in terms of control poses (the same for linear and cubic spline)
    cp_stride_ = std::round(params.sliding_window_opt.sliding_window_stride/params.traj_opt.dt_knots);
    VLOG(4) << "[PoseGraphOptimizer::initialize] cp_stride_ = " << cp_stride_;

    // Set the event pointer
    events_ptr_ = ptr;

    // Set the minimal number of events (per window stride) to update IG
    min_num_ev_per_win_ = params.sliding_window_opt.time_window_size
            * params.map_opt.backend_min_ev_rate
            /(params.warp_opt.event_sample_rate*ang_vel_estimator_->params.warp_opt.event_sample_rate);

    VLOG(1) << "Back-end initialized";
}


void PoseGraphOptimizer::pushAngVel(const ros::Time& ts,
                                    const Eigen::Vector3d& ang_vel)
{
    if(!time_window_initialized_)
    {
        // Initialize time window with the timestamp of the first ang_vel data
        t_win_beg_ = ts;
        t_win_end_ = t_win_beg_ + win_size_;
        t_ang_vel_beg_ = t_win_beg_;
        t_ang_vel_end_ = t_win_end_;

        // Create an empty trajectory, add control poses later
        TrajectorySettings traj_config;
        traj_config.t_beg = t_win_beg_;
        traj_config.t_end = t_win_end_;
        traj_config.dt_knots = params.traj_opt.dt_knots;
        if (params.traj_opt.spline_degree == 3)
            traj_ = new CubicTrajectory(traj_config);
        else
            traj_ = new LinearTrajectory(traj_config);

        // Initialize for the following angular velocity integration
        ang_vel_prev_ = AngVelEntry(ts, ang_vel);
        time_window_initialized_ = true;

        // Initialize the first pose as a pre-defined position
        const double theta = params.map_opt.Y_angle * M_PI / 180;
        Eigen::Matrix3d R0 = (Eigen::Matrix3d() << std::cos(theta), 0, std::sin(theta),
                              0, 1,               0,
                              -std::sin(theta), 0, std::cos(theta)).finished();
        pose_latest_ = PoseEntry(ts, Sophus::SO3d(R0));
    }

    // Push the estimated front-end angular velocity
    std::unique_lock<std::mutex> ang_vel_lock(mutex_ang_vel);
    frontend_ang_vel_.insert(AngVelEntry(ts, ang_vel));
    ang_vel_lock.unlock();
}

bool PoseGraphOptimizer::isReadyFrontendPoses()
{
    // Lock data that is accessible by both frontend and backend
    std::unique_lock<std::mutex> lock(mutex_ang_vel);

    // Return ture if the latest angular velocity is beyond the current time window
    if (!frontend_ang_vel_.empty() && !ev_subset_ts_map_.empty())
    {
        if (frontend_ang_vel_.rbegin()->first > t_win_end_)
        {
            VLOG(2) << "[Back-end] Ang_vel is ready, "
                       "perform the pose graph optimization";
            return true;
        }
        else { return false; }
    }
    return false;
}

void PoseGraphOptimizer::getEventSubset(const ros::Time& t_beg,
                                        const ros::Time& t_end)
{
    std::unique_lock<std::mutex> ev_lock(mutex_events);
    // Search for the begin/end idx of the event subset through a coarse-to-fine strategy
    // 1. Search at the event packet level
    std::map<ros::Time,int>::iterator ev_beg_iter = ev_subset_ts_map_.upper_bound(t_beg);
    std::map<ros::Time,int>::iterator ev_end_iter = ev_subset_ts_map_.lower_bound(t_end);
    // 2. Search at the by-event level (Do we really need this fine searching?)
    // (1) The begin index is already very accurate, we do not need to change it
    int ev_beg_idx = ev_beg_iter->second;
    // (2) We should search for a more accurate end index
    int ev_end_idx = ev_end_iter->second;
    const ros::Time t_end_mod = t_end-ros::Duration(1e-6); // Leave a small t_epsilon for safety
    while (events_ptr_->at(ev_end_idx).ts > t_end_mod)
    {
        // Move at 100 events' stride (do not need to check one by one)
        ev_end_idx -= 100;
        if (ev_end_idx <= ev_beg_idx)
        {
            ev_end_idx = ev_beg_idx + 1;
            break;
        }
    }

    // Copy events into the event subset
    event_subset_ = std::vector<dvs_msgs::Event>(events_ptr_->begin() + ev_beg_idx,
                                                 events_ptr_->begin() + ev_end_idx);

    // Erase the indexes of the used evenet packt in the look-up table
    ev_subset_ts_map_.erase(ev_subset_ts_map_.begin(), std::next(ev_beg_iter));

    // Delete the processed events (stored in the frontend)
    ang_vel_estimator_->deleteOldEvents(ev_beg_idx);
}

AngVelMap PoseGraphOptimizer::getAngVelSubset(const ros::Time& t_beg,
                                              const ros::Time& t_end)
{
    // Lock data that is accessible by both frontend and backend
    std::unique_lock<std::mutex> lock(mutex_ang_vel);

    // Set timestamp cursors for angular velocities from the front-end
    AngVelMap::iterator iter_beg = frontend_ang_vel_.upper_bound(t_beg);
    AngVelMap::iterator iter_end = frontend_ang_vel_.lower_bound(t_end);

    // Insert selected angular velocities into the subset
    AngVelMap ang_vel_subset;
    ang_vel_subset.clear();
    ang_vel_subset.insert(iter_beg, iter_end);
    VLOG(4) << "[getAngVelSubset] size ang_vel_subset = " << ang_vel_subset.size();

    // Erase used angular velocities
    VLOG(4) << "[getAngVelSubset] num ang_vel before erasing = " << frontend_ang_vel_.size();
    frontend_ang_vel_.erase(frontend_ang_vel_.begin(), iter_end);
    VLOG(4) << "[getAngVelSubset] num ang_vel after erasing = " << frontend_ang_vel_.size();

    return ang_vel_subset;
}

PoseMap PoseGraphOptimizer::integrateAngVel(const PoseEntry& pose_latest,
                                            const AngVelMap& ang_vel_subset)
{
    PoseMap pose_subset;
    PoseEntry pose_curr = pose_latest;
    for (auto const &ang_vel : ang_vel_subset)
    {
        // Skip the data with wrong timestamps (older than the current one)
        if (!(ang_vel.first > ang_vel_prev_.first) && !first_time_window_)
        {
            VLOG(0) << "Wrong ang_vel timestamp, skip!!!";
            continue;
        }
        // Compute rotation increment
        const double dt = (ang_vel.first - pose_curr.first).toSec();
        Eigen::Vector3d drotv = dt * ((ang_vel_prev_.second + ang_vel.second)/2.0);

        // Update current pose (post-multiplication)
        pose_curr.first = ang_vel.first;
        pose_curr.second = pose_curr.second * Sophus::SO3d::exp(drotv);

        // Insert into the frontend pose subset
        pose_subset.insert(pose_curr);

        // Update previous angular velocity, prepare for the next angular velocity
        ang_vel_prev_ = ang_vel;
    }
    VLOG(3) << "[integrateAngVel] Size frontend poses subset = "
            << pose_subset.size();

    return pose_subset;
}

void PoseGraphOptimizer::computeImageOfWarpedEvents(Trajectory* traj,
                                cv::Mat* iwe,
                                std::vector<cv::Mat>* iwe_deriv)
{
    event_warper_->computeImageOfWarpedEvents(traj,
                                              &event_subset_,
                                              iwe,
                                              iwe_deriv);
}

void PoseGraphOptimizer::increUpdateTraj(const std::vector<Eigen::Vector3d>& drotv)
{
    traj_->incrementalUpdate(drotv, idx_cp_opt_beg_);
}

Trajectory* PoseGraphOptimizer::copyAndUpdateTraj(const std::vector<Eigen::Vector3d>& drotv)
{
    return traj_->CopyAndIncrementalUpdate(drotv, idx_cp_traj_beg_, idx_cp_opt_beg_);
}

void PoseGraphOptimizer::processTimeWindow(const AngVelMap& ang_vel_subset)
{
    // Integrate angular velocity into frontend poses
    PoseMap frontend_poses = integrateAngVel(pose_latest_, ang_vel_subset);

    // Generate new control poses with front-end poses
    std::vector<Sophus::SO3d> ctrl_poses_new = traj_->generateCtrlPoses(frontend_poses,
                                                                        t_ang_vel_beg_,
                                                                        t_ang_vel_end_);
    // Add generated control poses to the trajectory
    // 1. For the first time window, add all generated control poses
    // 2. Otherwise, we need to remove the first 1/3 control poses for linear/cubic trajectory respectively
    if (first_time_window_)
    {
        VLOG(1) << "[processTimeWindow] First time window, add all generated control poses";

        // Fix the start of the trajectory
        if (params.traj_opt.spline_degree == 3)
            idx_cp_opt_beg_ = 3; // For cubic spline, fix the first three control poses
        else
            idx_cp_opt_beg_ = 1; // For linear spline, fix the first one control pose

        first_time_window_ = false;
    }
    else
    {
        const int num_cps_erase = (params.traj_opt.spline_degree == 3)? 3 : 1;
        VLOG(4) << "[processTimeWindow] num_cps_erase = " << num_cps_erase;
        VLOG(4) << "[processTimeWindow] num ctrl_poses_new before erase = " << ctrl_poses_new.size();
        ctrl_poses_new.erase(ctrl_poses_new.begin(), ctrl_poses_new.begin()+num_cps_erase);
        VLOG(4) << "[processTimeWindow] num ctrl_poses_new after erase = " << ctrl_poses_new.size();
    }

    VLOG(4) << "[processTimeWindow] size traj before add = " << traj_->size();
    traj_->pushbackCtrlPoses(ctrl_poses_new);
    VLOG(3) << "[processTimeWindow] size traj after add = " << traj_->size();

    // Compute the index of the control pose within the time window
    // Fix the first N control pose (N is the degree of the trajectory)
    idx_cp_traj_beg_ = count_window_ * cp_stride_;
    idx_cp_opt_beg_ = std::max(idx_cp_traj_beg_, idx_cp_opt_beg_);
    num_cp_opt_ = traj_->size() - idx_cp_opt_beg_;

    // Set the number of fixed control poses in the current time window
    event_warper_->setNumFixedCtrlPoses(idx_cp_opt_beg_-idx_cp_traj_beg_);
    // Set the start of the time window after sliding for event warper
    event_warper_->setNextWinBegTime(t_win_beg_ + win_stride_);

    // For updating alpha
    event_warper_->setFirstIter(true);

    // If the number of events in the current time window is too small,
    // the camera woule being stay still, so we do not perform BA
    if (event_subset_.size() > min_num_ev_per_win_)
    {
        // Perform CMax bundle adjustment for this time window
        VLOG(2) << "[Back-end] Perform CMax-BA for this time window";
        setupProblemAndOptimize_gsl();
        // Update the aligned global IWE
        event_warper_->updateIG();

        // Update updating times of each pixel on IG
        setUpdateTimesIG();
    }
    else
    {
        VLOG(1) << "[EventWarper::updateIG] event_subset_.size()=" << event_subset_.size()
                << " < min_num_ev_per_win_=" << min_num_ev_per_win_
                <<  ", camera nearly stays still";
    }

    // Update the latest pose, to prepare for the next time window
    pose_latest_.first = t_win_end_- ros::Duration(1e-6);
    pose_latest_.second = traj_->evaluate(pose_latest_.first);
    ts_event_image_ = t_win_end_;

    // Publish the aligned global IWE
    if (params.data_opt.show_iwe)
        publishEventImage();
}

void PoseGraphOptimizer::setUpdateTimesIG()
{
    // Update the map storing the visit time of each pixel
    const double dt_check_update = 0.05;
    const ros::Time t_check_update_time_end = t_win_beg_ + win_stride_;
    for (ros::Time t_check = t_win_beg_; t_check < t_check_update_time_end;
         t_check += ros::Duration(dt_check_update))
    {
        // Get pose at every 0.05s, update the visit time of the pixel within the FOV
        const Sophus::SO3d rot_check = traj_->evaluate(t_check);
        event_warper_->setUpdateTimesIG(rot_check, 3);
    }
}

void PoseGraphOptimizer::slideWindow()
{
    VLOG(3) << "[Back-end] Sliding window";

    // Slide the time cursors
    t_win_beg_ += win_stride_;
    t_ang_vel_beg_ = t_win_end_;
    t_win_end_ += win_stride_;
    t_ang_vel_end_ = t_win_end_;

    // Clear the event subset
    event_subset_.clear();

    // Time window number +1
    count_window_ += 1;
}

void PoseGraphOptimizer::Run()
{
    while(1)
    {
        // If all frontend angular velocity of current time window are collected
        if (isReadyFrontendPoses())
        {
            // Get the begin and end idx of the event subset
            getEventSubset(t_win_beg_, t_win_end_);

            // Get the angular velocity subset
            AngVelMap ang_vel_subset = getAngVelSubset(t_ang_vel_beg_, t_ang_vel_end_);

            // Perform pose graph optimization via contrast maximization
            processTimeWindow(ang_vel_subset);

            // Slide window, prepare for the next time window
            slideWindow();
        }
    }
}

void PoseGraphOptimizer::publishEventImage()
{
    if (image_pub_.getNumSubscribers() <= 0)
        return;

    // Get the global map
    cv::Mat IG_disp;
    event_warper_->getIG(IG_disp);
    // Gamma correction
    cv::normalize(IG_disp, IG_disp, 0, 1.f, cv::NORM_MINMAX, CV_32FC1);
    cv::pow(IG_disp, params.gamma, IG_disp);
    // Normalize to [0,255]
    cv::normalize(IG_disp, IG_disp, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    IG_disp = 255 - IG_disp; // Invert "color": dark events over white background

    // Draw the camera FOV for visualization
    if (params.draw_FOV)
    {
        // Convert the canvas to the BGR space
        cv::cvtColor(IG_disp, IG_disp, CV_GRAY2BGR);

        // Plot the current FOV
        ros::Time t_plot = t_win_end_ - ros::Duration(1e-6);
        const Sophus::SO3d pose_traj = traj_->evaluate(t_plot);
        event_warper_->drawSensorFOV(IG_disp, pose_traj, cv::Vec3i(255, 0, 0));
    }

    // Publish images (with motion compensation)
    cv_bridge::CvImage cv_iwe_image;
    cv_iwe_image.encoding = params.draw_FOV ? "bgr8": "mono8";
    IG_disp.copyTo(cv_iwe_image.image);

    sensor_msgs::ImagePtr msg = cv_iwe_image.toImageMsg();
    msg->header.stamp = ts_event_image_;
    image_pub_.publish(msg);
}

} // namespace
