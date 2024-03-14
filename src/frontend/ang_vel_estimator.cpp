#include "frontend/ang_vel_estimator.h"
#include "utils/image_geom_util.h"

#include <camera_info_manager/camera_info_manager.h>
#include <geometry_msgs/TwistStamped.h>
#include <sensor_msgs/Image.h>
#include <opencv2/highgui.hpp>
#include <glog/logging.h>

#include <sstream>
#include <string>
#include <vector>
#include <sys/stat.h>

namespace cmax_slam {

static const double rad2degFactor = 180.0 * M_1_PI;

AngVelEstimator::AngVelEstimator(ros::NodeHandle* nh): nh_(nh), it_(*nh)
{
    // Set publishers
    img_pub_ = it_.advertise("local_iwe", 1);
    ang_vel_pub_ = nh_->advertise<geometry_msgs::TwistStamped>("/dvs/angular_velocity", 1);

    // Initial value of motion parameters velocity
    ang_vel_ = cv::Point3d(0.,0.,0.);
}

AngVelEstimator::~AngVelEstimator()
{
    img_pub_.shutdown();
    ang_vel_pub_.shutdown();
}

void AngVelEstimator::initialize(image_geometry::PinholeCameraModel* cam,
                                 const AngVelEstParams& val,
                                 const std::vector<cv::Point3d>& precomputed_bearing_vectors)
{
    // Load camera information
    cam_width_ = cam->fullResolution().width;
    cam_height_ = cam->fullResolution().height;
    camera_matrix_ = cam->fullIntrinsicMatrix();

    // Load params
    params = val;
    // Get the pre-computed bearing vector
    precomputed_bearing_vectors_ = precomputed_bearing_vectors;

    // Initialize the event maintainance (containers and lists)
    std::unique_lock<std::mutex> ev_lock(pose_graph_optimizer_->mutex_events);
    events_.clear();
    num_event_total_ = 0;
    event_subsets_info_.clear();
    ev_lock.unlock();

    // Pre-allocate memory for the event subset (fixed size)
    event_subset_.reserve(val.num_events_per_packet);

    // Set frequency of the output angular velocity
    dt_av_ = ros::Duration(val.dt_ang_vel);

    // Setttings for sliding window
    sliding_window_initialized_ = false;
    num_ev_half_packet_ = val.num_events_per_packet/2;
    VLOG(1) << "Front-end initialized";
}

void AngVelEstimator::pushEvent(const dvs_msgs::Event& event)
{
    if (!sliding_window_initialized_)
    {
        VLOG(1) << " [Front-end] The first event arrived at t = " << std::setprecision(19) << event.ts.toSec();
        // Initialize sliding window (time cursors)
        time_packet_ = event.ts + dt_av_*0.5;
        time_get_subset_ = time_packet_;
        sliding_window_initialized_ = true;
    }

    // Add this new event into the total event vector
    std::unique_lock<std::mutex> ev_lock(pose_graph_optimizer_->mutex_events);
    events_.emplace_back(event);
    num_event_total_ += 1;

    // Get event subset info
    if (event.ts > time_get_subset_)
    {
        // Compute the indexes of the head and tail of the event subset
        const int idx_subset_beg = std::max(num_event_total_-num_ev_half_packet_, 0);
        const int idx_subset_end = num_event_total_+num_ev_half_packet_;

        // Push back into the event subset information list (front-end) and look-up table (back-end)
        event_subsets_info_.emplace_back(std::pair<int, int>(idx_subset_beg, idx_subset_end));
        pose_graph_optimizer_->ev_subset_ts_map_.insert(std::pair<ros::Time, int>(event.ts, num_event_total_-1));

        // Update time_packet_, to prepare for the next packet
        time_get_subset_ += dt_av_;
    }

    // Once the whole event packet is received, perform CMax angular velocity estimation
    if (!event_subsets_info_.empty() && num_event_total_ > event_subsets_info_.front().second)
    {
        // Get event subset for the current time window
        getEventSubset();

        // Unlock the mutex after event-related operations
        ev_lock.unlock();

        // If the time span of this event packet is to long, assume the ang_vel is 0
        const double timespan_packet = (event_subset_.back().ts - event_subset_.front().ts).toSec();
        if (timespan_packet > 10*params.dt_ang_vel)
        {
            VLOG(2) << "Time span of the event packet is too long, assume the angular velocity to be 0";
            ang_vel_ = cv::Point3d(0.0, 0.0, 0.0);
        }
        else
        {
            // Process the current time window
            processEventPacket();
        }

        // Feed the estimated angular velocity to the back-end
        Eigen::Vector3d ang_vel(ang_vel_.x, ang_vel_.y, ang_vel_.z);
        pose_graph_optimizer_->pushAngVel(time_packet_, ang_vel);

        // Save / Publish image
        if (params.data_opt.show_iwe)
            publishEventImage();

        // Publish estimated motion parameters
        publishAngularVelocity();

        // Slide time window for the next iteration
        slideWindow();
    }
}

void AngVelEstimator::getEventSubset()
{
    // Get event subset
    ev_beg_idx_ = event_subsets_info_.front().first;
    ev_end_idx_ = event_subsets_info_.front().second;
    event_subset_ = std::vector<dvs_msgs::Event>(events_.begin() + ev_beg_idx_,
                                                 events_.begin() + ev_end_idx_);

    // Erase the used event subset
    event_subsets_info_.pop_front();
}

void AngVelEstimator::deleteOldEvents(const int idx_backend)
{
    // Events before idx_del will be deleted
    const int num_ev_deleted = std::min(idx_backend, ev_beg_idx_);
    events_.erase(events_.begin(), events_.begin()+num_ev_deleted);

    // Update the event indexes maintained for sliding window
    num_event_total_ -= num_ev_deleted;
    ev_beg_idx_ -= num_ev_deleted;
    ev_end_idx_ -= num_ev_deleted;

    // Update the event subset information list
    for (auto it = event_subsets_info_.begin(); it != event_subsets_info_.end(); it++)
    {
        it->first -= num_ev_deleted;
        it->second -= num_ev_deleted;
    }

    // Update the event index look-up table for the back-end
    for (auto it = pose_graph_optimizer_->ev_subset_ts_map_.begin();
         it != pose_graph_optimizer_->ev_subset_ts_map_.end(); it++)
    {
        it->second -= num_ev_deleted;
    }
}

void AngVelEstimator::slideWindow()
{
    VLOG(3) << "[Front-end] sliding window";
    event_subset_.clear(); // Clear will not change the capacity of this vector

    // Slide the timestamp of the angular velocity
    time_packet_ += dt_av_;
}

void AngVelEstimator::processEventPacket()
{
    VLOG(3) << "[Front-end] process time window";
    double final_cost = setupProblemAndOptimize_gsl(ang_vel_); // GNU-GSL;
    VLOG(3) << "[AngVelEstimator::processEventPacket] final cost = " << final_cost;
}

void AngVelEstimator::publishAngularVelocity()
{
    // Publish motion parameters
    geometry_msgs::TwistStamped twist_msg;
    twist_msg.twist.angular.x = ang_vel_.x * rad2degFactor;
    twist_msg.twist.angular.y = ang_vel_.y * rad2degFactor;
    twist_msg.twist.angular.z = ang_vel_.z * rad2degFactor;
    twist_msg.header.stamp = time_packet_;
    twist_msg.header.frame_id = "body";
    ang_vel_pub_.publish(twist_msg);
}

void AngVelEstimator::publishEventImage()
{
    if (img_pub_.getNumSubscribers() <= 0)
        return;

    static cv::Mat img_original, img_warped, img_stacked;
    // Options to visualize the image of raw events and
    // the image of warped events (IWE) without blur
    OptionsWarp opts_warp_display = params.warp_opt;
    opts_warp_display.blur_sigma = 0.;

    // Compute image of raw events
    // (IWE assuming zero motion parameters, i.e., without motion compensation)
    computeImageOfWarpedEvents(cv::Point3d(0.), &img_original);

    // Compute IWE with estimated motion parameters (motion-compensated image)
    computeImageOfWarpedEvents(ang_vel_, &img_warped);

    cv::hconcat(img_original, img_warped, img_stacked);
    // Join both images side-by-side so that they are displayed with the same range
    // Scale the image to full range [0,255]
    cv::normalize(img_stacked, img_stacked, 0.f, 255.f, cv::NORM_MINMAX, CV_32FC1);
    // Invert "color": dark events over white background for better visualization
    img_stacked = 255.f - img_stacked;

    // Publish images (without and with motion compensation)
    cv_event_image_.encoding = "mono8";
    cv_event_image_.header.stamp = time_packet_;
    img_stacked.convertTo(cv_event_image_.image,CV_8UC1);
    img_pub_.publish(cv_event_image_.toImageMsg());
}

} // namespace
