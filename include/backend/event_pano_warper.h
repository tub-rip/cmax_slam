#pragma once

#include "backend/trajectory.h"
#include "backend/equirectangular_camera.h"
#include "utils/image_geom_util.h"
#include "utils/image_utils.h"
#include "utils/parameters.h"

#include <vector>

#include <ros/ros.h>
#include <dvs_msgs/Event.h>
#include <dvs_msgs/EventArray.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/CameraInfo.h>
#include <image_geometry/pinhole_camera_model.h>

namespace cmax_slam {

class EventWarper{
public:
    // Constructor
    EventWarper(const OptionsWarp& warp_opt, const OptionPanoMap& map_opt):
        warp_opt_(warp_opt), map_opt_(map_opt) {}
    ~EventWarper() {}

    // initialize camera and pano info
    void initialize(int camera_width, int camera_height,
                    std::vector<cv::Point3d>* precomputed_bearing_vectors_ptr);

    // For computing alpha
    void setFirstIter(bool val) { first_iter_ = val; }

    // Access to the global optimal sharp IWE
    void getIG(cv::Mat& map) const { IG_.copyTo(map); }

    // Access to IG
    void getIGp(cv::Mat& map) const { IGp_.copyTo(map); }

    // Update global optimal sharp IWE
    void updateIG();

    // Draw the FOV of the event camera, for plotting
    void warpEventToMap(const cv::Point2i& pt_in, const Sophus::SO3d& rot, cv::Point2d& pt_out);
    void drawSensorFOV(cv::Mat& canvas,
                       const Sophus::SO3d& rot,
                       const cv::Vec3i& color);

    // Set the start time of the next trajectory
    void setNextWinBegTime(const ros::Time& val) { t_next_win_beg_ = val; }

    // Set the number of fixed control poses in the current time window
    void setNumFixedCtrlPoses(const int val) { num_cps_fixed_ = val; }

    // Reset global optimal sharp IWE to zero
    void resetIG() { IG_.setTo(0); }

    // Compute IWE
    void computeImageOfWarpedEvents(Trajectory* traj,
                                    std::vector<dvs_msgs::Event>* event_subset,
                                    cv::Mat* iwe,
                                    std::vector<cv::Mat>* iwe_deriv);

    // Process events by batch to speed up, all events in the same batch share a common pose
    void warpAndAccumulateEvents(Trajectory* traj,
                                 std::vector<dvs_msgs::Event>::iterator event_begin,
                                 std::vector<dvs_msgs::Event>::iterator event_end,
                                 std::vector<cv::Mat>* iwe_deriv);

    // Adaptive alpha for global alignment: I = IL + alpha * IG
    void updateAlpha();

    // Update (normalized) IG and alpha
    void updateIGp();

    // Update the updating times of IG
    void setUpdateTimesIG(const Sophus::SO3d& rot,
                         const int radius);

private:
    // IL = IL_old + IL_new
    cv::Mat IL_old_; // Image of events that are going to be out of the optimization window
    cv::Mat IL_new_; // Image of events that are still in the optimization window in the next iteration

    // I = IL + alpha * IGp
    cv::Mat IL_, IG_, IGp_;
    double alpha_;

    // Stop updating IG at some point
    cv::Mat IG_update_times_map_;

    // Camera information (size, intrinsics, lens distortion)
    int sensor_width_, sensor_height_;
    dvs::EquirectangularCamera pano_cam_;

    // Map, panorama
    int pano_width_, pano_height_;
    cv::Size pano_size_;

    // Sliding window
    ros::Time t_next_win_beg_; // To distinguish events that should be on the old/new map

    // Number of fixed control poses in the current time window
    int num_cps_fixed_;

    // Precomputed bearing vector
    std::vector<cv::Point3d>* precomputed_bearing_vectors_;

    // Parameters
    OptionsWarp warp_opt_;
    OptionPanoMap map_opt_;

    // For updating alpha
    bool first_iter_;
};

}
