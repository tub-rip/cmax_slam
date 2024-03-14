#include "utils/image_geom_util.h"
#include "frontend/ang_vel_estimator.h"

#include <ros/time.h>
#include <glog/logging.h>
#include <opencv2/highgui.hpp>

namespace cmax_slam {

void AngVelEstimator::computeImageOfWarpedEvents(
        const cv::Point3d& ang_vel,
        cv::Mat* image_warped,
        cv::Mat* image_warped_deriv)
{
    // Create image of unwarped events
    *image_warped = cv::Mat::zeros(cam_height_, cam_width_, CV_32FC1);
    if (image_warped_deriv != nullptr)
        *image_warped_deriv = cv::Mat::zeros(cam_height_, cam_width_, CV_32FC3);

    // loop through all events
    // Get event_batch using the indeces of the head and tail of the current event packet
    for (int idx_ev_batch_beg = 0; idx_ev_batch_beg < event_subset_.size();
         idx_ev_batch_beg += params.warp_opt.event_batch_size)
    {
        auto idx_ev_batch_end = std::min(idx_ev_batch_beg + params.warp_opt.event_batch_size, int(event_subset_.size()));
        warpAndAccumulateEvents(ang_vel, idx_ev_batch_beg, idx_ev_batch_end,
                                time_packet_, image_warped, image_warped_deriv);
    }

    // Smooth the image (to spread the votes)
    // For speed, smoothing may not be used, since bilinear voting has been implemented.
    if (params.warp_opt.blur_sigma > 0)
    {
        cv::GaussianBlur(*image_warped, *image_warped, cv::Size(0,0), params.warp_opt.blur_sigma);
        if (image_warped_deriv != nullptr)
            cv::GaussianBlur(*image_warped_deriv, *image_warped_deriv,
                             cv::Size(0,0), params.warp_opt.blur_sigma);
    }
}

void AngVelEstimator::computeImageOfWarpedEvents(
        const cv::Point3d& ang_vel,
        cv::Mat* image_warped)
{
    // Create image of unwarped events
    *image_warped = cv::Mat::zeros(cam_height_, cam_width_, CV_32FC1);

    // loop through all events
    // Get event_batch using the indeces of the head and tail of the current event packet
    for (int idx_ev_batch_beg = 0; idx_ev_batch_beg < event_subset_.size();
         idx_ev_batch_beg += params.warp_opt.event_batch_size)
    {
        auto idx_ev_batch_end = std::min(idx_ev_batch_beg + params.warp_opt.event_batch_size, int(event_subset_.size()));
        warpAndAccumulateEvents(ang_vel, idx_ev_batch_beg, idx_ev_batch_end,
                                time_packet_, image_warped, nullptr);
    }
}

void AngVelEstimator::warpAndAccumulateEvents(
        const cv::Point3d& ang_vel,
        const int& idx_event_batch_begin,
        const int& idx_event_batch_end,
        const ros::Time time_ref,
        cv::Mat* image_warped,
        cv::Mat* image_warped_deriv)
{
    // All events in this batch share a common pose (for speed-up)
    ros::Time time_first = event_subset_.at(idx_event_batch_begin).ts;
    ros::Time time_last = event_subset_.at(idx_event_batch_end-1).ts;
    ros::Duration time_dt = time_last - time_first;
    //CHECK_GT(time_dt.toSec(), 0.) << "Events must span a non-zero time interval";
    CHECK_GE(time_dt.toSec(), 0.) << "Events must span a non-negative time interval";
    ros::Time time_batch = time_first + time_dt * 0.5;

    const double dt = time_batch.toSec() - time_ref.toSec(); // faster than Duration object
    const cv::Point3d delta_rot = ang_vel * dt;

    static cv::Point2d calibrated_pt;
    static cv::Matx23d jacobian_calibrated_pt_wrt_ang_vel;
    static cv::Matx23d* jacobian_calibrated_pt_wrt_ang_vel_ptr;
    jacobian_calibrated_pt_wrt_ang_vel_ptr =
            (image_warped_deriv == nullptr) ? nullptr : &jacobian_calibrated_pt_wrt_ang_vel;

    static cv::Matx22d jacobian_pix_pt_wrt_calib_pt;
    static cv::Matx22d* jacobian_pix_pt_wrt_calib_pt_ptr;
    jacobian_pix_pt_wrt_calib_pt_ptr =
            (image_warped_deriv == nullptr) ? nullptr : &jacobian_pix_pt_wrt_calib_pt;

    static cv::Matx23d jacobian_warped_pt_wrt_ang_vel;
    static cv::Matx23d* jacobian_warped_pt_wrt_ang_vel_ptr;
    jacobian_warped_pt_wrt_ang_vel_ptr =
            (image_warped_deriv == nullptr) ? nullptr : &jacobian_warped_pt_wrt_ang_vel;

    for (int event_idx = idx_event_batch_begin; event_idx < idx_event_batch_end; event_idx++)
    {
        // Get the event to process
        auto event = event_subset_.at(event_idx);

        // Approximation: use only the first two terms of the series expansion of the rotation matrix
        cv::Point3d point_3D = precomputed_bearing_vectors_.at(event.y*cam_width_+event.x);
        cv::Point3d point_3D_rotated = point_3D + delta_rot.cross(point_3D);

        static cv::Matx33d jacobian_warped_event_wrt_ang_vel;
        static cv::Matx33d* jacobian_warped_event_wrt_ang_vel_ptr;
        if (jacobian_calibrated_pt_wrt_ang_vel_ptr == nullptr)
            jacobian_warped_event_wrt_ang_vel_ptr = nullptr;
        else
        {
            jacobian_warped_event_wrt_ang_vel_ptr = &jacobian_warped_event_wrt_ang_vel;
            cross2Matrix((-dt)*point_3D, jacobian_warped_event_wrt_ang_vel_ptr);
        }

        // calibrated coordinates
        static cv::Matx23d jacobian_calibrated_pt_wrt_warped_event;
        static cv::Matx23d* jacobian_calibrated_pt_wrt_warped_event_ptr;
        if (jacobian_calibrated_pt_wrt_ang_vel_ptr == nullptr)
            jacobian_calibrated_pt_wrt_warped_event_ptr = nullptr;
        else
            jacobian_calibrated_pt_wrt_warped_event_ptr = &jacobian_calibrated_pt_wrt_warped_event;

        canonicalProjection(point_3D_rotated, &calibrated_pt,
                            jacobian_calibrated_pt_wrt_warped_event_ptr);

        // compute jacobian
        if (jacobian_calibrated_pt_wrt_ang_vel_ptr != nullptr)
            jacobian_calibrated_pt_wrt_ang_vel = jacobian_calibrated_pt_wrt_warped_event * jacobian_warped_event_wrt_ang_vel;

        // Apply intrinsic parameters
        cv::Point2d ev_warped_pt;
        applyIntrinsics(calibrated_pt, camera_matrix_, &ev_warped_pt,
                        jacobian_pix_pt_wrt_calib_pt_ptr);

        // Output Jacobian
        if (jacobian_warped_pt_wrt_ang_vel_ptr != nullptr)
            jacobian_warped_pt_wrt_ang_vel = jacobian_pix_pt_wrt_calib_pt * jacobian_calibrated_pt_wrt_ang_vel;

        // Accumulate warped events, using BILINEAR voting (polarity)
        // Bilinear voting is better than regular voting to get good derivative images
        int xx = ev_warped_pt.x, yy = ev_warped_pt.y;

        // if warped point is within the image, accumulate polarity
        if (1 <= xx && xx < cam_width_-2 && 1 <= yy && yy < cam_height_-2)
        {
            float dx = ev_warped_pt.x - xx,
                    dy = ev_warped_pt.y - yy;

            // Accumulate image used to compute contrast
            image_warped->at<float>(yy  ,xx  ) += (1.f-dx)*(1.f-dy);
            image_warped->at<float>(yy  ,xx+1) += dx*(1.f-dy);
            image_warped->at<float>(yy+1,xx  ) += (1.f-dx)*dy;
            image_warped->at<float>(yy+1,xx+1) += dx*dy;

            if (image_warped_deriv != nullptr)
            {
                CHECK_NOTNULL(jacobian_warped_pt_wrt_ang_vel_ptr);

                cv::Matx13d r0m = jacobian_warped_pt_wrt_ang_vel_ptr->row(0);
                cv::Matx13d r1m = jacobian_warped_pt_wrt_ang_vel_ptr->row(1);
                cv::Point3f r0 = cv::Point3f(r0m(0),r0m(1),r0m(2));
                cv::Point3f r1 = cv::Point3f(r1m(0),r1m(1),r1m(2));

                // Using Kronecker delta formulation and only differentiating weigths of bilinear voting
                image_warped_deriv->at<cv::Point3f>(yy  ,xx  ) += r0*(-(1.f-dy)) + r1*(-(1.f-dx));
                image_warped_deriv->at<cv::Point3f>(yy  ,xx+1) += r0*(1.f-dy)    + r1*(-dx);
                image_warped_deriv->at<cv::Point3f>(yy+1,xx  ) += r0*(-dy)       + r1*(1.f-dx);
                image_warped_deriv->at<cv::Point3f>(yy+1,xx+1) += r0*dy          + r1*dx;
            }
        }
    }
}

}
