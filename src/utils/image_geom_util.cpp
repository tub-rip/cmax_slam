#include "utils/image_geom_util.h"

#include <glog/logging.h>
#include <opencv2/highgui.hpp>


void applyIntrinsics(
        const cv::Point2d& pt_in,
        const cv::Matx33d& camera_matrix,
        cv::Point2d* pt_out,
        cv::Matx22d* intrinsics_jacobian
        )
{
    // apply intrinsic parameters
    pt_out->x = camera_matrix(0,0) * pt_in.x + camera_matrix(0,2);
    pt_out->y = camera_matrix(1,1) * pt_in.y + camera_matrix(1,2);

    // and compute the correspoding Jacobian
    if (intrinsics_jacobian != nullptr)
        *intrinsics_jacobian = cv::Matx<double,2,2>(camera_matrix(0,0), 0.,
                                                    0., camera_matrix(1,1));
}

void canonicalProjection(
        const cv::Point3d& object_pt_cam,
        cv::Point2d* image_pt,
        cv::Matx23d* jacobian
        )
{
    const double inverse_depth = 1.0 / object_pt_cam.z;

    // calibrated coordinates
    image_pt->x = object_pt_cam.x * inverse_depth;
    image_pt->y = object_pt_cam.y * inverse_depth;

    // and compute the jacobian of the calibrated image coordinates
    // with respect to the input camera coordinates of the 3D point
    if (jacobian != nullptr)
        *jacobian = cv::Matx<double,2,3>(inverse_depth, 0.0, -image_pt->x * inverse_depth,
                                         0.0, inverse_depth, -image_pt->y * inverse_depth);
}

void rotatePoint3DFirstOrder(
        const cv::Point3d& ang_vel,
        const double dt,
        const cv::Point3d& point_3D,
        cv::Point3d* point_3D_rotated,
        cv::Matx33d* jacobian
        )
{
    // Rotated point using first order Taylor approx of Rotation matrix
    *point_3D_rotated = point_3D + dt*(ang_vel.cross(point_3D));

    // Derivative of rotated point
    // (using 1st order Taylor approx of Rotation matrix) wrt angular velocity
    if (jacobian != nullptr)
        cross2Matrix((-dt)*point_3D, jacobian); // skew-symmetric
}


void rotatePoint3D_calibCoords(
        const cv::Point3d& ang_vel,
        const double dt,
        const cv::Point3d& point_3D,
        cv::Point2d* calibrated_pt,
        cv::Matx23d* jacobian_calibrated_pt_wrt_ang_vel
        )
{
    // approximation: use only the first two terms of the series expansion of the rotation matrix
    static cv::Point3d point_3D_rotated;
    static cv::Matx33d jacobian_warped_event_wrt_ang_vel;
    static cv::Matx33d* jacobian_warped_event_wrt_ang_vel_ptr;
    if (jacobian_calibrated_pt_wrt_ang_vel == nullptr)
        jacobian_warped_event_wrt_ang_vel_ptr = nullptr;
    else
        jacobian_warped_event_wrt_ang_vel_ptr = &jacobian_warped_event_wrt_ang_vel;

    // First order approximation
    rotatePoint3DFirstOrder(ang_vel,dt,point_3D,&point_3D_rotated,jacobian_warped_event_wrt_ang_vel_ptr);

    // project point, to obtain pixels (natural binning)

    // calibrated coordinates
    static cv::Matx23d jacobian_calibrated_pt_wrt_warped_event;
    static cv::Matx23d* jacobian_calibrated_pt_wrt_warped_event_ptr;
    if (jacobian_calibrated_pt_wrt_ang_vel == nullptr)
        jacobian_calibrated_pt_wrt_warped_event_ptr = nullptr;
    else
        jacobian_calibrated_pt_wrt_warped_event_ptr = &jacobian_calibrated_pt_wrt_warped_event;

    canonicalProjection(point_3D_rotated,calibrated_pt,jacobian_calibrated_pt_wrt_warped_event_ptr);

    // compute jacobian
    if (jacobian_calibrated_pt_wrt_ang_vel != nullptr)
        *jacobian_calibrated_pt_wrt_ang_vel =
            jacobian_calibrated_pt_wrt_warped_event * jacobian_warped_event_wrt_ang_vel;
}
