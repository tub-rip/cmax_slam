#pragma once

#include <opencv2/core.hpp>

inline void cross2Matrix(const cv::Point3d& vec, cv::Matx33d* mat)
{
    *mat = cv::Matx<double,3,3>(0, -vec.z, vec.y, vec.z, 0, -vec.x, -vec.y, vec.x, 0);
}

void applyIntrinsics(
  const cv::Point2d& pt_in,
  const cv::Matx33d& camera_matrix,
  cv::Point2d* pt_out,
  cv::Matx22d* intrinsics_jacobian
);


void canonicalProjection(
  const cv::Point3d& object_pt_cam,
  cv::Point2d* image_pt,
  cv::Matx23d* jacobian
);


void rotatePoint3D_firstOrder(
  const cv::Point3d& ang_vel,
  const double dt,
  const cv::Point3d& point_3D,
  cv::Point3d* point_3D_rotated,
  cv::Matx33d* jacobian
);


void rotatePoint3D_calibCoords(
  const cv::Point3d& ang_vel,
  const double dt,
  const cv::Point3d& point_3D,
  cv::Point2d* calibrated_pt,
  cv::Matx23d* jacobian_calibrated_pt_wrt_ang_vel
);
