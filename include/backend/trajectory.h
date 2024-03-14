#pragma once

#include <map>
#include <array>
#include <assert.h>

#include <opencv2/core.hpp>
#include <ros/ros.h>
#include <Eigen/Core>

#include "basalt/spline/so3_spline.h"

typedef std::pair<ros::Time, Sophus::SO3d> PoseEntry;
typedef std::map<ros::Time, Sophus::SO3d> PoseMap;
typedef std::vector<PoseEntry> PoseArray;

namespace cmax_slam {

struct TrajectorySettings{
    ros::Time t_beg;
    ros::Time t_end;
    double dt_knots;
};

class Trajectory
{
public:
    // Constructor
    virtual ~Trajectory() {}

    // Get timestampes for the start and the end of the trajectory
    double getBegTime() const { return t_beg_; }
//    double getEndTime() const { return t_end_; }

    // Get or change the number of control points (for re-initialization)
    virtual size_t size() = 0;
    virtual void resize(size_t n) = 0;

    // Get the number of control poses involved for one interpolation
    virtual int NumInvolvedControlPoses() const = 0;

    // Get or set the control point at given index
    virtual Sophus::SO3d getControlPose(const int idx) = 0;

    // Add multiple new control points at the tail of the trajectory (growing)
    virtual void pushbackCtrlPoses(const std::vector<Sophus::SO3d> &cps) = 0;

    // Evaluate trajectory at some timestamp
    virtual Sophus::SO3d evaluate(const ros::Time& t, int* idx_beg = nullptr,
                                  cv::Mat* jacobian = nullptr) = 0;

    virtual void incrementalUpdate(const std::vector<Eigen::Vector3d>& drotv, const int& idx_beg) = 0;

    virtual Trajectory* CopyAndIncrementalUpdate(const std::vector<Eigen::Vector3d>& drotv,
                                                 const int& idx_traj_beg,
                                                 const int& idx_opt_beg) = 0;

    // Generate control poses with new poses and time interval (mostly used for trajectory extension)
    virtual std::vector<Sophus::SO3d> generateCtrlPoses(PoseMap& poses,
                                                        const ros::Time& t_beg,
                                                        const ros::Time& t_end) = 0;

protected:
    // Trajectory configuration
    double t_beg_, dt_knots_;
    int64_t t_beg_ns_, dt_knots_ns_;

    /* Spline initializatoin */
    // Use given poses to compute initial control poses for the trajectory
    virtual void initializeCtrlPoses(PoseMap& poses,
                                     const ros::Time& t_beg,
                                     const ros::Time& t_end) = 0;
    // Fit control poses with the given poses (mostly called by the above functions)
    virtual std::vector<Sophus::SO3d> fitCtrlPoses(PoseMap& poses, const double t_beg, const int num_cps) = 0;

    // Interpolate a pose between two poses at the middle time point
    PoseEntry interpPoseMid(const PoseEntry& p1, const PoseEntry& p2);
};

class LinearTrajectory: public Trajectory
{
public:
    typedef typename basalt::So3Spline<2, double> SO3_Spline_Traj;
    typedef typename basalt::So3Spline<2, double>::JacobianStruct Jacobian_wrt_Control_Points;

    // Constructor
    LinearTrajectory(const TrajectorySettings& traj_config); // Create an empty trajectory (with no control poses)
    LinearTrajectory(PoseMap &poses, const TrajectorySettings& traj_config);
    LinearTrajectory(const double t_beg, const double dt_knots,
                     const std::vector<Sophus::SO3d>& ctrl_poses);

    // Deconstructor
    ~LinearTrajectory() { }

    // Get or change the number of control points (for re-initialization)
    size_t size() override { return spline_.size(); }
    void resize(size_t n) override { spline_.resize(n); }

    // Get the number of control poses involved for one interpolation
    int NumInvolvedControlPoses() const override { return 2; }

    // Get or set the control point at given index
    Sophus::SO3d getControlPose(const int idx) override;

    // Add new control points at the tail of the trajectory (growing)
    void pushbackCtrlPoses(const std::vector<Sophus::SO3d> &cps) override;

    // Query pose at given timestamp
    Sophus::SO3d evaluate(const ros::Time &t, int* idx_beg = nullptr,
                          cv::Mat *jacobian = nullptr) override; // jaocbian: 3X6

    // Update the trajectory by incremental rotation (delta R)
    void incrementalUpdate(const std::vector<Eigen::Vector3d>& drotv, const int& idx_beg) override;

    // Return a new trajectory that is updated from (a part of) the current trajectory using incremental rot_vec
    LinearTrajectory* CopyAndIncrementalUpdate (const std::vector<Eigen::Vector3d>& drotv,
                                                const int& idx_traj_beg,
                                                const int& idx_opt_beg) override;

    // Generate control poses with new poses and time interval (mostly used for trajectory extension)
    virtual std::vector<Sophus::SO3d> generateCtrlPoses(PoseMap& poses,
                                                        const ros::Time& t_beg,
                                                        const ros::Time& t_end) override;

protected:
    // Initialize control poses for this trajectory, with the poses on the trajectory
    void initializeCtrlPoses(PoseMap& poses,
                             const ros::Time& t_beg,
                             const ros::Time& t_end) override;

    // Fit control poses with the given poses (mostly called by the above functions)
    virtual std::vector<Sophus::SO3d> fitCtrlPoses(PoseMap& poses, const double t_beg, const int num_cps) override;

private:
    // Trajectory (linear spline)
    SO3_Spline_Traj spline_;
};

class CubicTrajectory: public Trajectory
{
public:
    typedef typename basalt::So3Spline<4, double> SO3_Spline_Traj;
    typedef typename basalt::So3Spline<4, double>::JacobianStruct Jacobian_wrt_Control_Points;

    // Constructor
    CubicTrajectory(const TrajectorySettings& traj_config); // Create an empty trajectory (with no control poses)
    CubicTrajectory(PoseMap &poses, const TrajectorySettings& traj_config);
    CubicTrajectory(const double t_beg, const double dt_knots,
                     const std::vector<Sophus::SO3d>& ctrl_poses);
    // Deconstructor
    ~CubicTrajectory() { }

    // Get or change the number of control points (for re-initialization)
    size_t size() override { return spline_.size(); }
    void resize(size_t n) override { spline_.resize(n); }

    // Get the number of control poses involved for one interpolation
    int NumInvolvedControlPoses() const override { return 4; }

    // Get or set the control point at given index
    Sophus::SO3d getControlPose(const int idx) override;

    // Add new control points at the tail of the trajectory (growing)
    void pushbackCtrlPoses(const std::vector<Sophus::SO3d> &cps) override;

    // Query pose at given timestamp
    Sophus::SO3d evaluate(const ros::Time &t, int* idx_beg = nullptr,
                          cv::Mat *jacobian = nullptr) override; // jacobian: 3X12

    // Update the trajectory by incremental rotation (delta R)
    void incrementalUpdate(const std::vector<Eigen::Vector3d>& drotv, const int& idx_beg) override;

    // Return a new trajectory that is updated from the current trajectory using incremental rot_vec
    CubicTrajectory* CopyAndIncrementalUpdate(const std::vector<Eigen::Vector3d>& drotv,
                                              const int& idx_traj_beg,
                                              const int& idx_opt_beg) override;

    // Generate control poses with new poses and time interval (mostly used for trajectory extension)
    virtual std::vector<Sophus::SO3d> generateCtrlPoses(PoseMap& poses,
                                                        const ros::Time& t_beg,
                                                        const ros::Time& t_end) override;

protected:
    // Initialize control poses for this trajectory, with the poses on the trajectory
    void initializeCtrlPoses(PoseMap& poses,
                             const ros::Time& t_beg,
                             const ros::Time& t_end) override;
    // Fit control poses with the given poses (mostly called by the above functions)
    virtual std::vector<Sophus::SO3d> fitCtrlPoses(PoseMap& poses, const double t_beg, const int num_cps) override;

private:
    // Trajectory (cubic spline)
    SO3_Spline_Traj spline_;
};

}
