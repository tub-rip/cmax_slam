#include "cmax_slam.h"
#include <glog/logging.h>
#include <camera_info_manager/camera_info_manager.h>

#include <string>
#include <sstream>
#include <filesystem>
#include <ctime>

namespace cmax_slam {

static const double rad2degFactor = 180.0 * M_1_PI;

CMaxSLAM::CMaxSLAM(ros::NodeHandle& nh)
    : nh_(nh)
    , pnh_("~")

{
    // Load test configurations
    // Topic info
    const std::string events_topic = pnh_.param<std::string>("events_topic", "/dvs/events");
    const std::string camera_info_topic = pnh_.param<std::string>("camera_info_topic", "/dvs/camera_info");

    LOG(INFO) << "Event topic: " << events_topic;
    LOG(INFO) << "Camera info topic: " << camera_info_topic;

    // Set up subscribers
    event_sub_ = nh_.subscribe(events_topic, 0, &CMaxSLAM::eventsCallback, this);
    camera_info_sub_ = nh_.subscribe(camera_info_topic, 0, &CMaxSLAM::cameraInfoCallback, this);
    got_camera_info_ = false;

    // Load prcessing options
    OptionsProcess process_opt;
    process_opt.contrast_measure = pnh_.param<int>("contrast_measure", 0);

    LOG(INFO) << "*************** Processing options ******************";
    LOG(INFO) << "Contrast Measure: " << ((process_opt.contrast_measure == 0)? "Variance": "Mean Square");

    // Load front-end configurations
    front_end_params_.process_opt = process_opt;
    front_end_params_.num_events_per_packet = pnh_.param<int>("num_events_per_packet", 30000);
    front_end_params_.dt_ang_vel = pnh_.param<double>("dt_ang_vel", 0.02);
    front_end_params_.warp_opt.blur_sigma = pnh_.param<double>("frontend_blur_sigma", 1.0);
    front_end_params_.warp_opt.event_batch_size = pnh_.param<int>("event_batch_size", 100);
    front_end_params_.warp_opt.event_sample_rate = pnh_.param<int>("frontend_event_sample_rate", 1);
    front_end_params_.data_opt.show_iwe = pnh_.param<bool>("show_local_iwe", false);

    LOG(INFO) << "*************** Front-end params ******************";
    LOG(INFO) << "frontend_blur_sigma: " << front_end_params_.warp_opt.blur_sigma;
    LOG(INFO) << "event_batch_size: " << front_end_params_.warp_opt.event_batch_size;
    LOG(INFO) << "frontend_event_sample_rate: " << front_end_params_.warp_opt.event_sample_rate;

    // Load back-end configurations
    back_end_params_.process_opt = process_opt;
    back_end_params_.sliding_window_opt.time_window_size = pnh_.param<double>("backend_time_window_size", 0.2);
    back_end_params_.sliding_window_opt.sliding_window_stride = pnh_.param<double>("backend_sliding_window_stride", 0.1);
    back_end_params_.warp_opt.blur_sigma = pnh_.param<double>("backend_blur_sigma", 1.0);
    back_end_params_.warp_opt.event_batch_size = front_end_params_.warp_opt.event_batch_size;
    back_end_params_.warp_opt.event_sample_rate = pnh_.param<int>("backend_event_sample_rate", 1);
    back_end_params_.traj_opt.dt_knots = pnh_.param<double>("dt_knots", 0.1);
    back_end_params_.traj_opt.spline_degree = pnh_.param<int>("spline_degree", 1);
    back_end_params_.data_opt.show_iwe = pnh_.param<bool>("show_pano_map", true);
    back_end_params_.map_opt.pano_height = pnh_.param<int>("pano_height", 1024);
    back_end_params_.map_opt.pano_width = 2 * back_end_params_.map_opt.pano_height;
    back_end_params_.map_opt.Y_angle = pnh_.param<double>("Y_angle", 0.0);
    back_end_params_.map_opt.backend_min_ev_rate = pnh_.param<int>("backend_min_ev_rate", 10);
    back_end_params_.map_opt.max_update_times = pnh_.param<int>("max_update_times", 10);
    back_end_params_.draw_FOV = pnh_.param<bool>("draw_FOV", false);
    back_end_params_.gamma = pnh_.param<double>("gamma", 0.75);

    LOG(INFO) << "*************** Back-end params ******************";
    LOG(INFO) << "time_window_size: " << back_end_params_.sliding_window_opt.time_window_size;
    LOG(INFO) << "sliding_window_stride: " << back_end_params_.sliding_window_opt.sliding_window_stride;
    LOG(INFO) << "backend_blur_sigma: " << back_end_params_.warp_opt.blur_sigma;
    LOG(INFO) << "event_batch_size: " << back_end_params_.warp_opt.event_batch_size;
    LOG(INFO) << "backend_event_sample_rate: " << back_end_params_.warp_opt.event_sample_rate;
    LOG(INFO) << "dt_knots: " << back_end_params_.traj_opt.dt_knots;
    LOG(INFO) << "spline_degree: " << back_end_params_.traj_opt.spline_degree;
    LOG(INFO) << "show_pano_map: " << back_end_params_.data_opt.show_iwe;
    LOG(INFO) << "pano_height: " << back_end_params_.map_opt.pano_height;
    LOG(INFO) << "Y-angle = " << back_end_params_.map_opt.Y_angle;
    LOG(INFO) << "draw_FOV = " << back_end_params_.draw_FOV;
    LOG(INFO) << "gamma = " << back_end_params_.gamma;

    // New a angular velocity estimator (the front-end runs in the main thread)
    ang_vel_estimator_ = new AngVelEstimator(&nh_);

    // New a pose graph optimizer
    pose_graph_optimizer_ = new PoseGraphOptimizer(&nh_);

    // Initialize the back-end thread and launch
    pose_graph_optim_ = new std::thread(&PoseGraphOptimizer::Run, pose_graph_optimizer_);

    // Set the pointer to each other
    ang_vel_estimator_->setBackend(pose_graph_optimizer_);
    pose_graph_optimizer_->setFrontend(ang_vel_estimator_);
}

CMaxSLAM::~CMaxSLAM()
{
    pose_graph_optim_->detach();
    delete ang_vel_estimator_;
    delete pose_graph_optimizer_;
}

void CMaxSLAM::precomputeBearingVectors()
{
    int sensor_width = cam.fullResolution().width;
    int sensor_height = cam.fullResolution().height;

    for(int y=0; y < sensor_height; y++)
    {
        for(int x=0; x < sensor_width; x++)
        {
            cv::Point2d rectified_point = cam.rectifyPoint(cv::Point2d(x,y));
            cv::Point3d bearing_vec = cam.projectPixelTo3dRay(rectified_point);
            precomputed_bearing_vectors.emplace_back(bearing_vec);
        }
    }
}

void CMaxSLAM::cameraInfoCallback(const sensor_msgs::CameraInfo::ConstPtr& camera_info)
{
    if(!got_camera_info_)
    {
        ROS_INFO("Loading camera information");
        cam.fromCameraInfo(camera_info);
        got_camera_info_ = true;

        ROS_INFO("Camera info got");
        camera_info_sub_.shutdown(); // no need to listen to this topic any more

        // Initialze the front-end
        precomputeBearingVectors();
        ang_vel_estimator_->initialize(&cam, front_end_params_, precomputed_bearing_vectors);

        // Intialize the back-end
        int camera_width = cam.fullResolution().width;
        int camera_height = cam.fullResolution().height;
        pose_graph_optimizer_->initialize(camera_width, camera_height,
                                          back_end_params_,
                                          &(ang_vel_estimator_->events_),
                                          &precomputed_bearing_vectors);
    }
}

void CMaxSLAM::eventsCallback(const dvs_msgs::EventArray::ConstPtr &msg)
{
    if(!got_camera_info_)
    {
        ROS_ERROR("Received events but camera info is still missing");
        return;
    }

    for (auto ev = msg->events.begin(); ev < msg->events.end();
         ev += front_end_params_.warp_opt.event_sample_rate)
    {
        // Push events into the frontend, which will pass them to the backend then.
        ang_vel_estimator_->pushEvent(*ev);
    }
}

}
