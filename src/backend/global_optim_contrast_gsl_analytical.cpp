#include "backend/global_optim_contrast_gsl.h"

#include <glog/logging.h>
#include <gsl/gsl_blas.h>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <sstream> // To save image using stringstream
#include <iomanip>
#include <chrono>

// -----------------------------------------------------------------------------
// GNU-GSL optimizer: Analytical derivative
// -----------------------------------------------------------------------------

void global_contrast_fdf (const gsl_vector *v, void *adata, double *f, gsl_vector *df)
{
    cmax_slam::PoseGraphOptimizer *rot_estimator = (cmax_slam::PoseGraphOptimizer *) adata;

    // Convert incremental rotation vector to a vector of Eigen::Vector3d
    std::vector<Eigen::Vector3d> drotv;
    for (int i = 0; i < rot_estimator->getNumOptControlPoses(); i++)
    {
        Eigen::Vector3d drot;
        drot << gsl_vector_get(v,3*i),
                gsl_vector_get(v,3*i+1),
                gsl_vector_get(v,3*i+2);
        drotv.push_back(drot);
    }

    // Incrementally update the trajectory
    cmax_slam::Trajectory* traj_temp = rot_estimator->copyAndUpdateTraj(drotv);

    // Compute image of warped events (panorama)
    static cv::Mat iwe;
    static std::vector<cv::Mat> iwe_deriv; // list of images
    static std::vector<cv::Mat>* iwe_deriv_ptr;
    // pointer df can be nullptr when contrast_fdf is called from contrast_f (to compute the cost alone)
    iwe_deriv_ptr = (df == nullptr) ? nullptr : &iwe_deriv;

    rot_estimator->computeImageOfWarpedEvents(traj_temp,
                                              &iwe,
                                              iwe_deriv_ptr);
    delete traj_temp;

    // Compute contrast (and gradient) given the IWE (and its derivatives)
    static cv::Mat gradient;
    static cv::Mat* gradient_ptr;
    gradient_ptr = (df == nullptr) ? nullptr : &gradient;
    const double contrast = cmax_slam::computeContrast(iwe, iwe_deriv_ptr, gradient_ptr,
                                                       rot_estimator->params.process_opt.contrast_measure);
    VLOG(3) << "contrast = " << contrast;

    // Output
    *f = -contrast; // change sign: minimize -constrast
    if (df != nullptr)
    {
        // change sign: minimize -constrast
        CHECK_NOTNULL(gradient_ptr);
        CHECK_EQ(iwe_deriv.size(), (*gradient_ptr).cols);
        for (size_t i = 0; i < iwe_deriv.size(); i++)
        {
            // change sign: minimize -constrast
            gsl_vector_set(df, i, -gradient.at<double>(0,i));
        }
    }
}

double global_contrast_f (const gsl_vector *v, void *adata)
{
    double cost;
    global_contrast_fdf (v, adata, &cost, nullptr);
    return cost;
}

void global_contrast_df (const gsl_vector *v, void *adata, gsl_vector *df)
{
    double cost;
    global_contrast_fdf (v, adata, &cost, df);
}
