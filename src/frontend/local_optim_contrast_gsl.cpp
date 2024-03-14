#include "frontend/ang_vel_estimator.h"
#include "frontend/local_focus_funcs.h"
#include "utils/image_utils.h" // for saving images

#include <gsl/gsl_vector.h>
#include <gsl/gsl_multimin.h>
#include <gsl/gsl_blas.h>

#include <opencv2/highgui.hpp>
#include <glog/logging.h>

#include <sstream> //to save image using stringstream
#include <iomanip>
#include <chrono>

static bool bSaveImage = false;

// -----------------------------------------------------------------------------
// GNU-GSL optimizer: Analytical derivative
void local_contrast_fdf (const gsl_vector *v, void *ptr, double *f, gsl_vector *df)
{
    cmax_slam::AngVelEstimator* estimator = (cmax_slam::AngVelEstimator*) ptr;

    // Parameter vector (from GSL to OpenCV)
    cv::Point3d ang_vel(gsl_vector_get(v,0), gsl_vector_get(v,1), gsl_vector_get(v,2));

    // Compute cost
    static cv::Mat iwe;
    static cv::Mat image_warped_deriv;
    static cv::Mat* image_warped_deriv_ptr;
    // pointer can be nullptr when contrast_fdf is called from contrast_f (to compute the cost alone)
    image_warped_deriv_ptr = (df == nullptr) ? nullptr : &image_warped_deriv;

    estimator->computeImageOfWarpedEvents(ang_vel, &iwe,
                                          image_warped_deriv_ptr);

    // Compute contrast given the event image
    static cv::Matx13d gradient;
    static cv::Matx13d* gradient_ptr;
    gradient_ptr = (df == nullptr) ? nullptr : &gradient;
    const double contrast = cmax_slam::computeContrast(iwe,
                                                       image_warped_deriv_ptr,
                                                       gradient_ptr,
                                                       estimator->params.process_opt.contrast_measure);
    VLOG(4) << "contrast = " << std::setprecision(10) << contrast;

    // Output
    *f = -contrast; // change sign: minimize -contrast
    if (df != nullptr)
    {
        // change sign: minimize -contrast
        gsl_vector_set(df, 0, -gradient(0));
        gsl_vector_set(df, 1, -gradient(1));
        gsl_vector_set(df, 2, -gradient(2));
    }
}

double local_contrast_f (const gsl_vector *v, void *adata)
{
    double cost;
    local_contrast_fdf (v, adata, &cost, nullptr);
    return cost;
}


void local_contrast_df (const gsl_vector *v, void *adata, gsl_vector *df)
{
    double cost;
    local_contrast_fdf (v, adata, &cost, df);
}


// -----------------------------------------------------------------------------
double cmax_slam::AngVelEstimator::setupProblemAndOptimize_gsl(cv::Point3d& ang_vel_)
{
    //PREPARE SOLVER
    // Choose a solver/minimizer type (algorithm)
    const gsl_multimin_fdfminimizer_type *solver_type;
    // A. Non-linear conjugate gradient
    solver_type = gsl_multimin_fdfminimizer_conjugate_fr; // Fletcher-Reeves
    //solver_type = gsl_multimin_fdfminimizer_conjugate_pr; // Polak-Ribiere
    // B. quasi-Newton methods: Broyden-Fletcher-Goldfarb-Shanno
    //solver_type = gsl_multimin_fdfminimizer_vector_bfgs; // BFGS
    //solver_type = gsl_multimin_fdfminimizer_vector_bfgs2; // BFGS2 is not as good

    //Routines to compute the cost function and its derivatives
    gsl_multimin_function_fdf solver_info;

    const int num_params = 3; // Size of angular velocity
    solver_info.n = num_params; // Size of the parameter vector

    solver_info.f = local_contrast_f; // Cost function
    solver_info.df = local_contrast_df; // Gradient of cost function
    solver_info.fdf = local_contrast_fdf; // Cost and gradient functions

    solver_info.params = this; // Auxiliary data

    //Initial parameter vector
    gsl_vector *vx = gsl_vector_alloc (num_params);
    gsl_vector_set (vx, 0, ang_vel_.x);
    gsl_vector_set (vx, 1, ang_vel_.y);
    gsl_vector_set (vx, 2, ang_vel_.z);

    //Initialize solver
    gsl_multimin_fdfminimizer *solver = gsl_multimin_fdfminimizer_alloc (solver_type, num_params);
    const double initial_step_size = 0.1;

    double tol = 0.05; // for solvers fr, pr and bfgs
    std::string str_bfgs2("vector_bfgs2");
    if (str_bfgs2.compare(solver_type->name) == 0){tol = 0.8;}

    // This call already evaluates the function
    gsl_multimin_fdfminimizer_set (solver, &solver_info, vx, initial_step_size, tol);

    //const double initial_cost = contrast_f(vx, &oAuxdata);
    const double initial_cost = solver->f;

    //ITERATE
    const int num_max_line_searches = 50;
//    const int num_max_line_searches = params.optim_params.max_num_iters;
    int status;
    const double epsabs_grad = 1e-3, tolfun = 1e-4;
//    const double epsabs_grad = params.optim_params.gradient_tolerance,
//                 tolfun = params.optim_params.function_tolerance;
    double cost_new = 1e9, cost_old = 1e9;
    size_t iter = 0;

    VLOG(3) << "Optimization. Solver type = " << solver_type->name;
    VLOG(3) << "iter=" << std::setw(3) << iter << "  ang_vel=["
            << gsl_vector_get(solver->x, 0) << " "
            << gsl_vector_get(solver->x, 1) << " "
            << gsl_vector_get(solver->x, 2) << "]  cost=" << std::setprecision(8) << solver->f;

    do
    {
        iter++;
        cost_old = cost_new;
        status = gsl_multimin_fdfminimizer_iterate (solver);
        //status == GLS_SUCCESS (0) means that the iteration reduced the function value

        VLOG(3) << "iter=" << std::setw(3) << iter << "  ang_vel=["
                << gsl_vector_get(solver->x, 0) << " "
                << gsl_vector_get(solver->x, 1) << " "
                << gsl_vector_get(solver->x, 2) << "]  cost=" << std::setprecision(8) << solver->f;

        /*
    // Save intermediate images of warped events during iteration.
    // Typically, the images are not that different.
    cv::Point3d ang_vel_iter;
    ang_vel_iter.x = gsl_vector_get(solver->x, 0);
    ang_vel_iter.y = gsl_vector_get(solver->x, 1);
    ang_vel_iter.z = gsl_vector_get(solver->x, 2);

    cv::Mat image_warped_iter;
    OptionsWarp opts_warp_display = oAuxdata.opts->opts_warp;
    opts_warp_display.blur_sigma = 0.;
    computeImageOfWarpedEvents(ang_vel_iter, events_subset_, cam_,
                              precomputed_bearing_vectors_, &image_warped_iter,
                              nullptr, opts_warp_display);

    double min_val, max_val;
    cv::minMaxLoc(image_warped_iter, &min_val, &max_val);
    LOG(INFO) << "min_val = " << min_val << "  maxval = " << max_val;

    // Scale the image to full range [0,255]
    cv::normalize(image_warped_iter, image_warped_iter, 0.f, 255.0f, cv::NORM_MINMAX, CV_32FC1);
    // Invert "color": dark events over white background for better visualization
    image_warped_iter = 255.0f - image_warped_iter;

    std::stringstream ss;
    ss << "/tmp/event_image_iter_" << std::setfill('0') << std::setw(8)
       << packet_number << "_" << std::setw(3) << iter << ".jpg";
    cv::imwrite(ss.str(), image_warped_iter );
    */

        if (status == GSL_SUCCESS)
        {
            //Test convergence due to stagnation in the value of the function
            cost_new = gsl_multimin_fdfminimizer_minimum(solver);
            if ( fabs( 1-cost_new/(cost_old + 1e-7) ) < tolfun )
            {
                VLOG(3) << "progress tolerance reached.";
                break;
            }
            else
                status = GSL_CONTINUE;
        }

        //Test convergence due to absolute norm of the gradient
        if (GSL_SUCCESS == gsl_multimin_test_gradient (solver->gradient, epsabs_grad))
        {
            VLOG(3) << "gradient tolerance reached.";
            break;
        }

        if (status != GSL_CONTINUE)
        {
            // The iteration was not successful (did not reduce the function value)
            VLOG(3) << "stopped iteration; status = " << status;
            VLOG_IF(3, (GSL_ENOPROG == status)) << "iteration is not making progress towards solution";
            break;
        }
    }
    while (status == GSL_CONTINUE && iter < num_max_line_searches);

    //SAVE RESULTS (best angular velocity)

    //Convert from GSL to OpenCV format
    gsl_vector *final_x = gsl_multimin_fdfminimizer_x(solver);
    ang_vel_.x = gsl_vector_get(final_x,0);
    ang_vel_.y = gsl_vector_get(final_x,1);
    ang_vel_.z = gsl_vector_get(final_x,2);

    //const double final_cost = contrast_f(final_x, &oAuxdata);
    //const double final_cost = solver->f;
    const double final_cost = gsl_multimin_fdfminimizer_minimum(solver);

    VLOG(3) << "--- Initial cost = " << std::setprecision(8) << initial_cost;
    VLOG(3) << "--- Final cost   = " << std::setprecision(8) << final_cost;
    VLOG(3) << "--- iter=" << std::setw(3) << iter << "  ang_vel=["
            << ang_vel_.x << " " << ang_vel_.y << " " << ang_vel_.z << "]";
    VLOG(3) << "--- function evaluations + gradient evaluations = "
            << cmax_slam::fcount << " + " << cmax_slam::gcount;

    ::bSaveImage = true;
    status = gsl_multimin_fdfminimizer_iterate(solver);

    //Release memory used during optimization
    gsl_multimin_fdfminimizer_free (solver);
    gsl_vector_free (vx);

    return final_cost;
}
