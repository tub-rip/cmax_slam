#include "backend/global_optim_contrast_gsl.h"
#include "backend/pose_graph_optimizer.h"
#include "backend/global_focus_funcs.h"

#include <glog/logging.h>

#include <gsl/gsl_multimin.h>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <sstream> //to save image using stringstream
#include <iomanip>

void cmax_slam::PoseGraphOptimizer::setupProblemAndOptimize_gsl()
{
    //PREPARE SOLVER
    //Solver/minimizer type (algorithm):
    const gsl_multimin_fdfminimizer_type *solver_type;
    solver_type = gsl_multimin_fdfminimizer_conjugate_fr; // Fletcher-Reeves conjugate gradient algorithm

    //Routines to compute the cost function and its derivatives
    gsl_multimin_function_fdf solver_info;

    const int num_params = 3 * num_cp_opt_;
    solver_info.n = num_params; // Size of the parameter vector

    // Defined in optim_contrast_gsl_analytical.cpp
    solver_info.f = global_contrast_f; // Cost function
    solver_info.df = global_contrast_df; // Gradient of cost function
    solver_info.fdf = global_contrast_fdf; // Cost and gradient functions

    solver_info.params = this; // Auxiliary data

    //Initial parameter vector (incremental angles with respect to control poses)
    gsl_vector *vx = gsl_vector_alloc(num_params);
    gsl_vector_set_zero(vx);

    //Initialize solver
    gsl_multimin_fdfminimizer *solver = gsl_multimin_fdfminimizer_alloc(solver_type, num_params);
    const double initial_step_size = 0.1;
    const double tol = 0.1; // for solvers fr, pr and bfgs

    // This call already evaluates the function
    gsl_multimin_fdfminimizer_set(solver, &solver_info, vx, initial_step_size, tol);

    //const double initial_cost = contrast_f(vx, &oAuxdata);
    const double initial_cost = solver->f;

    //ITERATE
    const int num_max_line_searches = 50;
    int status;
    const double epsabs_grad = 1e-4, tolfun = 1e-4;

//    const int num_max_line_searches = params.optim_params.max_num_iters;
//    int status;
//    const double epsabs_grad = params.optim_params.gradient_tolerance,
//                 tolfun = params.optim_params.function_tolerance;

    double cost_new = 1e9, cost_old = 1e9;
    size_t iter = 0;

    VLOG(2) << "Optimization. Solver type = " << solver_type->name;
    VLOG(2) << "iter=" << std::setw(3) << iter << "  param vec =["
            << gsl_vector_get(solver->x, 0) << " "
            << gsl_vector_get(solver->x, 1) << " "
            << gsl_vector_get(solver->x, 2) << "...]  cost=" << std::setprecision(8) << solver->f;

    do
    {
        iter++;
        cost_old = cost_new;
        status = gsl_multimin_fdfminimizer_iterate (solver);
        //status == GLS_SUCCESS (0) means that the iteration reduced the function value

        VLOG(2) << "iter=" << std::setw(1) << iter << "  param vec=["
                << gsl_vector_get(solver->x, 0) << " "
                << gsl_vector_get(solver->x, 1) << " "
                << gsl_vector_get(solver->x, 2) << "...]  cost=" << std::setprecision(8) << solver->f;

        if (status == GSL_SUCCESS)
        {
            //Test convergence due to stagnation in the value of the function
            cost_new = gsl_multimin_fdfminimizer_minimum(solver);

            double func_change_rate = fabs(1-cost_new/(cost_old + 1e-7));
            VLOG(3) << "func_change_rate = " << func_change_rate;

            if (func_change_rate < tolfun )
            {
                VLOG(1) << "progress tolerance reached.";
                break;
            }
            else
                status = GSL_CONTINUE;
        }

        // Test convergence due to absolute norm of the gradient
        if (GSL_SUCCESS == gsl_multimin_test_gradient (solver->gradient, epsabs_grad))
        {
            VLOG(1) << "gradient tolerance reached.";
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

    // Convert from GSL to OpenCV format
    gsl_vector *final_x = gsl_multimin_fdfminimizer_x(solver);

    // Get the optimal delta rotation vector
    std::vector<Eigen::Vector3d> optimal_drotv;
    for (int i = 0; i < num_cp_opt_; i++)
    {
        Eigen::Vector3d delta_rot;
        delta_rot << gsl_vector_get(final_x,3*i),
                     gsl_vector_get(final_x,3*i+1),
                     gsl_vector_get(final_x,3*i+2);
        optimal_drotv.push_back(delta_rot);
    }

    // Update to get the optimal trajectory
    traj_->incrementalUpdate(optimal_drotv, idx_cp_opt_beg_);
    VLOG(3) << "Get the optimal trajectory...";

    const double final_cost = gsl_multimin_fdfminimizer_minimum(solver);

    VLOG(1) << "--- Initial cost = " << std::setprecision(8) << initial_cost;
    VLOG(1) << "--- Final cost   = " << std::setprecision(8) << final_cost;
    VLOG(1) << "--- iter=" << std::setw(3) << iter;
    VLOG(2) << "--- function evaluations + gradient evaluations = "
            << cmax_slam::fcount << " + " << cmax_slam::gcount;
    VLOG(1) << "****************************************************";

    //Release memory used during optimization
    gsl_multimin_fdfminimizer_free (solver);
    gsl_vector_free (vx);
}
