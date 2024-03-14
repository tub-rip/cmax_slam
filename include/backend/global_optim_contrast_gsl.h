#pragma once

#include "backend/global_focus_funcs.h"
#include "backend/pose_graph_optimizer.h"
#include <opencv2/core.hpp>
#include <gsl/gsl_vector.h>


// GNU-GSL optimizer: Analytical derivative
void global_contrast_fdf (const gsl_vector *v, void* ptr, double *f, gsl_vector *df);
double global_contrast_f (const gsl_vector *v, void* ptr);
void global_contrast_df (const gsl_vector *v, void* ptr, gsl_vector *df);
void global_testDeriv_contrast_fdf (const gsl_vector *v, void* ptr, double *f, gsl_vector *df);
