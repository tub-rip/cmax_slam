#pragma once

#include <opencv2/core.hpp>

namespace cmax_slam {

enum {
  VARIANCE_CONTRAST, // 0
  MEAN_SQUARE_CONTRAST,
  IMAGE_GRADIENT_MAGNITUDE_CONTRAST
};

double computeContrast(
  const cv::Mat& image,
  cv::Mat* image_deriv,
  cv::Matx13d* gradient,
  const int contrast_measure
);

// Counters for function and gradient evaluations
extern unsigned int fcount, gcount;

} // namespace
