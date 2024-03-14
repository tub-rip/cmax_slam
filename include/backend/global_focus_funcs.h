#pragma once

#include <opencv2/core.hpp>

namespace cmax_slam {

enum {
  VARIANCE_CONTRAST, // 0
  MEAN_SQUARE_CONTRAST,
};

double computeContrast(
  const cv::Mat& image,
  std::vector<cv::Mat>* image_deriv, // list of images
  cv::Mat* gradient,
  const int contrast_measure
);

// Counters for function and gradient evaluations
extern unsigned int fcount, gcount;

} // namespace
