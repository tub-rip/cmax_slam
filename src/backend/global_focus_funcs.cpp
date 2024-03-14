#include "backend/global_focus_funcs.h"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <glog/logging.h>

namespace cmax_slam
{
unsigned int global_fcount=0, global_gcount=0;
}

double contrast_MeanSquare(const cv::Mat& image, cv::Mat* gradient,
  const std::vector<cv::Mat>& channels)
{
  const int num_pixels = image.rows * image.cols;
  const double contrast = cv::norm(image, cv::NORM_L2SQR) / static_cast<double>(num_pixels);

  if (gradient != nullptr)
  {
    *gradient = cv::Mat::zeros(1,channels.size(),CV_64FC1);
    for (int i=0; i < channels.size(); i++)
    {
      gradient->at<double>(0,i) = 2.* cv::mean(image.mul(channels.at(i)))[0];
    }
  }
  return contrast;
}


double contrast_Variance(const cv::Mat& image, cv::Mat* gradient,
  const std::vector<cv::Mat>& channels)
{
  static cv::Vec4d mean, stddev;
  cv::meanStdDev(image,mean,stddev);
  const double contrast = stddev[0]*stddev[0];

  if (gradient != nullptr)
  {
    *gradient = cv::Mat::zeros(1,channels.size(),CV_64FC1);
    cv::Mat image_zeromean = 2.*(image - mean[0]);
    for (int i=0; i < channels.size(); i++)
    {
      cv::Scalar mean_ch = cv::mean(channels.at(i));
      gradient->at<double>(0,i) = cv::mean( image_zeromean.mul( channels.at(i)-mean_ch[0] ) )[0]; // eq(8) in Guillermo's paper
    }
  }
  return contrast;
}


namespace cmax_slam {

double computeContrast(
  const cv::Mat& image,
  std::vector<cv::Mat>* image_deriv, // list of images
  cv::Mat* gradient,
  const int contrast_measure
)
{
  // Branch according to contrast / focus measure
  double contrast;
  switch (contrast_measure)
  {
    case MEAN_SQUARE_CONTRAST:
      contrast = contrast_MeanSquare(image, gradient, *image_deriv);
      break;
    default:
      contrast = contrast_Variance(image, gradient, *image_deriv);
      break;
  }

  // Counters. Increment here instead of within each function
  cmax_slam::global_fcount++;
  if (gradient != nullptr)
  {
    cmax_slam::global_gcount++;
  }

  //LOG(INFO) << "C = " << std::setprecision(15) << contrast;
  return contrast;
}

} // namespace
