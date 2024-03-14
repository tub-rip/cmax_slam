#include "frontend/local_focus_funcs.h"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <glog/logging.h>
#include <math.h>

static constexpr int num_channels = 3; // angular velocity has 3 parameters

double contrast_MeanSquare(const cv::Mat& img, cv::Matx13d* gradient,
  std::vector<cv::Mat>& channels)
{
  // MEAN SQUARE
  const int num_pixels = img.rows * img.cols;
  const double contrast = cv::norm(img, cv::NORM_L2SQR) / static_cast<double>(num_pixels);

  if (gradient != nullptr)
  {
    for (int ii = 0; ii < num_channels; ii++)
    {
      (*gradient)(ii) = 2.* cv::mean(img.mul(channels.at(ii)))[0];
    }
  }
  return contrast;
}

double contrast_Variance(const cv::Mat& img, cv::Matx13d* gradient,
  std::vector<cv::Mat>& channels)
{
  // Variance
  static cv::Vec4d mean, stddev;
  cv::meanStdDev(img,mean,stddev);
  const double contrast = stddev[0]*stddev[0];

  if (gradient != nullptr)
  {
    cv::Mat img_zeromean = 2.*(img - mean[0]);
    for (int ii = 0; ii < num_channels; ii++)
    {
      (*gradient)(ii) =
        cv::mean( img_zeromean.mul(channels.at(ii) - cv::mean(channels.at(ii))[0]) )[0];
    }
  }
  return contrast;
}


double contrast_ImageGradientMagnitude(const cv::Mat& img, cv::Matx13d* gradient,
  std::vector<cv::Mat>& channels)
{
  // Compute magnitude of the gradient of the IWE
  cv::Mat grad_x, grad_y;
  cv::Sobel( img, grad_x, CV_32FC1, 1, 0);
  cv::Sobel( img, grad_y, CV_32FC1, 0, 1);
  // L1 or L2 norm of the (high-frequency) Gradient magnitude of the IWE should work
  // Use squared L2 norm for easy analytical derivative
  cv::Mat img_high_freq = grad_x.mul(grad_x) + grad_y.mul(grad_y);
  const double contrast = cv::mean(img_high_freq)[0];
  //LOG(INFO) << "GradMagnitude Contrast L2 = " << contrast;

  if (gradient != nullptr)
  {
    cv::Mat img_deriv_grad_x, img_deriv_grad_y;
    for (int i=0; i < num_channels; i++)
    {
      // Analytical derivative by swapping derivatives (Schwarz's theorem)
      cv::Sobel( channels.at(i), img_deriv_grad_x, CV_32FC1, 1, 0);
      cv::Sobel( channels.at(i), img_deriv_grad_y, CV_32FC1, 0, 1);
      (*gradient)(i) =
        2. * cv::mean( grad_x.mul(img_deriv_grad_x) + grad_y.mul(img_deriv_grad_y) )[0];
    }
  }
  return contrast;
}


namespace cmax_slam
{

// Counters for function and gradient evaluations
unsigned int fcount = 0, gcount = 0;

double computeContrast(
  const cv::Mat& img,
  cv::Mat* img_deriv,
  cv::Matx13d* gradient,
  const int contrast_measure
)
{
  std::vector<cv::Mat> channels;
  if (gradient != nullptr)
  {
    CHECK_NOTNULL(img_deriv);
    cv::split(*img_deriv, channels);
  }

  // Branch according to contrast / focus measure
  double contrast;
  switch (contrast_measure)
  {
    case MEAN_SQUARE_CONTRAST:
      contrast = contrast_MeanSquare(img, gradient, channels);
      break;
    case IMAGE_GRADIENT_MAGNITUDE_CONTRAST:
      contrast = contrast_ImageGradientMagnitude(img, gradient, channels);
      break;
    default:
      contrast = contrast_Variance(img, gradient, channels);
      break;
  }

  // Counters. Increment here instead of within each function
  fcount++;
  if (gradient != nullptr)
  {
    gcount++;
  }

  //LOG(INFO) << "C = " << std::setprecision(15) << contrast;
  return contrast;
}

} // namespace
