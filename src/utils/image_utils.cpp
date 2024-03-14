#include "utils/image_utils.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <assert.h>
#include <glog/logging.h>

void save_image_maxabs(const cv::Mat& img, const std::string& filename)
{
  // Write image to disk, normalizing by maximum absolute value and having the zero level at mid-gray
  //
  // To write the image as it is, to the range [0,255], do:
  // cv::Mat img_out;
  // cv::imwrite(filename, cv::normalize(img, img_out, 0, 255, NORM_MINMAX, CV_8UC1));

  // Compute min and max values, otherwise, use
  double min_val, max_val;
  cv::minMaxLoc(img, &min_val, &max_val);
  //std::cout << "save_image: min= " << min_val << "  max=" << max_val;

  // Convert the image to the range [0,255]
  max_val = std::max(fabs(max_val),fabs(min_val));
  cv::Mat img_out = (img + max_val) * (255. / (2.*max_val));

  // Write image to disk
  VLOG(2) << "Saving " << filename;
  cv::imwrite(filename, img_out);
}


void save_image_minmax(const cv::Mat& img, const std::string& filename)
{
  // Write image to disk, converting it to the range [0,255]
  cv::Mat img_out;
  cv::normalize(img, img_out, 0, 255, cv::NORM_MINMAX, CV_8UC1);
  VLOG(4) << "Saving " << filename;
  cv::imwrite(filename, img_out);
}

// -----------------------------------------------------------------------------
cv::Mat saveDerivativeImages(
        const std::vector<cv::Mat>& iwe_deriv,
        const std::stringstream& ss
        )
{
    // Generate an image collecting all derivative images, in 3-column format
    const int num_cols = 3*iwe_deriv.at(0).cols;
    const int num_rows = (iwe_deriv.size()/3)*iwe_deriv.at(0).rows;
    cv::Mat img_grid = cv::Mat::zeros(num_rows,num_cols,CV_32FC1);

    for (size_t i=0, j=0; i < iwe_deriv.size(); i+=3, j++)
    {
        cv::Mat img_dummy, img_dummy_concat;
        cv::hconcat(iwe_deriv.at(i), iwe_deriv.at(i+1), img_dummy);
        cv::hconcat(img_dummy, iwe_deriv.at(i+2), img_dummy_concat);
        cv::Mat orows = img_grid.rowRange(j*iwe_deriv.at(0).rows,(j+1)*iwe_deriv.at(0).rows);
        img_dummy_concat.copyTo(orows);
    }

    save_image_maxabs(img_grid, ss.str());
    return img_grid;
}

/**
* \brief Compute robust min and max values (statistics) of an image
* \note Requires sorting
*/
void minMaxLocRobust(const cv::Mat& image, double& rmin, double& rmax,
                     const double& percentage_pixels_to_discard)
{
  cv::Mat image_as_row = image.reshape(0,1);
  cv::Mat image_as_row_sorted;
  cv::sort(image_as_row, image_as_row_sorted, cv::SORT_EVERY_ROW + cv::SORT_ASCENDING);
  image_as_row_sorted.convertTo(image_as_row_sorted, CV_64FC1);
  const int single_row_idx_min = (0.5f*percentage_pixels_to_discard/100.f)*image.total();
  const int single_row_idx_max = (1.f - 0.5f*percentage_pixels_to_discard/100.f)*image.total();
  rmin = image_as_row_sorted.at<double>(single_row_idx_min);
  rmax = image_as_row_sorted.at<double>(single_row_idx_max);
}


/**
* \brief Normalize image to the range [0,255] using robust min and max values
*/
void normalize(const cv::Mat& src, cv::Mat& dst, const double& percentage_pixels_to_discard)
{
  double rmin_val, rmax_val;
  minMaxLocRobust(src, rmin_val, rmax_val, percentage_pixels_to_discard);
  //std::cout << "min_val: " << rmin_val << ", " << "max_val: " << rmax_val << std::endl;
  const double scale = ((rmax_val != rmin_val) ? 255.f / (rmax_val - rmin_val) : 1.f);
  cv::Mat state_image_normalized = scale * (src - rmin_val);
  state_image_normalized.convertTo(dst, CV_8UC1);
}
