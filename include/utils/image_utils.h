#pragma once

#include <opencv2/core.hpp>
#include <iostream>

// Image processing utilities

void save_image_maxabs(const cv::Mat& img, const std::string& filename);

void save_image_minmax(const cv::Mat& img, const std::string& filename);

cv::Mat saveDerivativeImages(const std::vector<cv::Mat>& iwe_deriv, const std::stringstream& ss);

void minMaxLocRobust(const cv::Mat& image, double& rmin, double& rmax,
                     const double& percentage_pixels_to_discard);

void normalize(const cv::Mat& src, cv::Mat& dst, const double& percentage_pixels_to_discard);
