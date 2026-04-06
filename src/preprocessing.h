#pragma once

#include <vector>
#include <opencv2/opencv.hpp>

cv::Mat letterbox(const cv::Mat& image, int target_size = 640, const cv::Scalar& pad_color = cv::Scalar(114, 114, 114));
std::vector<float> preprocessImage(const cv::Mat& letterboxed_img);
