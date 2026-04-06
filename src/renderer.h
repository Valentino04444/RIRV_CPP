#pragma once

#include <vector>
#include <opencv2/opencv.hpp>
#include "types.h"

void drawDetections(cv::Mat& frame, const std::vector<Detection>& detections);
void drawInfo(cv::Mat& frame, int fps, size_t num_detections);
