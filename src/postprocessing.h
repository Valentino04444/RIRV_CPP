#pragma once

#include <vector>
#include "types.h"

std::vector<Detection> parseDetections(const float* output_data, int num_detections,
                                       int orig_w, int orig_h, int input_size = 640,
                                       float conf_threshold = 0.25f);
