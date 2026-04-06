#include "postprocessing.h"
#include <algorithm>

std::vector<Detection> parseDetections(const float* output_data, int num_detections,
                                       int orig_w, int orig_h, int input_size,
                                       float conf_threshold)
{
    std::vector<Detection> detections;

    float scale = std::min(static_cast<float>(input_size) / orig_w,
                           static_cast<float>(input_size) / orig_h);
    int pad_left = (input_size - static_cast<int>(orig_w * scale)) / 2;
    int pad_top  = (input_size - static_cast<int>(orig_h * scale)) / 2;

    for (int i = 0; i < num_detections; ++i) {
        const float* det = output_data + i * 6;

        float x1 = det[0];
        float y1 = det[1];
        float x2 = det[2];
        float y2 = det[3];
        float conf = det[4];
        int class_id = static_cast<int>(det[5]);

        if (conf < conf_threshold) continue;

        x1 = (x1 - pad_left) / scale;
        y1 = (y1 - pad_top)  / scale;
        x2 = (x2 - pad_left) / scale;
        y2 = (y2 - pad_top)  / scale;

        x1 = std::max(0.0f, std::min(x1, static_cast<float>(orig_w)));
        y1 = std::max(0.0f, std::min(y1, static_cast<float>(orig_h)));
        x2 = std::max(0.0f, std::min(x2, static_cast<float>(orig_w)));
        y2 = std::max(0.0f, std::min(y2, static_cast<float>(orig_h)));

        detections.push_back({x1, y1, x2, y2, conf, class_id});
    }

    return detections;
}
