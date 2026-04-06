#include "renderer.h"
#include <string>

void drawDetections(cv::Mat& frame, const std::vector<Detection>& detections)
{
    for (const auto& d : detections) {
        cv::Scalar color = (d.class_id == 0) ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
        cv::rectangle(frame, cv::Point(static_cast<int>(d.x1), static_cast<int>(d.y1)),
                      cv::Point(static_cast<int>(d.x2), static_cast<int>(d.y2)), color, 2);

        std::string label = "Class " + std::to_string(d.class_id) + " " + std::to_string(d.conf).substr(0, 4);
        cv::putText(frame, label, cv::Point(static_cast<int>(d.x1), static_cast<int>(d.y1) - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 2);
    }
}

void drawInfo(cv::Mat& frame, int fps, size_t num_detections)
{
    std::string fps_text = "FPS: " + std::to_string(fps);
    cv::putText(frame, fps_text, cv::Point(10, 30),
                cv::FONT_HERSHEY_SIMPLEX, 1.0,
                cv::Scalar(0, 255, 0), 2);

    std::string det_text = "Dets: " + std::to_string(num_detections);
    cv::putText(frame, det_text, cv::Point(10, 60),
                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 255), 2);
}
