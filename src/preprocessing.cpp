#include "preprocessing.h"

cv::Mat letterbox(const cv::Mat& image, int target_size, const cv::Scalar& pad_color)
{
    int ih = image.rows, iw = image.cols;
    float scale = std::min(static_cast<float>(target_size) / ih, static_cast<float>(target_size) / iw);
    int nh = static_cast<int>(ih * scale);
    int nw = static_cast<int>(iw * scale);

    cv::Mat resized;
    cv::resize(image, resized, cv::Size(nw, nh), 0, 0, cv::INTER_LINEAR);

    cv::Mat padded(target_size, target_size, CV_8UC3, pad_color);
    int top = (target_size - nh) / 2;
    int left = (target_size - nw) / 2;
    resized.copyTo(padded(cv::Rect(left, top, nw, nh)));
    return padded;
}

std::vector<float> preprocessImage(const cv::Mat& letterboxed_img)
{
    cv::Mat rgb;
    cv::cvtColor(letterboxed_img, rgb, cv::COLOR_BGR2RGB);
    cv::Mat float_img;
    rgb.convertTo(float_img, CV_32F, 1.0f / 255.0f);

    int h = float_img.rows, w = float_img.cols;
    std::vector<float> tensor(3 * h * w);

    float* tensor_ptr = tensor.data();
    const float* img_ptr = reinterpret_cast<const float*>(float_img.data);

    for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j) {
            const float* pixel = img_ptr + (i * w + j) * 3;
            tensor_ptr[0 * h * w + i * w + j] = pixel[0]; // R
            tensor_ptr[1 * h * w + i * w + j] = pixel[1]; // G
            tensor_ptr[2 * h * w + i * w + j] = pixel[2]; // B
        }
    }
    return tensor;
}
