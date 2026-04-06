#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>

#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <cuda_runtime.h>

#include "logger.h"
#include "types.h"
#include "preprocessing.h"
#include "postprocessing.h"
#include "renderer.h"

int main()
{
    int fps = 0;
    int frame_counter = 0;
    auto start_time = std::chrono::high_resolution_clock::now();


    std::string video_path = "/home/valen/RIRV_Project/videos/DayDrive1.mp4";
    std::string engine_path = "/home/valen/RIRV_Project/models/yolo26n.engine";

    // Load engine
    std::ifstream file(engine_path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "Error opening engine: " << engine_path << std::endl;
        return -1;
    }
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> engine_data(size);
    file.read(engine_data.data(), size);
    file.close();

    Logger logger;
    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(logger);
    nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(engine_data.data(), size);
    nvinfer1::IExecutionContext* context = engine->createExecutionContext();

    if (!engine || !context) {
        std::cerr << "Failed to create engine or context!" << std::endl;
        return -1;
    }

    std::cout << "TensorRT engine loaded successfully!" << std::endl;

    // Tensor names (from your print)
    const char* input_name = "images";
    const char* output_name = "output0";

    // Allocate GPU buffers
    size_t input_elements = 1 * 3 * 640 * 640;
    size_t output_elements = 1 * 300 * 6;   // your exact output shape

    void* d_input = nullptr;
    void* d_output = nullptr;
    cudaMalloc(&d_input, input_elements * sizeof(float));
    cudaMalloc(&d_output, output_elements * sizeof(float));

    // Open video
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "Could not open video: " << video_path << std::endl;
        return -1;
    }

    std::cout << "Starting inference... Press 'q' to quit\n";

    cv::Mat frame;
    int frame_count = 0;
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    while (true) {
        cap >> frame;
        frame_count++;
        if (frame.empty()) {
            std::cout << "End of video after " << frame_count << " frames.\n";
            break;
        }

        int orig_h = frame.rows;
        int orig_w = frame.cols;

        cv::Mat letterboxed = letterbox(frame, 640);
        std::vector<float> input_tensor = preprocessImage(letterboxed);

        // Copy input to GPU
        cudaMemcpyAsync(d_input, input_tensor.data(), input_tensor.size() * sizeof(float),
                        cudaMemcpyHostToDevice, stream);

        // Set tensors (TensorRT 10+ style)
        context->setTensorAddress(input_name, d_input);
        context->setTensorAddress(output_name, d_output);

        // Run inference
        if (!context->enqueueV3(stream)) {
            std::cerr << "enqueueV3 failed on frame " << frame_count << std::endl;
        }
        cudaStreamSynchronize(stream);

        // Copy output back
        std::vector<float> output_tensor(output_elements);
        cudaMemcpyAsync(output_tensor.data(), d_output, output_elements * sizeof(float),
                        cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        // Parse output: [1, 300, 6] → xyxy, conf, class_id  (already NMS'ed)
        std::vector<Detection> detections = parseDetections(output_tensor.data(), 300, orig_w, orig_h);

        // Draw detections and overlay info
        drawDetections(frame, detections);

        // FPS Calculation
        frame_counter++;
        auto current_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = current_time - start_time;

        if (elapsed.count() >= 1.0) {
            fps = frame_counter;
            frame_counter = 0;
            start_time = current_time;

            std::cout << "Frame " << frame_count << " | FPS: " << fps 
                      << " | Detections: " << detections.size() << std::endl;
        }

        drawInfo(frame, fps, detections.size());

        cv::imshow("YOLO26n TensorRT Inference", frame);
        if (cv::waitKey(1) == 'q') break;   
    }

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    cudaStreamDestroy(stream);

    delete context;
    delete engine;
    delete runtime;

    std::cout << "Program finished!" << std::endl;
    return 0;
}
