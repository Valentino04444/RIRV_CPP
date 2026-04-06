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
#include "preprocessing_gpu.h"
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

    // Read native video FPS for correct playback speed
    double video_fps = cap.get(cv::CAP_PROP_FPS);
    if (video_fps <= 0) video_fps = 30.0;
    double frame_time_ms = 1000.0 / video_fps;

    std::cout << "Video FPS: " << video_fps << " (" << frame_time_ms << " ms/frame)\n";

    // Pre-allocate pinned host memory for output (avoids alloc every frame)
    float* h_output = nullptr;
    cudaMallocHost(&h_output, output_elements * sizeof(float));

    // GPU preprocessing buffers (allocated on first frame)
    GpuPreprocessBuffers gpu_buf = {};

    // Set tensor addresses once (they don't change between frames)
    context->setTensorAddress(input_name, d_input);
    context->setTensorAddress(output_name, d_output);

    // Absolute frame pacing: track when each frame should be displayed
    auto next_frame_time = std::chrono::high_resolution_clock::now();
    auto frame_interval = std::chrono::duration_cast<std::chrono::high_resolution_clock::duration>(
        std::chrono::duration<double>(1.0 / video_fps));

    while (true) {
        cap >> frame;
        frame_count++;
        if (frame.empty()) {
            std::cout << "End of video after " << frame_count << " frames.\n";
            break;
        }

        int orig_h = frame.rows;
        int orig_w = frame.cols;

        // Allocate GPU buffers on first frame (now we know dimensions)
        if (!gpu_buf.d_frame) {
            initGpuPreprocessBuffers(gpu_buf, orig_w, orig_h);
        }

        // GPU preprocessing: upload raw BGR → letterbox+RGB+normalize+CHW → d_input
        preprocessGpu(frame.data, orig_w, orig_h, gpu_buf,
                      static_cast<float*>(d_input), 640, stream);

        // Run inference (preprocessGpu already enqueued async work on same stream)
        if (!context->enqueueV3(stream)) {
            std::cerr << "enqueueV3 failed on frame " << frame_count << std::endl;
        }

        // Copy output back to pinned host memory — single sync covers everything
        cudaMemcpyAsync(h_output, d_output, output_elements * sizeof(float),
                        cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        // Parse output: [1, 300, 6] → xyxy, conf, class_id  (already NMS'ed)
        std::vector<Detection> detections = parseDetections(h_output, 300, orig_w, orig_h);

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

        // Absolute frame pacing: wait until the target time for this frame
        next_frame_time += frame_interval;
        auto now = std::chrono::high_resolution_clock::now();
        int wait_ms = std::max(1, static_cast<int>(
            std::chrono::duration_cast<std::chrono::milliseconds>(next_frame_time - now).count()));
        if (cv::waitKey(wait_ms) == 'q') break;

        // If we fell behind, reset the pacer to avoid a burst of catch-up frames
        if (std::chrono::high_resolution_clock::now() > next_frame_time + frame_interval) {
            next_frame_time = std::chrono::high_resolution_clock::now();
        }
    }

    // Cleanup
    freeGpuPreprocessBuffers(gpu_buf);
    cudaFreeHost(h_output);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaStreamDestroy(stream);

    delete context;
    delete engine;
    delete runtime;

    std::cout << "Program finished!" << std::endl;
    return 0;
}
