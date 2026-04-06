#pragma once

#include <cuda_runtime.h>

// Allocates pinned host memory and GPU buffer for the raw frame.
// Call once at startup after knowing frame dimensions.
struct GpuPreprocessBuffers {
    unsigned char* d_frame;     // raw BGR frame on GPU
    unsigned char* h_frame;     // pinned host memory for fast H2D
    size_t frame_bytes;         // allocated size in bytes
    int alloc_w, alloc_h;       // dimensions we allocated for
};

void initGpuPreprocessBuffers(GpuPreprocessBuffers& buf, int frame_w, int frame_h);
void freeGpuPreprocessBuffers(GpuPreprocessBuffers& buf);

// Runs the full preprocessing pipeline on GPU:
//   upload raw BGR frame → letterbox resize + BGR→RGB + normalize + HWC→CHW
// Result is written directly into d_input (the TensorRT input buffer).
void preprocessGpu(const unsigned char* h_frame_data, int frame_w, int frame_h,
                   GpuPreprocessBuffers& buf, float* d_input, int target_size,
                   cudaStream_t stream);
