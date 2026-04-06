#include "preprocessing_gpu.h"
#include <cstdio>
#include <algorithm>

// Single kernel: letterbox resize + BGR→RGB + normalize [0,1] + HWC→CHW
// Each thread handles one (out_y, out_x) pixel, writing 3 channel values.
__global__ void letterboxPreprocessKernel(
    const unsigned char* __restrict__ src, int src_w, int src_h,
    float* __restrict__ dst, int target_size,
    float scale, int pad_left, int pad_top, int new_w, int new_h)
{
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (out_x >= target_size || out_y >= target_size) return;

    const float pad_val = 114.0f / 255.0f;
    int plane_size = target_size * target_size;

    // Check if this pixel is in the padded border
    int rx = out_x - pad_left;
    int ry = out_y - pad_top;

    if (rx < 0 || rx >= new_w || ry < 0 || ry >= new_h) {
        // Padding region
        dst[0 * plane_size + out_y * target_size + out_x] = pad_val;
        dst[1 * plane_size + out_y * target_size + out_x] = pad_val;
        dst[2 * plane_size + out_y * target_size + out_x] = pad_val;
        return;
    }

    // Map back to source coordinates (bilinear interpolation)
    float src_xf = rx / scale;
    float src_yf = ry / scale;

    int x0 = (int)src_xf;
    int y0 = (int)src_yf;
    int x1 = min(x0 + 1, src_w - 1);
    int y1 = min(y0 + 1, src_h - 1);
    x0 = min(x0, src_w - 1);
    y0 = min(y0, src_h - 1);

    float xd = src_xf - x0;
    float yd = src_yf - y0;

    // Read 4 BGR pixels for bilinear interp
    const unsigned char* p00 = src + (y0 * src_w + x0) * 3;
    const unsigned char* p01 = src + (y0 * src_w + x1) * 3;
    const unsigned char* p10 = src + (y1 * src_w + x0) * 3;
    const unsigned char* p11 = src + (y1 * src_w + x1) * 3;

    // Bilinear interpolation for each BGR channel, convert to RGB, normalize
    #pragma unroll
    for (int c = 0; c < 3; ++c) {
        // Source is BGR, we want RGB: channel 0→2(B), 1→1(G), 2→0(R)
        int src_c = 2 - c;  // RGB channel c maps to BGR channel (2-c)

        float v00 = p00[src_c];
        float v01 = p01[src_c];
        float v10 = p10[src_c];
        float v11 = p11[src_c];

        float val = v00 * (1.0f - xd) * (1.0f - yd)
                  + v01 * xd * (1.0f - yd)
                  + v10 * (1.0f - xd) * yd
                  + v11 * xd * yd;

        dst[c * plane_size + out_y * target_size + out_x] = val / 255.0f;
    }
}

void initGpuPreprocessBuffers(GpuPreprocessBuffers& buf, int frame_w, int frame_h)
{
    buf.alloc_w = frame_w;
    buf.alloc_h = frame_h;
    buf.frame_bytes = (size_t)frame_w * frame_h * 3;

    cudaMalloc(&buf.d_frame, buf.frame_bytes);
    cudaMallocHost(&buf.h_frame, buf.frame_bytes);  // pinned host memory
}

void freeGpuPreprocessBuffers(GpuPreprocessBuffers& buf)
{
    if (buf.d_frame) { cudaFree(buf.d_frame); buf.d_frame = nullptr; }
    if (buf.h_frame) { cudaFreeHost(buf.h_frame); buf.h_frame = nullptr; }
}

void preprocessGpu(const unsigned char* h_frame_data, int frame_w, int frame_h,
                   GpuPreprocessBuffers& buf, float* d_input, int target_size,
                   cudaStream_t stream)
{
    size_t bytes = (size_t)frame_w * frame_h * 3;

    // Copy raw BGR frame into pinned memory, then H2D async
    memcpy(buf.h_frame, h_frame_data, bytes);
    cudaMemcpyAsync(buf.d_frame, buf.h_frame, bytes, cudaMemcpyHostToDevice, stream);

    // Compute letterbox parameters
    float scale_w = (float)target_size / frame_w;
    float scale_h = (float)target_size / frame_h;
    float scale = (scale_w < scale_h) ? scale_w : scale_h;

    int new_w = (int)(frame_w * scale);
    int new_h = (int)(frame_h * scale);
    int pad_left = (target_size - new_w) / 2;
    int pad_top  = (target_size - new_h) / 2;

    // Launch kernel: one thread per output pixel
    dim3 block(32, 32);
    dim3 grid((target_size + block.x - 1) / block.x,
              (target_size + block.y - 1) / block.y);

    letterboxPreprocessKernel<<<grid, block, 0, stream>>>(
        buf.d_frame, frame_w, frame_h,
        d_input, target_size,
        scale, pad_left, pad_top, new_w, new_h);
}
