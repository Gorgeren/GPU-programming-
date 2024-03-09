#pragma once

#include <string>
#include "functions.cuh"

#define CSC(call)                                                      \
do {                                                                   \
    cudaError_t status = call;                                         \
    if (status != cudaSuccess) {                                       \
        fprintf(stderr, "ERROR[%s:%d]: %s\n", __FILE__, __LINE__,      \
                cudaGetErrorString(status));                            \
        exit(0);                                                       \
    }                                                                  \
} while(0)

namespace texture_help {
void read_binarypic(const std::string& path, int& w, int&h, uchar4*& data) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        fprintf(stderr, "ERROR in opening %s file", path.c_str());
        exit(1);
    }
    in.read(reinterpret_cast<char*>(&w), sizeof(w));
    in.read(reinterpret_cast<char*>(&h), sizeof(h));
    data = new uchar4[w * h];
    in.read(reinterpret_cast<char*>(data), sizeof(uchar4) * w * h);
}
};
struct texture_t {
    int w, h;
    uchar4 *cpu_data, *gpu_data;
    bool gpu;
    __host__ __device__ texture_t() 
    :   w(0), h(0), cpu_data(nullptr), gpu_data(nullptr), gpu(0) {}

    __host__ __device__ texture_t(const texture_t& tex) {
        w = tex.w;
        h = tex.h;
        cpu_data = tex.cpu_data;
        gpu_data = tex.gpu_data;
        gpu = tex.gpu;
    }        
    
    void read_texture(const std::string& path) {
        texture_help::read_binarypic(path, w, h, cpu_data);
        if(gpu) {
            CSC(cudaMalloc(&gpu_data, sizeof(uchar4) * w * h));
            CSC(cudaMemcpy(gpu_data, cpu_data, sizeof(uchar4) * w * h, cudaMemcpyHostToDevice));
        }
    }

    __host__ __device__ vec3 get_pixel(float x, float y) const {
        int u = max(0, min((int)(x * w), w - 1));
		int v = max(0, min((int)(y * h), h - 1));
        int idx = v * w + u;
        uchar4 pix = gpu ? gpu_data[idx] : cpu_data[idx];
        return vec3{static_cast<float>(pix.x), static_cast<float>(pix.y), static_cast<float>(pix.y)} / 255;
    }

};