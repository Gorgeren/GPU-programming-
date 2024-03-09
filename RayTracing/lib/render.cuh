#pragma once

#include <iomanip>
#include "functions.cuh"
#define MAGIC_CONSTANT 1e-3

__global__ void gpuSSAA(vec3 *source, uchar4 *dst, int width, int height, int factor) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	int offsetx = blockDim.x * gridDim.x;
	int offsety = blockDim.y * gridDim.y;
	float factor2 = factor * factor;
	for (int j = idy; j < height; j += offsety) {
		for (int i = idx; i < width; i += offsetx) {
			vec3 color{};
			for (int v = 0; v < factor; ++v) {
				for (int u = 0; u < factor; ++u) {
					color += source[(j * factor + v) * width * factor + i * factor + u];
				}
			}
			color /= factor2;
			dst[j * width + i] = vec3::to_uchar4(color);
		}
	}
}

void cpuSSAA(vec3 *source, uchar4 *dst, int width, int height, int factor) {
	float factor2 = factor * factor;
	for (int j = 0; j < height; ++j) {
		for (int i = 0; i < width; ++i) {
			vec3 color{};
			for (int v = 0; v < factor; ++v) {
				for (int u = 0; u < factor; ++u) {
					color += source[(j * factor + v) * width * factor + i * factor + u];
				}
			}
			color /= factor2;
			dst[j * width + i] = vec3::to_uchar4(color);
		}
	}
}

void trace(const ray* rays_in, int invec_size, ray* rays_out, int& outvec_size,
           vec3* data) {
    for (int k = 0; k < invec_size; ++k) {
        int n = polygons.size();
        int min_i = n;
        float min_len = INF;
        for (int i = 0; i < n; i++) {
            float len = polygons[i].intersect(rays_in[k]);
            if (len != -1 && len < min_len) {
                min_i = i;
                min_len = len;
            }
        }
        if (min_i == n) {
            continue;
        }
        vec3 hit = rays_in[k].sp + min_len * rays_in[k].dir;
        data[rays_in[k].id] +=
            phong_shading(rays_in[k], hit, polygons[min_i], min_i, torches.data(),
                          torches_count, polygons.data(), n);
        if (polygons[min_i].transparent > 0) {
            rays_out[outvec_size++] =
                ray(hit + MAGIC_CONSTANT * rays_in[k].dir, rays_in[k].dir, rays_in[k].id,
                    polygons[min_i].transparent * rays_in[k].coefs *
                        polygons[min_i].get_color(rays_in[k], hit));
        }
        if (polygons[min_i].reflection > 0) {
            vec3 reflected = rays_in[k].reflect(polygons[min_i].tr.n);
            rays_out[outvec_size++] =
                ray(hit + MAGIC_CONSTANT * reflected, reflected, rays_in[k].id,
                    polygons[min_i].reflection * rays_in[k].coefs *
                        polygons[min_i].get_color(rays_in[k], hit));
        }
    }
}
/**
 * @brief Renders a frame of a 3D scene on CPU.
 *
 * This function generates a frame of a 3D scene based on the camera's position,
 * direction, frame resolution, field of view angle, and writes the result to
 * the provided data buffer.
 *
 * @param pc The position of the camera in the 3D scene.
 * @param pv The direction the camera is facing in the 3D scene.
 * @param w The width of the frame in pixels.
 * @param h The height of the frame in pixels.
 * @param angle The field of view angle of the camera, in degrees.
 * @param data A pointer to the buffer where the rendered frame will be stored.
 */
void render_cpu(const vec3& pc, const vec3& pv, int w, int h,
                float angle, uchar4* data) {
    static int frame_id = 0;
    int invec_size = w * h * ssaa * ssaa;
    vec3* res = new vec3[invec_size]{};
    ray* ray_in = ray::initrays(pc, pv, w, h, angle);
    long long total_rays = 0;
	auto start = std::chrono::high_resolution_clock::now();
    for (int rec = 0; rec < n_rec && invec_size > 0; ++rec) {
        total_rays += invec_size;
        ray* ray_out = new ray[2 * invec_size];
        int outvec_size = 0;
        trace(ray_in, invec_size, ray_out, outvec_size, res);
        delete[] ray_in;
        ray_in = ray_out;
        invec_size = outvec_size;
    }
    auto end = std::chrono::high_resolution_clock::now();
	auto time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    cpuSSAA(res, data, w, h, ssaa);
    delete[] ray_in;
    delete[] res;
    std::cout << ++frame_id << " "<< std::setw(4) << " " << total_rays << " " << std::setw(4) << " "<< time << '\n';
}

#define CSC(call)                                                      \
do {                                                                   \
    cudaError_t status = call;                                         \
    if (status != cudaSuccess) {                                       \
        fprintf(stderr, "ERROR[%s:%d]: %s\n", __FILE__, __LINE__,      \
                cudaGetErrorString(status));                            \
        exit(0);                                                       \
    }                                                                  \
} while(0)

torch* dev_torches;
polygon* dev_polygons;


void destroy_gpu_variables() {
    CSC(cudaFree(dev_torches));
    CSC(cudaFree(dev_polygons));
}

__global__ void initialize_data(vec3* dev_data, const int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int offset = gridDim.x * blockDim.x;
    for (int i = idx; i < n; i += offset) {
        dev_data[i] = vec3(0, 0, 0);
    }
}

/**
 * @brief Renders a frame of a 3D scene on GPU.
 *
 * This function generates a frame of a 3D scene based on the camera's position,
 * direction, frame resolution, field of view angle, and writes the result to
 * the provided data buffer.
 *
 * @param pc The position of the camera in the 3D scene.
 * @param pv The direction the camera is facing in the 3D scene.
 * @param w The width of the frame in pixels.
 * @param h The height of the frame in pixels.
 * @param angle The field of view angle of the camera, in degrees.
 * @param data A pointer to the buffer where the rendered frame will be stored.
 */
__global__ void trace_gpu(const ray* rays_in, const int input_size, ray* rays_out,
                          int* out_size, vec3* dev_data,
                          const torch* dev_torhces, int torches_count,
                          const polygon* dev_polygons, int n_polygons) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int offset = gridDim.x * blockDim.x;
    for (int k = idx; k < input_size; k += offset) {
        int idx = n_polygons;
        float min_len = INF;
        for (int i = 0; i < n_polygons; i++) {
            float len = dev_polygons[i].intersect(rays_in[k]);
            if (len > -1 + EPS && len < min_len) {
                idx = i;
                min_len = len;
            }
        }
        if (idx == n_polygons) {
            continue;
        }
        vec3 hit = rays_in[k].sp + min_len * rays_in[k].dir;
        vec3 poly_color = dev_polygons[idx].get_color(rays_in[k], hit);
        vec3::atomic_add(
            &dev_data[rays_in[k].id],
            phong_shading(rays_in[k], hit, dev_polygons[idx], idx, dev_torhces,
                          torches_count, dev_polygons, n_polygons));
        if (dev_polygons[idx].transparent > 0) {
            rays_out[atomicAdd(out_size, 1)] = 
                ray(hit + MAGIC_CONSTANT * rays_in[k].dir, rays_in[k].dir, rays_in[k].id,
                    dev_polygons[idx].transparent * rays_in[k].coefs *
                        poly_color);
        }
        if (dev_polygons[idx].reflection > 0) {
            vec3 reflected =
                rays_in[k].reflect(dev_polygons[idx].tr.n);
            rays_out[atomicAdd(out_size, 1)] =
                ray(hit + MAGIC_CONSTANT * reflected, reflected, rays_in[k].id,
                    dev_polygons[idx].reflection * rays_in[k].coefs *
                        poly_color);
        }
    }
}

__global__ void init_rays_gpu(const vec3 pc, const vec3 pv, int w, int h,
                              float angle, ray* dev_rays) {
    float dw = 2.0 / (w - 1.0);
    float dh = 2.0 / (h - 1.0);
    float z = 1.0 / std::tan(angle * M_PI / 360.0);
    vec3 bz = pv - pc;
    vec3 bx = vec3::prod(bz, vec3(0, 0, 1));
    vec3 by = vec3::prod(bx, bz);
    bx.norm();
    by.norm();
    bz.norm();

    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int idy = blockDim.y * blockIdx.y + threadIdx.y;
    const int offsetx = blockDim.x * gridDim.x;
    const int offsety = blockDim.y * gridDim.y;

    for (int i = idx; i < w; i += offsetx) {
        for (int j = idy; j < h; j += offsety) {
            vec3 v(-1.0 + dw * i, (-1.0 + dh * j) * h / w, z);
            vec3 dir = vec3::mult(bx, by, bz, v);
            int pix_id = (h - 1 - j) * w + i;
            dev_rays[i * h + j] = ray(pc, dir, pix_id);
        }
    }
}

// __global__ void write_data(uchar4* dev_data, vec3* dev_data_vec,
//                                int sz) {
//     const int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     const int offset = gridDim.x * blockDim.x;
//     for (int i = idx; i < sz; i += offset) {
//         dev_data[i] = vec3::to_uchar4(dev_data_vec[i]);
//     }
// }

void render_gpu(const vec3 pc, const vec3 pv, int w, int h,
                float angle, uchar4* dev_data) {
    CSC(cudaMalloc(&dev_torches, sizeof(torch) * torches_count));
    CSC(cudaMemcpy(dev_torches, torches.data(),
                   sizeof(torch) * torches_count, cudaMemcpyHostToDevice));
    CSC(cudaMalloc(&dev_polygons, sizeof(polygon) * polygons.size()));
    CSC(cudaMemcpy(dev_polygons, polygons.data(), sizeof(polygon) * polygons.size(),
                   cudaMemcpyHostToDevice));
    static int frame = 0;
    int input_size = w * h * ssaa * ssaa;
    vec3* indev_vec3;
    CSC(cudaMalloc(&indev_vec3, sizeof(vec3) * input_size));
    initialize_data<<<256, 256>>>(indev_vec3, input_size);
    ray* dev_ray_in;
    CSC(cudaMalloc(&dev_ray_in, sizeof(ray) * input_size));
    init_rays_gpu<<<dim3(64, 64), dim3(1,32)>>>(pc, pv, w * ssaa, h * ssaa, angle, dev_ray_in);
    CSC(cudaDeviceSynchronize());
    CSC(cudaGetLastError());
    const int ZERO = 0;
    long long rays_count = 0;
    int n = polygons.size();
	cudaEvent_t start;
    CSC(cudaEventCreate(&start));
	cudaEvent_t end;
	CSC(cudaEventCreate(&end));
	CSC(cudaEventRecord(start));
	CSC(cudaEventRecord(start));
    for (int rec = 0; rec < n_rec && input_size; ++rec) {
        rays_count += input_size;
        ray* dev_ray_out;
        CSC(cudaMalloc(&dev_ray_out, 2 * sizeof(ray) * input_size));
        int* out_size;
        CSC(cudaMalloc(&out_size, sizeof(int)));
        CSC(cudaGetLastError());
        CSC(cudaMemcpy(out_size, (void*)&ZERO, sizeof(int), cudaMemcpyHostToDevice));
        trace_gpu<<<256, 256>>>(dev_ray_in, input_size, dev_ray_out, out_size,
                                       indev_vec3, dev_torches, torches_count,
                                       dev_polygons, n);
        CSC(cudaGetLastError());
        CSC(cudaFree(dev_ray_in));
        dev_ray_in = dev_ray_out;
        CSC(cudaMemcpy(&input_size, out_size, sizeof(int), cudaMemcpyDeviceToHost));
        CSC(cudaFree(out_size));
        CSC(cudaGetLastError());
    }
    CSC(cudaEventRecord(end));
	CSC(cudaEventSynchronize(end));
    float time;
	CSC(cudaEventElapsedTime(&time, start, end));
    gpuSSAA<<<dim3(64, 64), dim3(1, 32)>>>(indev_vec3, dev_data, w, h, ssaa);
    CSC(cudaGetLastError());
	CSC(cudaDeviceSynchronize());
    CSC(cudaFree(dev_ray_in));
    CSC(cudaFree(indev_vec3));
    CSC(cudaGetLastError());
    CSC(cudaFree(dev_torches));
    CSC(cudaFree(dev_polygons));

    printf("%d\t%f\t%lli\n", ++frame, time, rays_count);
}

