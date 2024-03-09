#ifndef RAY_CUH
#define RAY_CUH
#include "polygon.cuh"
#include "functions.cuh"
#include "structs.cuh"
struct ray {
    vec3 sp; // source position
    vec3 dir; // direction 
    int id;
    vec3 coefs;

    __host__ __device__ ray() : dir(), id{}, coefs() {}
    __host__ __device__ ray(const vec3& sp, const vec3& dir, int id) 
                            : sp(sp), dir(dir), id(id), coefs{1,1,1} {
        this->dir.norm();
    }
    __host__ __device__ ray(const vec3& sp, const vec3& dir, int id,
                            const vec3& coefs)
                            : sp(sp), dir(dir), id(id), coefs{coefs} {
        this->dir.norm();
    }
    __host__ __device__ static ray* initrays(const vec3& pc, const vec3& pv, int w, int h, double angle) {
        ray* res = new ray[w * h];
        double dw = 2.0 / (w - 1.0);
        double dh = 2.0 / (h - 1.0);
        double z = 1.0 / std::tan(angle * M_PI / 360.0);
        vec3 bz = pv - pc;
        vec3 bx = vec3::prod(bz, vec3(0, 0, 1));
        vec3 by = vec3::prod(bx, bz);
        bx.norm();
        by.norm();
        bz.norm();
        for (int i = 0; i < w; i++) {
            for (int j = 0; j < h; j++) {
                vec3 v(-1.0 + dw * i, (-1.0 + dh * j) * h / w, z);
                vec3 dir = vec3::mult(bx, by, bz, v);
                int pix_id = (h - 1 - j) * w + i;
                res[i * h + j] = ray(pc, dir, pix_id);
            }
        }
        return res;
    }
    __host__ __device__ vec3 reflect(const vec3& n) const {
        vec3 r = dir - 2 * vec3::dot(n, dir) * n;
        r.norm();
        return r;
    }
};


#endif