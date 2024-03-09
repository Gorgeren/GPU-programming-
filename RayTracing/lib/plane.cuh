#pragma once 

#include "structs.cuh"
#include "polygon.cuh"

struct plane {
    float a, b, c, d;
    __host__ __device__ plane(const trig& tr) {
        vec3 v0 = tr.a;
        vec3 v1 = tr.b - v0;
        vec3 v2 = tr.c - v0;
        a = v1.y * v2.z - v1.z * v2.y;
        b = (-1.0) * (v1.x * v2.z - v1.z * v2.x);
        c = v1.x * v2.y - v1.y * v2.x;
        d = -v0.x * (v1.y * v2.z - v1.z * v2.y) + (v0.y) * (v1.x * v2.z - v1.z * v2.x) +(-v0.z) * (v1.x * v2.y - v1.y * v2.x);
    }
    __host__ __device__ static float distance(const vec3& from, vec3 dir, const trig& trig) {
        plane p(trig);
        dir.norm();
        return -(p.a * from.x + p.b * from.y + p.c * from.z + p.d) / (p.a * dir.x + p.b * dir.y + p.c * dir.z);
    }
};