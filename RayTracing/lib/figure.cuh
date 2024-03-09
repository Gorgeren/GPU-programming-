#ifndef FIGURE_CUH
#define FIGURE_CUH
#include <vector>
#include <iostream>

#include "structs.cuh"

struct figure_t {
    vec3 center;
    float r;
    vec3 color;
    float reflection;
    float transparency;
    int lights;
    figure_t() = default;
    figure_t(const vec3& center, const float& r, const vec3& color
            , float reflection, float transparency, int lights)
            : center(center)
            , r(r)
            , color(color)
            , transparency(transparency), lights(lights) {};
    friend std::istream& operator>>(std::istream&, figure_t&);
};

std::istream& operator>>(std::istream& in, figure_t& fig) {
    in >> fig.center;
    in >> fig.color;
    in >> fig.r;
    in >>fig.reflection;
    in >> fig.transparency;
    in >> fig.lights;
    return in;
}
#endif /* FIGURE_CUH */