#ifndef TRIANGLE_CUH
#define TRIANGLE_CUH

#include "structs.cuh"
#include "texture.cuh"
#include "ray.cuh"
#include "functions.cuh"

struct trig {
    vec3 a;
    vec3 b;
    vec3 c;
    vec3 n;
    __host__ __device__ trig() = default;
    __host__ __device__ trig(const vec3& a, const vec3& b, const vec3& c)
        :   a(a), b(b), c(c) {
            n = vec3::prod(b - a, c - a);
            n.norm();
        }

    __host__ __device__ trig(const trig& tr) {
        a = tr.a;
        b = tr.b;
        c = tr.c;
        n = tr.n;
    }
    void bidirectional_n(const vec3& nor) {
        if(vec3::dot(n, nor) < 0) {
            std::swap(a, c);
            n = vec3::prod(b - a, c - a);
            n.norm();
        }
    }
    trig& operator+=(const vec3& vec) {
        a += vec;
        b += vec;
        c += vec;
        return *this;
    }
};
trig operator+(const trig& tr, const vec3 vec) {
    return {tr.a + vec, tr.b + vec, tr.c + vec};
}

extern int torches_count;
struct polygon {
    trig tr;
    vec3 color;
    vec3 e1, e2;
    double reflection, transparent;
    int light;
    bool texture;
    texture_t tex;
    vec3 b1, b2, b3;
    __host__ __device__ polygon(const trig& tr, const vec3& color, double r, double t,
                                int light = 0, bool istexture = 0,
                                texture_t tex = texture_t(), vec3 b1= vec3(),
                                vec3 b2 = vec3(), vec3 b3 = vec3())
                                : tr(tr), color(color), reflection(r), transparent(t),
                                  light(light) , texture(istexture),
                                  tex(tex) , b1(b1), b2(b2), b3(b3)
    {
        e1 = tr.b - tr.a;
        e2 = tr.c - tr.a;
    }
    __host__ __device__ double intersect(const ray &ray) const {
        vec3 P = vec3::prod(ray.dir, e2);
        double div = vec3::dot(P, e1);
        if (fabs(div) < EPS) {
            return -1;
        }
        vec3 T = ray.sp - tr.a;
        double u = vec3::dot(P, T) / div;
        if (u < 0.0 || u > 1.0) {
            return -1;
        }
        vec3 Q = vec3::prod(T, e1);
        double v = vec3::dot(Q, ray.dir) / div;
        if (v < 0.0 || u + v > 1.0) {
            return -1;
        }
        double res = vec3::dot(Q, e2) / div;
        return res < 0 ? -1 : res;
	}

    __host__ __device__ vec3 get_color(const ray& r, const vec3& hit) const {
        if (texture) {
            vec3 p = hit - b3;
            double y =
                (p.x * b1.y - p.y * b1.x) / (b2.x * b1.y - b2.y * b1.x);
            double x =
                (p.x * b2.y - p.y * b2.x) / (b1.x * b2.y - b1.y * b2.x);
            return tex.get_pixel(x, y);
        } else if (light > 0 && vec3::dot(tr.n, r.dir) > 0.0) {
            vec3 vl = (b2 - b1) / (light + 1);
            for (int i = 1; i <= light; ++i) {
                vec3 p_light = b1 + i * vl;
                if ((p_light - hit).len() < LIGHT_SIZE) {
                    return LIGHT;
                }
            }
        }
        return color;
    }
};

#define COEF_AMBIENT 0.25
#define COEF_SOURCE 1.0
#define K_D 1.0
#define K_S 0.5

__host__ __device__ vec3 phong_shading(const ray r, const vec3 hit,
                                        const polygon poly, int id,
                                        const torch* lights,
                                        int n_sources, const polygon* polygons,
                                        int n_polygons) {
    vec3 poly_color = poly.get_color(r, hit);

    double blend = 1.f;
    if(poly.light == 0) {
        blend = 1.f - poly.reflection - poly.transparent;
    }
    // blend = poly.reflection + poly.transparent;

    vec3 clr = COEF_AMBIENT * blend * r.coefs * poly_color;
    for (int j = 0; j < n_sources; ++j) {
        double len_max = (lights[j].p - hit).len();
        ray r_light(hit, lights[j].p - hit, r.id);
        vec3 coef_vis(1.0, 1.0, 1.0);
        for (int i = 0; i < n_polygons; ++i) {
            if (i == id) {
                continue;
            }
            double len = polygons[i].intersect(r_light);;
            if (len != -1 && len < len_max) {
                coef_vis *= polygons[i].transparent;
            }
        }
        vec3 clr_a = blend * r.coefs * coef_vis * lights[j].color * poly_color;
        double coef_diffusal = vec3::dot(poly.tr.n, r_light.dir);
        double coef_specular = 0.0;
        if (coef_diffusal < 0.0) {
            coef_diffusal = 0.0;
        } else {
            vec3 reflected = r_light.reflect(poly.tr.n);
            coef_specular = vec3::dot(reflected, r.dir);
            if (coef_specular < 0.0) {
                coef_specular = 0.0;
            } else {
                coef_specular = std::pow(coef_specular, 9);
            }
        }
        clr +=
            COEF_SOURCE * (K_D * coef_diffusal + K_S * coef_specular) * clr_a;
    }
    clr.x = (clr.x < 0) ? 0 : ((clr.x > 1) ? 1 : clr.x);
    clr.y = (clr.y < 0) ? 0 : ((clr.y > 1) ? 1 : clr.y);
    clr.z = (clr.z < 0) ? 0 : ((clr.z > 1) ? 1 : clr.z);
    return clr;
}
#endif