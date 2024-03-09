#ifndef STRUCTS_HPP
#define STRUCTS_HPP
#include <iostream>
#include <iomanip>

template <typename T>
struct vec3_t {
    T x;
    T y;
    T z;

    __host__ __device__ vec3_t() : x{0}, y{0}, z{0} {};
    __host__ __device__ vec3_t(const T& _x, const T& _y, const T& _z)
                            : x(_x), y(_y), z(_z){};
    __host__ __device__ vec3_t(const vec3_t& v) : x(v.x), y(v.y), z(v.z){};
    
    __host__ __device__ static vec3_t cyl(const T& _r, const T& _z,
                                            const T& _phi) {
        return {_r * std::cos(_phi), _r * std::sin(_phi), _z};
    }
    __host__ __device__ static vec3_t mult(const vec3_t &a, const vec3_t &b, const vec3_t &c, const vec3_t &v) {
    return {a.x * v.x + b.x * v.y + c.x * v.z,
            a.y * v.x + b.y * v.y + c.y * v.z,
            a.z * v.x + b.z * v.y + c.z * v.z};
    }
    __host__ __device__ static vec3_t norm(const vec3_t &v) {
        T l = v.len();
        return vec3_t{v.x / l, v.y / l, v.z / l};
    }
    __host__ __device__ static T dot(const vec3_t& a, const vec3_t& b) {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    }
    __host__ __device__ T len() const {
        return std::sqrt(dot(*this, *this));
    }
    __host__ __device__ vec3_t& norm() {
        T l = this->len();
        x /= l, y /= l, z /= l;
        return *this;
    }
    __host__ __device__ static vec3_t prod(const vec3_t &a, const vec3_t &b) {
        return vec3_t{a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
    }

    __host__ __device__ vec3_t& prod(const vec3_t &b) {
        T nx = y * b.z - z * b.y;
        T ny = z * b.x - x * b.z;
        T nz = x * b.y - y * b.x;
        x = nx;
        y = ny;
        z = nz;
        return *this;
    }
    __host__ __device__ static uchar4 to_uchar4(vec3_t a) {
        if(a.x > 1) a.x = 1;
        if(a.y > 1) a.y = 1;
        if(a.z > 1) a.z = 1;
        if(a.x < 0) a.x = 0;
        if(a.y < 0) a.y = 0;
        if(a.z < 0) a.z = 0;
        return  make_uchar4(static_cast<u_char>(a.x * 255),
                static_cast<u_char>(a.y * 255), 
                static_cast<u_char>(a.z * 255), 
                static_cast<u_char>(255));
    }

    __host__ __device__ vec3_t& operator+=(const vec3_t& v) {
        x += v.x, y += v.y, z += v.z;
        return *this;
    }

    __host__ __device__ vec3_t& operator-=(const vec3_t& v) {
        x -= v.x, y -= v.y, z -= v.z;
        return *this;
    }
    __host__ __device__ vec3_t& operator+=(const T& coeff) {
        x += coeff, y += coeff, z += coeff;
        return *this;
    }
    __host__ __device__ vec3_t& operator-=(const T& coeff) {
        x -= coeff, y -= coeff, z -= coeff;
        return *this;
    }
    __host__ __device__ vec3_t& operator/=(const T& coeff) {
        x /= coeff, y /= coeff, z /= coeff;
        return *this;
    }
    
    __host__ __device__ vec3_t& operator*=(const T& coeff) {
        x *= coeff, y *= coeff, z *= coeff;
        return *this;
    }
    __device__ static void atomic_add(vec3_t* a, const vec3_t b) {
        atomicAdd(&(a->x), b.x);
        atomicAdd(&(a->y), b.y);
        atomicAdd(&(a->z), b.z);
    }
    
};

template <typename T>
__host__ __device__ vec3_t<T> operator*(const vec3_t<T>& a, const vec3_t<T>& b) {
    return vec3_t<T>{a.x * b.x, a.y * b.y, a.z * b.z};
}
template <typename T>
__host__ __device__ vec3_t<T> operator*(float coeff, const vec3_t<T>& v) {
    return vec3_t<T>{coeff * v.x, coeff * v.y, coeff * v.z};
}
template <typename T>
__host__ __device__ vec3_t<T> operator/(const vec3_t<T>& v, float coeff) {
    return vec3_t<T>{v.x / coeff, v.y / coeff, v.z / coeff};
}
template <typename T>
__host__ __device__ vec3_t<T> operator+(const vec3_t<T>& a, const vec3_t<T>& b) {
    return vec3_t<T>{a.x + b.x, a.y + b.y, a.z + b.z};
}
template <typename T>
__host__ __device__ vec3_t<T> operator-(const vec3_t<T>& a, const vec3_t<T>& b) {
    return vec3_t<T>{a.x - b.x, a.y - b.y, a.z - b.z};
}

template <typename T>
std::ostream& operator<<(std::ostream& out, const vec3_t<T> &v) {
    out << std::fixed << v.x << ' ' << v.y << ' ' << v.z << '\n';
    return out;
}

template <typename T>
std::istream& operator>>(std::istream& in, vec3_t<T>& v) {
    in >> v.x >> v.y >> v.z;
    return in;
}

using vec3 = vec3_t<float>;
using vec3i = vec3_t<int>;

struct camera {
	vec3 pos;
	float angle;
	vec3 i, j, k;
    __host__ __device__ camera(const vec3& pos, float angle)
        : pos(pos), angle(angle) {}

	void dir(const vec3 &target) {
		vec3 up = {0, 1, 0};
		k = (target - pos).norm();
		i = up.prod(k).norm();
		j = k.prod(i);
	}
};
struct torch {
    vec3 p;
    vec3 color;

    friend std::istream& operator>>(std::istream& in, torch& source) {
        in >> source.p >> source.color;
        return in;
    }
};

#endif /*STRUCTS_HPP*/