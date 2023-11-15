#pragma once
#include "matrix.hpp"
template <typename T>
void swapLine(T *matrix, T *helper, int i, int size) {
    // find min
    T curr = matrix[i + size * i];
    T max = curr;
    int j = i;
    for(int idx = i; idx < size; ++idx) {
        if(std::abs(max) < std::abs(matrix[idx + size * i])) {
            max = std::abs(matrix[idx + size * i]);
            j = idx;
        }
    }
    if (i == j) return;
    for(int idx = 0; idx < size; ++idx) {
        std::swap(matrix[i + size * idx], matrix[j + size * idx]);
        std::swap(helper[i + size * idx], helper[j + size * idx]);
    }
    return;
}
template <typename T>
void subtractRow_straight(T* matrix, T* helper, int idx, int size) {
    T major;
    major = matrix[idx+ size * idx];
    for(int i = idx + 1; i < size; ++i) {
        T scale =  matrix[i+ size * idx] / major;
        for(int j = idx; j < size; ++j) {
            matrix[i+ size * j] -= scale * matrix[idx+ size * j];
        }
        for(int j = 0; j < size; ++j) {
            helper[i+ size * j] -= scale * helper[idx+ size * j];
        }
    }
}
template <typename T>
void subtractRow_reverse(T* matrix, T* helper, int idx, int size) {
    T major = matrix[idx + size * idx];
    for(int i = idx - 1; i > -1; --i) {
        T scale =  matrix[i + size * idx] / major;
        for(int j = idx; j > -1; --j) {
            matrix[i + size * j] -= scale * matrix[idx + size * j];
        }
        for(int j = 0; j < size; ++j) {
            helper[i + size * j] -= scale * helper[idx + size * j];
        }
    }
}
template <typename T>
void normalizate(T* matrix, T* helper, int size) {
    for(int i = 0; i < size; ++i) {
        T major = matrix[i + size * i];
        matrix[i + size * i] = 1;
        for(int j = 0; j < size; ++j) {
            helper[i + size * j] /= major;
        }
    }
}
template <typename T>
T* inverse_GaussMethod (T* matrix, int size) {
    // Чтобы ускорить можно использовать matrix
    // вместо copy
    auto copy = copyMatrix(matrix, size);
    auto helper = identity_matrix<T>(size);
    for(int i = 0; i < size; ++i) {
        swapLine(copy, helper, i, size);
        subtractRow_straight(copy, helper, i, size);
    }
    for(int i = size - 1; i > -1; --i) {
        subtractRow_reverse(copy, helper, i, size);
    }
    normalizate(copy, helper, size);
    deleter(copy);
    return helper;
}
template <typename T>
T* inverse_matrix(T* matrix, int size) {
    return inverse_GaussMethod (matrix, size);
}