#pragma once
#include "matrix.hpp"
template <typename T>
void swapLine(T **matrix, T **helper, int i, int size) {
    // find min
    T curr = matrix[i][i];
    T max = curr;
    int j = i;
    for(int idx = i; idx < size; ++idx) {
        if(std::abs(max) < std::abs(matrix[idx][i])) {
            max = std::abs(matrix[idx][i]);
            j = idx;
        }
    }
    if (i == j) return;
    for(int idx = 0; idx < size; ++idx) {
        std::swap(matrix[i][idx], matrix[j][idx]);
        std::swap(helper[i][idx], helper[j][idx]);
    }
    return;
}
template <typename T>
void subtractRow_straight(T **matrix, T **helper, int idx, int size) {
    T major;
    major = matrix[idx][idx];
    for(int i = idx + 1; i < size; ++i) {
        T scale =  matrix[i][idx] / major;
        for(int j = idx; j < size; ++j) {
            matrix[i][j] -= scale * matrix[idx][j];
        }
        for(int j = 0; j < size; ++j) {
            helper[i][j] -= scale * helper[idx][j];
        }
    }
}
template <typename T>
void subtractRow_reverse(T **matrix, T **helper, int idx, int size) {
    T major = matrix[idx][idx];
    for(int i = idx - 1; i > -1; --i) {
        T scale =  matrix[i][idx] / major;
        for(int j = idx; j > -1; --j) {
            matrix[i][j] -= scale * matrix[idx][j];
        }
        for(int j = 0; j < size; ++j) {
            helper[i][j] -= scale * helper[idx][j];
        }
    }
}
template <typename T>
void normalizate(T** matrix, T** helper, int size) {
    for(int i = 0; i < size; ++i) {
        T major = matrix[i][i];
        matrix[i][i] = 1;
        for(int j = 0; j < size; ++j) {
            helper[i][j] /= major;
        }
    }
}
template <typename T>
T** inverse_GaussMethod (T **matrix, int size) {
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
    deleter(copy, size);
    return helper;
}
template <typename T>
T** inverse_matrix(T** matrix, int size) {
    return inverse_GaussMethod (matrix, size);
}