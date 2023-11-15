#pragma once
#include <iostream>
#include <iomanip>

template <typename T>
T *initialization(int matrix_size) {
    T *matrix = new T[matrix_size * matrix_size]{};
    return matrix;
}
template <typename T>
T *identity_matrix(int size) {
    auto matrix = initialization<T>(size);
    for(int i = 0; i < size; ++i) {
        matrix[i * size + i] = 1;
    }
    return matrix;
}
template <typename T>
void deleter(T *matrix) {
    delete[] matrix;
}
template <typename T>
// Храним элементы матрицы по колонкам
void read(T *matrix, int size) {
    for(int i = 0; i < size; ++i) {
        for(int j = 0; j < size; ++j) {
            std::cin >> matrix[i + j * size];
        }
    }
}

template <typename T>
void printMatrix(const T* matrix, int size) {
    int chars = 3;
    for(int i = 0; i < size; ++i) {
        for(int j = 0; j < size; ++j) {
            std::cout <<std::fixed<<std::setprecision(chars)<< std::setw(chars + 4) <<matrix[i + size * j] << ' ';
        }
        std::cout << '\n';
    }
    std::cout << std::endl;
}
template <typename T>
void printTwoMatrix(T*  matrix,  T*  helper, int size) {
    int chars = 4;
    for(int i = 0; i < size; ++i) {
        for(int j = 0; j < size; ++j) {
            std::cout <<std::fixed<< std::setprecision(chars) <<  std::setw(chars + 4) <<matrix[i + j * size] << ' ';
        }
        std::cout << std::setw(chars) << "| " ;
        for(int j = 0; j < size; ++j) {
            std::cout  <<std::fixed<<std::setprecision(chars)<< std::setw(chars + 4) << helper[i + size * j] << ' ';
        }
        std::cout << '\n';
    }
    std::cout << std::endl;
}
template <typename T>
T* mulMatrix(T* first, T* second, int size) {
    auto res = initialization<T>(size);
    for(int i = 0; i < size; ++i) {
        for(int j = 0; j < size; ++j) {
            for(int k = 0; k < size; ++k) {
                res[i + size * j] += first[i + size * k] * second[k + size * j];
            }
        }
    }
    return res;
}
template <typename T> 
T* copyMatrix(T* src, int size) {
    auto res = initialization<T>(size);
    for(int i = 0; i < size; ++i) {
        for(int j = 0; j < size; ++j) {
            res[i + size * j] = src[i + size * j];
        }
    }
    return res;
}

