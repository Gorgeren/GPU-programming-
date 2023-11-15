#include <iostream>
#include "lib/matrix.hpp"
int main() {
    int matrix_size;
    std::cin >> matrix_size;
    auto matrix = initialization<double>(matrix_size);
    read(matrix, matrix_size);
    deleter(matrix, matrix_size);
}