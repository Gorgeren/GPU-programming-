#include "lib/matrix.hpp"
#include "lib/cpu_func.hpp"

int main() {
    int matrix_size;
    std::cin >> matrix_size;
    auto matrix = initialization<double>(matrix_size);
    read(matrix, matrix_size);
    printMatrix(matrix, matrix_size);
    auto res = inverse_matrix<double>(matrix, matrix_size);

    std::cout << "Inversed Matrix\n";
    printMatrix(res, matrix_size);

    std::cout << "Testing Identity\n";
    auto checkIdent = mulMatrix(matrix, res, matrix_size);
    printMatrix(checkIdent, matrix_size);

    deleter(checkIdent, matrix_size);
    deleter(matrix, matrix_size);
    deleter(res, matrix_size);
}