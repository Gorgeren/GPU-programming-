#include <iostream>
#include <time.h>
#include <string>
#include <stdlib.h>
#include <fstream>
int main(int args, char** argv) {
    std::ios::sync_with_stdio(0);
    std::cin.tie(0);
    std::cout.tie(0);
    std::ofstream out("in");
    long long n = 1e7;
    if(args == 2) {
        n = std::stoll(argv[1]);
    }
    out << n << '\n';
    for(int i = 0; i < n; ++i) {
        out << (i % 20)  + (3.0 / ((i % 113) + 3)) << ' ';
    }
    out << '\n';
    for(int i = 0; i < n; ++i) {
        out << 1 + (i % 27) + (7.0 / (i % 111 + 1)) << ' ';
    }
    out << '\n';
}