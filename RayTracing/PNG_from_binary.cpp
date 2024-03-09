#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "lib/stb_image_write.h"
#include <fstream>
#include <vector>
#include <iostream>

int main(int argc, char* argv[]) {
    std::string inpath = "out.bin";
    std::string outpath = "out.png";
    if(argc == 2) {
        inpath = argv[1];
    }
    if(argc > 2) {
        inpath = argv[1];
        outpath = argv[2];
    }
    std::ifstream fin(inpath, std::ios::binary);
    if (!fin) {
        std::cerr << "ERROR in reading file: " << inpath << std::endl;
        return 1;
    }

    int w, h;
    fin.read(reinterpret_cast<char*>(&w), sizeof(w));
    fin.read(reinterpret_cast<char*>(&h), sizeof(h));
    try {
        std::vector<unsigned char> buff(4 * w * h);
        fin.read(reinterpret_cast<char*>(buff.data()), buff.size());
        fin.close();
        stbi_write_png(outpath.c_str(), w, h, 4, buff.data(), 4 * w);
        std::cout << outpath << ": OK\n";
    } catch (...) {
        // std::cerr << "Vector error: " << e.what() << '\n';
        std::cout << "w = " << w << " h = " << h << '\n';
        std::cout << "file name : " << inpath << "\n";
        // Вы можете добавить здесь код для обработки ошибки, например, завершение программы или попытка освободить некоторые ресурсы
    }

    return 0;
}