#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#define STB_IMAGE_IMPLEMENTATION
#include "lib/stb_image.h"

struct uchar4 {
    unsigned char x, y, z, w;
};
std::ostream& operator<<(std::ostream& out, const uchar4& ch) {
    out << (int)ch.x << " "<< (int)ch.y << " "<< (int)ch.z << " "<< (int)ch.w<< "\n";
    return out;
}
void write_binarypic(const std::string& res_file, int w, int h, uchar4* data) {
    std::ofstream out(res_file, std::ios::binary);
    if (!out) {
        std::cerr << "ERROR in opening " <<  res_file << " file " << std::endl;
        delete[] data;
        exit(1);
    }
    out.write(reinterpret_cast<const char*>(&w), sizeof(w));
    out.write(reinterpret_cast<const char*>(&h), sizeof(h));
    out.write(reinterpret_cast<const char*>(data), sizeof(uchar4) * w * h);
}

int find_last(const std::string& str) {
    int idx = -1;
    int j = 0;
    for(char i: str) {
        if(i == '/') idx = j;
        j++;
    }
    return idx;
}
int main(int argc, char* argv[]) {
    std::string infile = "";
    std::string outfile = "";
    if(argc == 1) {
        std::cout << "You have to give me the pic's path\n";
        exit(1);
    }
    if(argc == 2) {
        infile = argv[1];
        auto it = infile.find(".png");
        if(it != std::string::npos && infile.size() - 4 == it) {
            outfile = infile.substr(0, it) + ".bin";
        }
        else outfile = infile + ".bin";
    }
    if(argc == 3) {
        infile = argv[1];
        outfile = argv[2];
    }
    int w, h, channels;

    unsigned char* imageData = stbi_load(infile.c_str(), &w, &h, &channels, 4);
    if (imageData == nullptr) {
        std::cerr << "Failed to load image: " << infile << std::endl;
        return 1;
    }

    if (channels != 4) {
        // std::cerr << "Image does not have 4 channels." << std::endl;
        // std::cerr << ((uchar4*)imageData)[0] << '\n';
        // stbi_image_free(imageData);
        // return 1;
    }

    write_binarypic(outfile, w, h, (uchar4*)imageData);
    stbi_image_free(imageData);
    return 0;
}
