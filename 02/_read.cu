#include <stdlib.h>
#include <iostream>
int main() {
    int w, h;
    std::string path;
    std::cin >> path;
    FILE *fp = fopen(path.c_str(), "rb");
    fread(&w, sizeof(w), 1, fp);
    fread(&h, sizeof(h), 1, fp);
    std::cout << w << ' ' << h << '\n';
    uchar4 *arr = new uchar4[w * h];
    fread(arr, sizeof(uchar4), w * h, fp);
    for(int i = 0; i < h; ++i) {
        for(int j = 0; j < w; ++j) {
            auto p = arr[i*w + j];
            std::cout << p.x << ' ' << p.y << ' ' << p.z << ' ' << p.w;
        }
        std::cout << '\n';
    }
}