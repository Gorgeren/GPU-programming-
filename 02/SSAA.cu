#include <stdio.h>
#include <stdlib.h>
#include <vector_types.h>
#include <iostream>

struct Im {
    int w, h;
    uchar4 *arr;
};
#define CSC(call)                                            \
do {                                                         \
    cudaError_t res = call;                                  \
    if(res != cudaSuccess) {                                 \
        fprintf(stderr, "ERROR in %s: %d. Message: %s\n",    \
                __FILE__, __LINE__, cudaGetErrorString(res));\
        exit(1);                                             \
    }                                                        \
} while(0);

__global__ void kernel(cudaTextureObject_t tex, uchar4 *out, int weight, int hight, int w, int h) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;
    int x, y;
    int scaleW = weight / w;
    int scaleH = hight / h;
    int r, g, b, a;
    uchar4 p;
    for(y = idy * scaleH ; y < hight; y += offsety) {
        for(x = idx * scaleW; x < weight; x += offsetx) {
            int count = 0;
            r = 0, g = 0, b = 0, a = 0;
            for(int i = x; i < x + scaleW; ++i) {
                for(int j = y; j < y + scaleH; ++j) {
                    p = tex2D<uchar4>(tex, i, j);
                    r += p.x;
                    g += p.y;
                    b += p.z;
                    a += p.w;
                    count++;
                }
            }
            r /= count;
            g /= count;
            b /= count;
            a /= count;
            if((y/scaleH) * w + (x/scaleW) >= w * h) continue;
            out[(y/scaleH) * w + (x/scaleW)] = make_uchar4(r, g, b, a);
        }
    }
}

void read_img(Im& img, const std::string& path) {
    FILE *fp = fopen(path.c_str(), "rb");
    fread(&img.w, sizeof(img.w), 1, fp);
    fread(&img.h, sizeof(img.h), 1, fp);
    img.arr =(uchar4 *) malloc(img.w * img.h * sizeof(uchar4));
    fread(img.arr, sizeof(uchar4), img.w * img.h, fp);
    fclose(fp);
}
void write_img(Im& img, const std::string& path) {
    FILE *fp = fopen(path.c_str(), "wb");
    fwrite(&img.w, sizeof(int), 1, fp);
    fwrite(&img.h, sizeof(int), 1, fp);
    fwrite(img.arr, sizeof(uchar4), img.w * img.h, fp);
    fclose(fp);
}
// #define checker
int main() {
    Im data;
    std::string in = "in.data";
    std::string out = "out.data";
#ifdef checker
    std::cin >> in;
    std::cin >> out;
    read_img(data, in);
#else
   read_img(data, in);
   std::cout << data.w << ' ' << data.h << ' ' << '\n';
#endif
    cudaArray *arr;
    cudaChannelFormatDesc ch = cudaCreateChannelDesc<uchar4>();
    // std::cout << "w ==" << data.w << ' ' << "h == " << data.h << '\n';
    CSC(cudaMallocArray(&arr, &ch, data.w, data.h));
    // std::cin >> in;
    CSC(cudaMemcpy2DToArray(arr, 0, 0, data.arr, data.w * sizeof(uchar4), data.w * sizeof(uchar4), data.h, cudaMemcpyHostToDevice));
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = arr;

    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = false;

    cudaTextureObject_t tex = 0;
    CSC(cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL));
    Im dev_out;
#ifndef checker
    std::cout << "scale\n";
    int scale;
    std::cin >> scale;
    dev_out.w = data.w / scale;
    dev_out.h = data.h / scale;
    std::cout << dev_out.w << ' ' << dev_out.h << '\n';
#else
    std::cin >> dev_out.w >> dev_out.h;
#endif
    CSC(cudaMalloc(&dev_out.arr, sizeof(uchar4)* dev_out.w * dev_out.h));
    CSC(cudaGetLastError());
    kernel<<<dim3(32, 32), dim3(32, 32)>>> (tex, dev_out.arr, data.w, data.h, dev_out.w, dev_out.h);
    CSC(cudaDeviceSynchronize());
    CSC(cudaGetLastError());
    free(data.arr);
    data.w = dev_out.w;
    data.h = dev_out.h;
    data.arr = (uchar4 *)malloc(data.w * data.h * sizeof(uchar4));
    CSC(cudaMemcpy(data.arr, dev_out.arr, sizeof(uchar4) * dev_out.w * dev_out.h, cudaMemcpyDeviceToHost));
    CSC(cudaDestroyTextureObject(tex));
    CSC(cudaFreeArray(arr));
    CSC(cudaFree(dev_out.arr));
    write_img(data, out);
    free(data.arr);
    return 0;
}