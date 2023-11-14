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

__global__ void kernel(cudaTextureObject_t tex, Im out, int weight, int hight) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;
    int x, y;
    int scaleW = weight / out.w;
    int scaleH = hight / out.h;
    int r, g, b, a;
    uchar4 p;

    for(y = idy * scaleH ; y < hight; y += offsety) {
        for(x = idx * scaleW; x < weight; x += offsetx) {
            p = tex2D<uchar4>(tex, x, y);
            out.arr[(y/scaleH) * out.w + (x/scaleW)] = make_uchar4(p.x, p.y, p.z, 255);
        }
    }
}

void read_img(Im& img, const std::string& path) {
    FILE *fp = fopen(path.c_str(), "rb");
    fread(&img.w, sizeof(img.w), 1, fp);
    fread(&img.h, sizeof(img.h), 1, fp);
    img.arr = new uchar4[img.w * img.h];
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
// #define cheker
int main() {
    Im data;
    std::string in = "in.data";
#ifdef cheker
    std::cout << "enter filename\n";
    std::cin >> in;
#endif
    read_img(data, in);
    std::cout << data.w << ' ' << data.h << ' ' << sizeof(data.w) << '\n';
    cudaArray *arr;
    cudaChannelFormatDesc ch = cudaCreateChannelDesc<uchar4>();
    CSC(cudaMallocArray(&arr, &ch, data.w, data.h));
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
#elif
    std::cin >> dev_out.w >> dev_out.h;
#endif
    std::cout << dev_out.w << ' ' << dev_out.h << '\n';
    CSC(cudaMalloc(&dev_out.arr, sizeof(uchar4)* dev_out.w * dev_out.h));
    CSC(cudaGetLastError());

    kernel<<<dim3(32, 32), dim3(32, 32)>>> (tex, dev_out, data.w, data.h);
    CSC(cudaDeviceSynchronize());
    CSC(cudaGetLastError());
    delete[] data.arr;
    data.w = dev_out.w;
    data.h = dev_out.h;
    data.arr = new uchar4[data.w * data.h];
    CSC(cudaMemcpy(data.arr, dev_out.arr, sizeof(uchar4) * dev_out.w * dev_out.h, cudaMemcpyDeviceToHost));
    CSC(cudaDestroyTextureObject(tex));
    CSC(cudaFreeArray(arr));
    CSC(cudaFree(dev_out.arr));

    write_img(data, "out.data");
    free(data.arr);
    return 0;
}