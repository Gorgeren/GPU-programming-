#include <stdio.h>
#include <stdlib.h>
#include <vector_types.h>
#include <iostream>

#define CSC(call)                                            \
do {                                                         \
    cudaError_t res = call;                                  \
    if(res != cudaSuccess) {                                 \
        fprintf(stderr, "ERROR in %s: %d. Message: %s\n",    \
                __FILE__, __LINE__, cudaGetErrorString(res));\
        exit(1);                                             \
    }                                                        \
} while(0);
__global__ void kernel(cudaTextureObject_t tex, uchar4 *out, int w, int h) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;
    int x, y;
    uchar4 p;
    for(y = idy; y < h; y += offsety) {
        for(x = idx; x < w; x += offsetx) {
            p = tex2D<uchar4>(tex, x, y);
            int Y = static_cast<int>(0.299 * p.x + 0.587 * p.y + 0.114 * p.z);
            Y = Y > 255 ? 255 : Y;
            out[y * w + x] = make_uchar4(Y, Y, Y, p.w);
        }
    }
}
int main() {
    int w, h;
    std::cout << "enter filename\n";
    std::string in;
    std::cin >> in;
    FILE *fp = fopen(in.c_str(), "rb");
    fread(&w, sizeof(w), 1, fp);
    fread(&h, sizeof(h), 1, fp);
    uchar4 *data = new uchar4[w * h];
    fread(data, sizeof(uchar4), w * h, fp);
    fclose(fp);
    cudaArray *arr;
    cudaChannelFormatDesc ch = cudaCreateChannelDesc<uchar4>();
    CSC(cudaMallocArray(&arr, &ch, w, h));
    CSC(cudaMemcpy2DToArray(arr, 0, 0, data, w * sizeof(uchar4), w * sizeof(uchar4), h, cudaMemcpyHostToDevice));
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

    uchar4 *dev_out;
    CSC(cudaMalloc(&dev_out, sizeof(uchar4)* w * h));
    kernel<<<dim3(16, 16), dim3(32, 32)>>> (tex, dev_out, w, h);
    CSC(cudaGetLastError());
    CSC(cudaMemcpy(data, dev_out, sizeof(uchar4) * w * h, cudaMemcpyDeviceToHost));

    CSC(cudaDestroyTextureObject(tex));
    CSC(cudaFreeArray(arr));
    CSC(cudaFree(dev_out));

    fp = fopen("out.data", "wb");
    fwrite(&w, sizeof(int), 1, fp);
    fwrite(&h, sizeof(int), 1, fp);
    fwrite(data, sizeof(uchar4), w * h, fp);
    fclose(fp);
    free(data);
    return 0;
}