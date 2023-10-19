
#include <iostream>

#define CSC(call) 							\
do { 										\
	cudaError_t status = call;				\
	if (status != cudaSuccess) {																				\
		fprintf(stderr, "ERROR is %s:%d. Message: %s\n", __FILE__, __LINE__, cudaGetErrorString(status));		\
		exit(0);								\
	}											\
} while(0)

template <typename T> 
__global__ void kernel(T *arr1, T *arr2, T *res, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = gridDim.x * blockDim.x;
    while(idx < size) {
        res[idx] = arr1[idx] + arr2[idx];
        idx += offset;
    }
}
// #define TEST
int main() {
    using date_type = double;
    std::ios::sync_with_stdio(false);
    int n;
    std::cin >> n;
    date_type *arr1 = new date_type[n];
    date_type *arr2 = new date_type[n];

    for(int i = 0; i < n; ++i) {
        date_type tmp;
        std::cin >> tmp;
        arr1[i] = tmp;
    }
    for(int i = 0; i < n; ++i) {
        date_type tmp;
        std::cin >> tmp;
        arr2[i] = tmp;
    }
    date_type *cuda_arr1;
    date_type *cuda_arr2;
    date_type *cuda_res;
    size_t countOfBytes = sizeof(arr1[0]) * n;
    CSC(cudaMalloc(&cuda_arr1, countOfBytes));
    CSC(cudaMalloc(&cuda_arr2, countOfBytes));
    CSC(cudaMalloc(&cuda_res, countOfBytes));

    CSC(cudaMemcpy(cuda_arr1, arr1, countOfBytes, cudaMemcpyHostToDevice));
    CSC(cudaMemcpy(cuda_arr2, arr2, countOfBytes, cudaMemcpyHostToDevice));
#ifdef TEST
    float min = 100;
    int min_i = 0, min_j = 0;
    for(int i = 1; i <= 1024; i *= 2) {
        for (int j = 32; j <= 1024; j *= 2) {
            cudaEvent_t start, stop;
            CSC(cudaEventCreate(&start));
            CSC(cudaEventCreate(&stop));
            CSC(cudaEventRecord(start));
            kernel<<<1024, 1024>>> (cuda_arr1, cuda_arr2, cuda_res, n);
            CSC(cudaEventRecord(stop));
            CSC(cudaEventSynchronize(stop));
            float t;
        	CSC(cudaEventElapsedTime(&t, start, stop));
            CSC(cudaEventDestroy(start));
	        CSC(cudaEventDestroy(stop));
            if(t < min) {
                min = std::min(t, min);
                min_i = i;
                min_j = j;
            }
            std::cout << "Grid =  "<< i << "Block size = " << j << " time = " << t << " ms\n";
        }
    }
    std::cout << "min is = " << min << "for Grid size = " << min_i << " Block size = " << min_j << '\n';
#else
    kernel<<<1024, 1024>>> (cuda_arr1, cuda_arr2, cuda_res, n);
#endif /* TEST */
    date_type *res = new date_type[n]{};
    CSC(cudaMemcpy(res, cuda_res, countOfBytes, cudaMemcpyDeviceToHost));
    std::cout.precision(6);
    for(int i = 0; i < n; ++i) {
        std::cout << std::fixed << res[i] << ' ';
    }
    std::cout << '\n';
    CSC(cudaFree(cuda_arr1));
    CSC(cudaFree(cuda_arr2));
    CSC(cudaFree(cuda_res));
    delete []arr1;
    delete []arr2;
    delete []res;
}