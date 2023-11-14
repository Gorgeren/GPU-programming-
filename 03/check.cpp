#include <iostream>
#include <cmath>
#include <chrono>

#define CSC(call)                                            \

struct uchar4 {
    unsigned char x;
    unsigned char y;
    unsigned char z;
    unsigned char w;
};
struct uchar3 {
    unsigned char x;
    unsigned char y;
    unsigned char z;

};
struct uint3 {
    int x, y, z;
};


struct double3 {
    double x, y, z;
};
const int MAX_COUNT_OF_CLASSES = 32;
const int MATRIX_SIZE = 9;
double3 AVG[MAX_COUNT_OF_CLASSES];
double COV[MAX_COUNT_OF_CLASSES * MATRIX_SIZE];
double LOG_DET_COV[MAX_COUNT_OF_CLASSES];
double INVERSED_COV[MAX_COUNT_OF_CLASSES * MATRIX_SIZE];
struct Im {
    int w, h;
    uchar4 *arr;
};
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
uchar3 getpixel(Im& img, int x, int y) {
    uchar4 p = img.arr[img.w * y + x];
    uchar3 res;
    res.x = p.x;
    res.y = p.y;
    res.z = p.z;
    return res;
}

using matrix_3x3 = double[9];
double3 operator-(const double3& first, const double3& second) {
    double3 res;
    res.x = first.x - second.x;
    res.y = first.y - second.y;
    res.z = first.z - second.z;
    return res;
}
double* mul(const double3& column, const double3 &row) {
    double *arr = new matrix_3x3;
    arr[0] = column.x * row.x;
    arr[1] = column.x * row.y;
    arr[2] = column.x * row.z;
    arr[3] = column.y * row.x;
    arr[4] = column.y * row.y;
    arr[5] = column.y * row.z;
    arr[6] = column.z * row.x;
    arr[7] = column.z * row.y;
    arr[8] = column.z * row.z;
    return arr;
}
void sum(double* res, const double* matrix2) {
    for(int i = 0; i < 9; ++i) {
        res[i] += matrix2[i];
    }
}
void div(double* res, int x) {
    for(int i = 0; i < 9; ++i) {
        res[i] /= x;
    }
}
void print_cov(double *arr, int nc);
double3 *calc_avg(Im &data, int **arr,int nc) {
    double3 *avg = new double3[nc];
    for(int i = 0; i < nc; ++i) {
        int count = arr[i][0];
        uint3 sum = {0};
        for(int j = 1; j <= 2 * count; j += 2) {
            uchar3 p = getpixel(data, arr[i][j], arr[i][j+1]);
            sum.x += p.x;
            sum.y += p.y;
            sum.z += p.z;
        }
        avg[i].x = static_cast<double>(sum.x) / count;
        avg[i].y = static_cast<double>(sum.y) / count;
        avg[i].z = static_cast<double>(sum.z) / count;
    }
    return avg;
}
double *calc_cov(Im &data, int **arr, int nc, double3 *avg) {
    double *cov = new double[nc * 9];
    for(int class_ = 0; class_ < nc; ++class_) {
        int count = arr[class_][0];
        double *cov_ = new matrix_3x3{};
        for(int j = 1; j <= 2 * count; j += 2) {
            int x = arr[class_][j];
            int y = arr[class_][j + 1];
            uchar3 pix = getpixel(data, x, y);
            double3 p;
            p.x = pix.x;
            p.y = pix.y;
            p.z = pix.z;
            double *tmp = mul(p - avg[class_], p - avg[class_]);
            sum(cov_, tmp);
            delete[] tmp;
        }
        div(cov_, count - 1);
        for(int k = 0; k < 9; k++) {
            cov[class_ * 9 + k] = cov_[k];
        }
        delete[] cov_;
    }
    return cov;
}

void print_cov(double *arr, int nc) {
    for(int class_ = 0; class_ < nc; ++class_) {
        std::cout << "cov matrix for " << class_ + 1<< " class\n";
        for(int i = 0; i < 3; ++i) {
            for(int j = 0; j < 3; ++j) {
                std::cout.precision(3);
                std::cout << std::fixed << arr[class_ * 9 + 3 * i + j] << ' ';
            }
            std::cout << '\n';
        }
    }
    std::cout.flush();
}
void print_avg(double3 *arr, int size) {
    std::cout << "avg matrix\n";
    for(int i = 0; i < size; ++i) {
        std::cout << "(" <<arr[i].x << ',' << arr[i].y << ',' << arr[i].z<< ")" << '\n';
    }
    std::cout.flush();
}
void print_pixel(Im &im, int i, int j) {
    const uchar4 &p = im.arr[j * im.w + i];
    std::cout << "("<< static_cast<int>(p.x) << ',' << static_cast<int>(p.y) << ',' << static_cast<int>(p.z) << ")";
}
void print_img(Im &im) {
    for(int y = 0; y < im.w; ++y) {
        for(int x = 0; x < im.h; ++x) {
            print_pixel(im, x, y);
            std::cout << ' ';
        }
        std::cout << '\n';
    }
    std::cout.flush();
}
void print_class(Im &im, int class_, int **arr) {
    int count = arr[class_][0];
    for(int j = 1; j <= 2 * count; j += 2) {
        int x = arr[class_][j];
        int y = arr[class_][j + 1];
        print_pixel(im, x, y);
        std::cout << ' ';
    }
}
void print_classes(Im &im, int nc, int **arr) {
    for(int i = 0; i < nc; ++i) {
    std::cout << "Class: " << i + 1 << '\n';
        print_class(im, i, arr);
        std::cout << '\n';
    }
    std::cout.flush();
}
int** read_uniq_pixels(Im &im, int nc) {
    int **arr = new int*[nc];
    for(int i = 0; i < nc; ++i) {
        int np;
        std::cin >> np;
        arr[i] = new int[2 * np + 1];
        arr[i][0] = np;
        for(int j = 1; j <= 2 * np; ++j) {
            std::cin >> arr[i][j];
        }
    }
    return arr;
}
void free_uniq(int **arr, int nc) {
    for(int i = 0; i < nc; ++i) {
        delete[] arr[i];
    }
    delete[] arr;
}
double calc(double3 p, int j) {
    double3 pix;
    pix.x = p.x - AVG[j].x;
    pix.y = p.y - AVG[j].y;
    pix.z = p.z - AVG[j].z;
    double3 tmp;
    tmp.x = pix.x * INVERSED_COV[j * 9 + 0] +
            pix.y * INVERSED_COV[j * 9 + 3] + 
            pix.z * INVERSED_COV[j * 9 + 6];
    tmp.y = pix.x * INVERSED_COV[j * 9 + 1] +
            pix.y * INVERSED_COV[j * 9 + 4] + 
            pix.z * INVERSED_COV[j * 9 + 7];
    tmp.z = pix.x * INVERSED_COV[j * 9 + 2] +
            pix.y * INVERSED_COV[j * 9 + 5] + 
            pix.z * INVERSED_COV[j * 9 + 8];
    double res =  - (tmp.x * pix.x + tmp.y * pix.y + tmp.z * pix.z) - LOG_DET_COV[j];
    return res;
}
// jc = arg max j[-(p - avg_j) ^ T * cov_j ^(-1) * (p - avg_j) - log(|det(cov_j))]
void kernel(uchar4 *img, int nc, int size) {
for(int idx = 0; idx < size; ++idx) {
        uchar4 p = img[idx];
        double3 pix;
        pix.x = p.x;
        pix.y = p.y;
        pix.z = p.z;
        double max = calc(pix, 0);
        int class_ = 0;
        for(int j = 1; j < nc; ++j) {
            double curr = calc(pix, j);
            if(curr > max) {
                max = curr;
                class_ = j;
            }
        }
        img[idx].w = class_;
        #ifndef checker
        // img[idx].w = 255;

        // img[idx].x = AVG[class_].x;
        // img[idx].y = AVG[class_].y;
        // img[idx].z = AVG[class_].z;

        #endif //checker
    }
}
// 1 2 3
// * * *1
// * * *2
// * * *3
double algebraic_addition(const double *matrix, int nc, int i, int j) {
    int tmp = (i + j) & 1 ? -1: 1;
    int a[8] = {};
    int count = 0;
    for(int y = 0; y < 3; ++y) {
        for(int x = 0; x < 3; ++x) {
            if(y != i && j != x) {
                a[count * 2] = x;
                a[count * 2 + 1] = y;
                count++;
            }
        }
    }
    return tmp * (matrix[nc * 9 + 3 * a[0] + a[1]] * matrix[nc * 9 + 3 * a[6] + a[7]]
               - (matrix[nc * 9 + 3 * a[2] + a[3]] * matrix[nc * 9 + 3 * a[4] + a[5]]));
}
double det(const double *cov, int i);
void inverse_matrix_(double *matrix, const double *src, int nc) {
    double deter = det(src, nc);
    for(int i = 0; i < 3; ++i) {
        for(int j = 0; j < 3; ++j) {
            matrix[9 * nc + 3 * i + j] = algebraic_addition(src, nc, i, j) / deter;
        }
    }
}
double *inversed_matrix(double *cov, int nc) {
    double *res = new double[nc * 9]{};
    for(int i = 0; i < nc; ++i) {
        inverse_matrix_(res, cov, i);
    }
    return res;
}
double det(const double *cov, int i) {
    return cov[i * 9 + 0] * (cov[i * 9 + 4] * cov[i * 9 + 8] - cov[i * 9 + 7] * cov[i * 9 + 5]) 
         - cov[i * 9 + 1] * (cov[i * 9 + 3] * cov[i * 9 + 8] - cov[i * 9 + 6] * cov[i * 9 + 5]) 
         + cov[i * 9 + 2] * (cov[i * 9 + 3] * cov[i * 9 + 7] - cov[i * 9 + 4] * cov[i * 9 + 6]);
}
double *calc_log_det_cov(const double *cov, int nc) {
    double *res = new double[nc];
    for(int i = 0; i < nc; ++i) {
        res[i] = log(det(cov, i));
    }
    return res;
}

// #define checker
int main() {
    Im data;
    std::string path = "in.data";
    std::string outpath = "out.data";
    int nc;
#ifdef checker
    std::cin >> path;
    std::cin >> outpath;
#endif
    read_img(data, path);
    std::cin >> nc;
    int **arr = read_uniq_pixels(data, nc);
#ifndef checker
    print_classes(data, nc, arr);
#endif
    double3 *avg = calc_avg(data, arr, nc);
    double *cov = calc_cov(data, arr, nc, avg);
    free_uniq(arr, nc);
    double *log_det_cov = calc_log_det_cov(cov, nc);
    double *inversed_cov = inversed_matrix(cov, nc);
#ifndef checker
    // std::cout << "img\n";
    // print_img(data);
    print_avg(avg, nc);
    print_cov(cov, nc);
#endif
    for(int i = 0; i < nc; ++i) {
        AVG[i] = avg[i];
    }
    for(int i = 0; i < nc; ++i) {
        COV[i] = cov[i];
    }
    for(int i = 0; i < nc; ++i) {
        LOG_DET_COV[i] = log_det_cov[i];
    }
    for(int i = 0; i < MATRIX_SIZE * nc; ++i) {
        INVERSED_COV[i] = inversed_cov[i];
    }
    auto start_time = std::chrono::high_resolution_clock::now();
    kernel(data.arr, nc, data.w * data.h);
     auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "time " << duration.count() << " ms" << std::endl;
    write_img(data, outpath);
    CSC(cudaFree(img));
    delete[] data.arr;
    delete[] avg;
    delete[] cov;
    delete[] log_det_cov;
    delete[] inversed_cov;
}