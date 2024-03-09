#include <iostream>
#include <cmath>
#include <fstream>
#include <thread>
#include <unistd.h>

#define INF 1000000000
#define EPS 1e-4
#define MAGIC_THICKNESS 0.1f
#define COLOR vec3{0.29f, 0.25f, 0.64f}
#define MAGIC_CHARACTERISTICS COLOR, 0.f, 0.f
#define LIGHT vec3{30., 30., 30.}
#define LIGHT_SIZE (MAGIC_THICKNESS / 4.f)

#ifdef debug
std::ofstream logfile("o.plt");
#endif

#include "lib/functions.cuh"
#include "lib/render.cuh"

const int FIGURES = 3;
int frames;
std::string outpattern;
int w, h;
float angle;
float r0_c, z0_c, phi0_c, Ar_c, Az_c, wr_c, wz_c, wphi_c, pr_c, pz_c;
float r0_n, z0_n, phi0_n, Ar_n, Az_n, wr_n, wz_n, wphi_n, pr_n, pz_n;
figure_t figures[3];
texture_t floor_tex;
std::string texturepath;
std::vector<vec3> floor_coord(4);
int torches_count;
std::vector<torch> torches;
int n_rec;
int ssaa;
int main(int argc, const char* argv[]) {
    bool gpu = 1;
    if(argc >= 2) {
        std::vector<std::string> args(argv + 1, argv + argc);
        for(const auto& str : args) {
            if(str == "--cpu") {
                gpu = 0;
            }
            if(str == "--gpu") {
                gpu = 1;
            }
            if(str == "--default") {
                std::fstream example_ouput("in/normal.in");
                std::string example;
                while(std::getline(example_ouput, example)) {
                    std::cout << example << '\n';
                }
                return 0;
            }
        }
    }
    read_in(gpu);
    char outfile[100];
    uchar4 *data = new uchar4[w * h];
    uchar4 *dev_data = nullptr;
    CSC(cudaMalloc(&dev_data, sizeof(uchar4) * w * h));
    float tau = 2 * M_PI / frames;
#ifdef debug
    std::cerr << "Runnig in debug mode\n";
    // w = std::min(w, 800);
    // h = std::min(h, 800);
    for(int k = 0; k < frames; ++k) {
        float t = k * tau;
        vec3 camera_pos = vec3::cyl(r0_c + Ar_c * sin(wr_c * t + pr_c),
                             z0_c + Az_c * sin(wz_c * t + pz_c),
                             phi0_c + wphi_c * t);
        logfile << camera_pos << '\n';
        if(k % 10 == 0) {
            vec3 camera_v = vec3::cyl(r0_n + Ar_n * sin(wr_n * t + pr_n),
                            z0_n + Az_n * sin(wz_n * t + pz_n),
                            phi0_n + wphi_n * t);
            logfile << camera_pos + (camera_v - camera_pos).norm() << '\n';
            logfile << camera_pos << '\n';
        }
    }
    vec3 firstpos = vec3::cyl(r0_c + Ar_c * sin(pr_c),
                             z0_c + Az_c * sin(pz_c),
                             phi0_c);
    logfile << firstpos << '\n';
    logfile << firstpos << '\n' << firstpos + vec3{0,0,1} << '\n';

    logfile << "\n\n\n";
    frames = 1;
#endif
    vec3 tmp;
    for(int k = 738; k < 744; ++k) {
        float t = k * tau;
        vec3 camera_pos = vec3::cyl(r0_c + Ar_c * sin(wr_c * t + pr_c),
                             z0_c + Az_c * sin(wz_c * t + pz_c),
                             phi0_c + wphi_c * t);
        vec3 camera_v = vec3::cyl(r0_n + Ar_n * sin(wr_n * t + pr_n),
                             z0_n + Az_n * sin(wz_n * t + pz_n),
                             phi0_n + wphi_n * t);

        gpu ?
            render_gpu(camera_pos, camera_v, w, h, angle, dev_data)
        :
            render_cpu(camera_pos, camera_v, w, h, angle, data);
        if(gpu) {
            CSC(cudaMemcpy(data, dev_data, sizeof(uchar4) * w * h, cudaMemcpyDeviceToHost));
        }
        sprintf(outfile, outpattern.c_str(), k);
        std::cout << k << '\n';
        write_binarypic(outfile, w, h, data);
    }
    delete[] data;
}
