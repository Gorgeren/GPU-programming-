#ifndef FUNCTIONS_CUH
#define FUNCTIONS_CUH
#include <fstream>
#include <iostream>
#include <vector>
#include <set>

#include "polygon.cuh"
#include "structs.cuh"
#include "figure.cuh"
#include "plane.cuh"
#include "texture.cuh"

#define CSC(call)                                                      \
do {                                                                   \
    cudaError_t status = call;                                         \
    if (status != cudaSuccess) {                                       \
        fprintf(stderr, "ERROR[%s:%d]: %s\n", __FILE__, __LINE__,      \
                cudaGetErrorString(status));                            \
        exit(0);                                                       \
    }                                                                  \
} while(0)




extern const int FIGURES;
extern int frames;
extern std::string outpattern;
extern int w, h;
extern float angle;
extern float r0_c, z0_c, phi0_c, Ar_c, Az_c, wr_c, wz_c, wphi_c, pr_c, pz_c;
extern float r0_n, z0_n, phi0_n, Ar_n, Az_n, wr_n, wz_n, wphi_n, pr_n, pz_n;
extern figure_t figures[3];
extern texture_t floor_tex;
extern std::string texturepath;
extern std::vector<vec3> floor_coord;
extern int torches_count;
extern std::vector<torch> torches;
extern int n_rec;
extern int ssaa;


float figure_side(const std::vector<vec3>& points) {
    float side = INF;
    int n = points.size();
    for(int i = 0; i < n - 1; ++i) {
        for(int j = i + 1; j < n; ++j) {
            side = std::min((points[i] - points[j]).len(), side);
        }
    }
    return side;
}

std::vector<polygon> polygons;
std::pair<std::vector<vec3>, std::vector<vec3i>> parse_obj(const std::string& file_name, int figure_id) {
    const figure_t &fig = figures[figure_id];
    std::vector<vec3> points;
    std::vector<vec3i> poligons;
    {
        std::ifstream in(file_name);
        if(!in) {
            std::cerr << "file: " << file_name << " wasn't open";
            exit(1);
        }
        std::string ch;
        vec3 point;
        vec3i poligon;
        while(in >> ch) {
            if(ch == "v") {
                in >> point;
                points.push_back(fig.r * point); 

            } else if(ch == "f") {
                in >> poligon;
                poligon -= 1;
                poligons.push_back(poligon);
            }
        }
    }
    return {points, poligons};
}

void print_norm(const trig& tr) {
    vec3 center = tr.a + tr.b + tr.c;
    center = center / 3.;
    std::cout << center;
    std::cout << center + tr.n;
    std::cout << "\n\n\n";
}
void print_trig(std::ofstream& out, const trig& tr) {
    out << tr.a << tr.b << tr.c << tr.a;
    out << "\n\n\n";
}
void build_figure(const std::string &file_name, int figure_id) {
    const figure_t &fig = figures[figure_id];
    auto tmp = parse_obj(file_name, figure_id); // В с++11 нет auto [] 
    const std::vector<vec3> &points = tmp.first;
    const std::vector<vec3i> &poligons = tmp.second;
    int n = static_cast<int>(points.size());
    std::vector<std::set<int>> point_poligon(n);
    for(int i = 0; i < static_cast<int>(poligons.size()); ++i) {
        const vec3i& poligon = poligons[i];
        point_poligon[poligon.x].insert(i);
        point_poligon[poligon.y].insert(i);
        point_poligon[poligon.z].insert(i);
    }

    float edge_size = figure_side(points);
    std::set<int> repeat_side;
    for(int i = 0; i < n - 1; ++i) {
        for(int j = i + 1; j < n; ++j) {
            const vec3 &pi = points[i];
            const vec3 &pj = points[j];
            float len = (pi - pj).len();
            if(std::abs(len - edge_size) <= EPS) {
                std::vector<trig> common_tr;
                std::vector<int> pol_id; 
                for(int p: point_poligon[i]) {
                    if(point_poligon[j].count(p)) {
                        pol_id.push_back(p);
                        const vec3i& tmp = poligons[p];
                        common_tr.push_back({points[tmp.x], points[tmp.y], points[tmp.z]});
                    }
                }

                trig trig1 = common_tr[0];
                trig trig2 = common_tr[1];
                int Pid1 = pol_id[0];
                int Pid2 = pol_id[1];

                // нужно будет проверить, что нормаль торчит наружу
                vec3 n1 = MAGIC_THICKNESS * trig1.n;
                vec3 n2 = MAGIC_THICKNESS * trig2.n;
                vec3 n_avg = (n1 + n2) / 2;
                trig1 += n1;
                trig2 += n2;
                if(!repeat_side.count(Pid1))
                    polygons.push_back({trig1 + fig.center, fig.color, fig.reflection, fig.transparency});
                    repeat_side.insert(Pid1);
                if(!repeat_side.count(Pid2))
                    polygons.push_back({trig2 + fig.center, fig.color, fig.reflection, fig.transparency});
                    repeat_side.insert(Pid2);
                if(std::abs(vec3::dot(trig1.n, trig2.n) - 1) < EPS) {
                    continue;
                }
                vec3 pj1{pj + n1};
                vec3 pj2{pj + n2};
                vec3 pi1{pi + n1};
                vec3 pi2{pi + n2};

                trig trigPi2Pi1Pj1(pi1, pj2, pi2);
                trig trigPi2Pj2Pj1(pi1, pj1, pj2);
                trigPi2Pi1Pj1.bidirectional_n(n_avg);
                trigPi2Pj2Pj1.bidirectional_n(n_avg);
                polygons.push_back({trigPi2Pi1Pj1 + fig.center, MAGIC_CHARACTERISTICS,
                                    fig.lights, 0, texture_t(),
                                    (pi1 + pi2) / 2 + fig.center,
                                    (pj1 + pj2) / 2 + fig.center
                                    });
                
                polygons.push_back({trigPi2Pj2Pj1 + fig.center, MAGIC_CHARACTERISTICS,
                                    fig.lights, 0, texture_t(),
                                    (pi1 + pi2) / 2 + fig.center,
                                    (pj1 + pj2) / 2 + fig.center
                                    });
                float dist = plane::distance(vec3{}, pi, trigPi2Pi1Pj1);
                trig addi(pi1, pi2, dist * pi / pi.len());
                dist = plane::distance(vec3{}, pj, trigPi2Pi1Pj1);
                trig addj(pj1, dist * pj / pj.len(), pj2);
                addi.bidirectional_n(n_avg);
                addj.bidirectional_n(n_avg);
                addi += fig.center;
                addj += fig.center;
                polygons.push_back({addi, MAGIC_CHARACTERISTICS});
                polygons.push_back({addj, MAGIC_CHARACTERISTICS});
            }
        }
    }
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

void read_in(bool gpu) {
    std::cin >> frames;
    std::cin >> outpattern;
    std::cin >> w >> h >> angle;
    std::cin >> r0_c >> z0_c >> phi0_c >> Ar_c >> Az_c >> wr_c >> wz_c >> wphi_c >>
        pr_c >> pz_c;
    std::cin >> r0_n >> z0_n >> phi0_n >> Ar_n >> Az_n >> wr_n >> wz_n >> wphi_n >>
        pr_n >> pz_n;
    for(int i = 0; i < FIGURES; ++i) {
        std::cin >> figures[i];
    }
    // floor
    for(vec3 &coord: floor_coord) {
        std::cin >> coord;
    }
    std::cin >> texturepath;
    vec3 color;
    float refl;
    std::cin >> color >> refl;
    floor_tex.gpu = gpu;
    floor_tex.read_texture(texturepath);
    trig first = {floor_coord[0], floor_coord[2], floor_coord[1]};
    trig second = {floor_coord[2], floor_coord[0], floor_coord[3]};
    vec3 b1 = floor_coord[1] - floor_coord[2];
    vec3 b2 = floor_coord[1] - floor_coord[0];
    vec3 b3 = floor_coord[0] + floor_coord[2] - floor_coord[1];
    polygons.push_back({first, color, refl, 0, 0, 1, floor_tex, b1, b2, b3});
    b1 = floor_coord[0] - floor_coord[3];
    b2 = floor_coord[2] - floor_coord[3];
    b3 = floor_coord[3];
    polygons.push_back({second, color, refl, 0, 0, 1, floor_tex, b1, b2, b3});
    // end floor
    std::cin >> torches_count;
    torches.resize(torches_count);
    for(auto &torch: torches) {
        std::cin >> torch;
    }
    build_figure("in/cube.obj", 0);
    build_figure("in/octahedron.obj", 1);
    build_figure("in/tetrahedron.obj", 2);
    std::cin >> n_rec;
    std::cin >> ssaa;
#ifdef debug
    for(const polygon& i: polygons) {
        print_trig(logfile, i.tr);
        // print_norm(i);
    }
    logfile << "\n\n\n";
    for(const torch& i: torches) {
        logfile << i.p << '\n';
        logfile << i.p + vec3{0, 0, 1} << "\n\n\n";
    }
#endif
}

#endif
