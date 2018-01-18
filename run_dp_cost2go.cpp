#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <vector>
#include <chrono>
#include <iostream>
#include "dp_utils.h"
#include <algorithm>

struct cpu_prob_mem {
    float * c2g;
    int *  from;
    cpu_prob_mem(std::vector<float> const & /*a_options*/,
                 std::vector<float> const & v_options,
                 std::vector<float> const & s_options,
                 int n_times)
    {
        size_t V_SZ = v_options.size();
        size_t S_SZ = s_options.size();
        size_t sz = n_times*V_SZ*S_SZ;
        c2g = new float[sz];
        from = new int[sz];
    }

    ~cpu_prob_mem() {
        delete[] c2g;
        delete[] from;
    }
};

int speed_dp_serial(cpu_prob_mem & p_mem,
                    std::vector<float> const & a_options,
                     std::vector<float> const & v_options,
                     std::vector<float> const & s_options,
                     int n_times,
                     int initial_v_idx)
{

    if(initial_v_idx < 0 || initial_v_idx >= v_options.size()) {
        std::cerr << "Invalid inital speed " << std::endl;
        return -1;
    }

    float const COST_INFEASIBLE = 999999.9f;

    size_t A_SZ = a_options.size();
    size_t V_SZ = v_options.size();
    size_t S_SZ = s_options.size();

//    size_t sz = n_times*V_SZ*S_SZ;
//    float * c2g = new float[sz];
//    int *  from = new int[sz];

    //initialize data
    for(int t_idx=0; t_idx<n_times; ++t_idx) {
        for(int s_idx=0; s_idx<S_SZ; ++s_idx) {
            for(int v_idx=0; v_idx<V_SZ; ++v_idx) {
                p_mem.from[t_idx*S_SZ*V_SZ + s_idx*V_SZ + v_idx] = -1;
                if(t_idx < n_times-1) {
                    p_mem.c2g[t_idx*S_SZ*V_SZ + s_idx*V_SZ + v_idx] = COST_INFEASIBLE;
                } else {
                    //cost to go from last time is zero
                    p_mem.c2g[t_idx*S_SZ*V_SZ + s_idx*V_SZ + v_idx] = 0.0;
                }
            }
        }
    }

    float const delta_s   = s_options[1]-s_options[0];
    float const delta_v   = v_options[1]-v_options[0];

    for(int t_idx=n_times-1; t_idx>0; t_idx--) {
        for(int v_idx=0; v_idx<V_SZ; ++v_idx) {
            for(int s_idx=0; s_idx<S_SZ; ++s_idx) {
                for(int a_idx=0; a_idx<A_SZ; ++a_idx) {
                    //move backward in time
                    float const s_end = s_options[s_idx];
                    float const v_end = v_options[v_idx];

                    //this is wrong!
                    //float s_start = s_options[s_idx] - v_options[v_idx] - a_options[a_idx]/2;

                    float v_start = v_options[v_idx] - a_options[a_idx];
                    float s_start = s_options[s_idx] - v_start - a_options[a_idx]/2;



                    int s_start_idx = s_start/delta_s;
                    int v_start_idx = v_start/delta_v;

//                    if(fabs(s_end-1.25) < 1.0e-5 && fabs(v_end-2.5) < 1.0e-5) {
//                        std::cout << "a = " << a_options[a_idx] << " v_start = " << v_start << " s_start = " << s_start << std::endl;
//                        std::cout << "s_start_idx = " << s_start_idx << " v_start_idx = " << v_start_idx << std::endl;
//                    }

                    if(v_start_idx < 0 || s_start_idx < 0 ||
                            v_start_idx >= V_SZ || s_start_idx >= S_SZ) {
                        //Nop
                    } else {

                        //check if cost is lower, then update it
                        float acc_cost   = 0.5*a_options[a_idx]*a_options[a_idx];
                        //v_cost has to be on end speed!
                        float v_cost     = 0.2*(v_end-10.0)*(v_end-10.0);
                        float delta_cost = acc_cost + v_cost;

                        //Cost for being close to a Normally distriubted obstacle
                        // @ s=40,sigma=2.0
                        float const d_mean   = s_end-40.0f;
                        float const d_sigma  = 2.0f;
                        // cost ~ how much probabily mass is within 2.0 from 0.0
                        // int_{-inf}^2 - inf_{-inf}^-2
                        delta_cost += call_norm_cdf((2.0f-d_mean)/d_sigma) - call_norm_cdf((-2.0f-d_mean)/d_sigma);


                        //Hard coded obstacle... (at t=3, s=[20,30] is in collision)
                        if(t_idx == 3 && s_end >= 20.0 && s_end <= 30.0) {
                            delta_cost = COST_INFEASIBLE;
                        }

                        //c2g (t,s,v)
                        int c2g_idx_curr  = (t_idx  )*(S_SZ*V_SZ) + s_idx*V_SZ + v_idx;
                        int c2g_idx_start = (t_idx-1)*(S_SZ*V_SZ) + s_start_idx*V_SZ + v_start_idx;

                        if(delta_cost + p_mem.c2g[c2g_idx_curr] < p_mem.c2g[c2g_idx_start]) {
                            p_mem.c2g[c2g_idx_start]  = delta_cost + p_mem.c2g[c2g_idx_curr];
                            p_mem.from[c2g_idx_start] = s_idx*V_SZ + v_idx;
                        }
                    }
                }
            }
        }
    }

    //backtrack
    //auto started = std::chrono::high_resolution_clock::now();

    float cost = p_mem.c2g[0*(S_SZ*V_SZ) + 0*V_SZ + initial_v_idx];
    std::cout << "Optimal cost = " << cost << std::endl;

    std::cout << "optimal speed prof: ";
    std::cout << "(" << s_options[0] << ", " << v_options[initial_v_idx] << "); ";
    int idx_nxt = p_mem.from[initial_v_idx];
    for(int t_idx=1; t_idx<n_times; ++t_idx) {
        //unwind index
        int s_idx = idx_nxt/V_SZ;
        int v_idx = idx_nxt-(s_idx*V_SZ);
        std::cout << "(" << s_options[s_idx] << ", " << v_options[v_idx] << "); ";
        idx_nxt   = p_mem.from[t_idx*(S_SZ*V_SZ) + s_idx*V_SZ + v_idx];
    }
    std::cout << std::endl;

    //auto done = std::chrono::high_resolution_clock::now();
    //std::cout << "Backtrack time = " << std::chrono::duration_cast<std::chrono::milliseconds>(done-started).count() << " ms " << std::endl;

//    delete[] from;
//    delete[] c2g;
    return 0;
}


extern
int speed_dp(prob_mem & p_mem,
             std::vector<float> const & a_options,
             std::vector<float> const & v_options,
             std::vector<float> const & s_options,
             int n_times,
             int initial_v_idx,
             bool print);

extern prob_mem setup_memory(std::vector<float> const & a_options,
                             std::vector<float> const & v_options,
                             std::vector<float> const & s_options,
                             int n_times);

extern void clear_memory(prob_mem & p_mem);

int main(int argc, char **argv)
{
    std::vector<float> const a_options = {-2.5,  -1.25,  0.0,    1.25,  2.5};
    std::vector<float> const v_options = {0.0, 1.25, 2.5, 3.75, 5.0, 6.25, 7.5, 8.75, 10.0, 11.25, 12.5, 13.75, 15.0, 16.25, 17.5, 18.75, 20.0, 21.25, 22.5, 23.75, 25.0, 26.25, 27.5, 28.75, 30.0, 31.25, 32.5, 33.75, 35.0};
    std::vector<float> const s_options = {0.0, 0.625, 1.25, 1.875, 2.5, 3.125, 3.75, 4.375, 5.0, 5.625, 6.25, 6.875, 7.5, 8.125, 8.75, 9.375, 10.0, 10.625, 11.25, 11.875, 12.5, 13.125, 13.75, 14.375, 15.0, 15.625, 16.25, 16.875, 17.5, 18.125, 18.75, 19.375, 20.0, 20.625, 21.25, 21.875, 22.5, 23.125, 23.75, 24.375, 25.0, 25.625, 26.25, 26.875, 27.5, 28.125, 28.75, 29.375, 30.0, 30.625, 31.25, 31.875, 32.5, 33.125, 33.75, 34.375, 35.0, 35.625, 36.25, 36.875, 37.5, 38.125, 38.75, 39.375, 40.0, 40.625, 41.25, 41.875, 42.5, 43.125, 43.75, 44.375, 45.0, 45.625, 46.25, 46.875, 47.5, 48.125, 48.75, 49.375, 50.0, 50.625, 51.25, 51.875, 52.5, 53.125, 53.75, 54.375, 55.0, 55.625, 56.25, 56.875, 57.5, 58.125, 58.75, 59.375, 60.0, 60.625, 61.25, 61.875, 62.5, 63.125, 63.75, 64.375, 65.0, 65.625, 66.25, 66.875, 67.5, 68.125, 68.75, 69.375, 70.0, 70.625, 71.25, 71.875, 72.5, 73.125, 73.75, 74.375, 75.0, 75.625, 76.25, 76.875, 77.5, 78.125, 78.75, 79.375, 80.0, 80.625, 81.25, 81.875, 82.5, 83.125, 83.75, 84.375, 85.0, 85.625, 86.25, 86.875, 87.5, 88.125, 88.75, 89.375, 90.0, 90.625, 91.25, 91.875, 92.5, 93.125, 93.75, 94.375, 95.0, 95.625, 96.25, 96.875, 97.5, 98.125, 98.75, 99.375, 100.0, 100.625, 101.25, 101.875, 102.5, 103.125, 103.75, 104.375, 105.0, 105.625, 106.25, 106.875, 107.5, 108.125, 108.75, 109.375, 110.0, 110.625, 111.25, 111.875, 112.5, 113.125, 113.75, 114.375, 115.0, 115.625, 116.25, 116.875, 117.5, 118.125, 118.75, 119.375, 120.0, 120.625, 121.25, 121.875, 122.5, 123.125, 123.75, 124.375, 125.0, 125.625, 126.25, 126.875, 127.5, 128.125, 128.75, 129.375, 130.0, 130.625, 131.25, 131.875, 132.5, 133.125, 133.75, 134.375, 135.0, 135.625, 136.25, 136.875, 137.5, 138.125, 138.75, 139.375, 140.0, 140.625, 141.25, 141.875, 142.5, 143.125, 143.75, 144.375, 145.0, 145.625, 146.25, 146.875, 147.5, 148.125, 148.75, 149.375, 150.0, 150.625, 151.25, 151.875, 152.5, 153.125, 153.75, 154.375, 155.0, 155.625, 156.25, 156.875, 157.5, 158.125, 158.75, 159.375, 160.0, 160.625, 161.25, 161.875, 162.5, 163.125, 163.75, 164.375, 165.0, 165.625, 166.25, 166.875, 167.5, 168.125, 168.75, 169.375, 170.0, 170.625, 171.25, 171.875, 172.5, 173.125, 173.75, 174.375, 175.0, 175.625, 176.25, 176.875, 177.5, 178.125, 178.75, 179.375, 180.0, 180.625, 181.25, 181.875, 182.5, 183.125, 183.75, 184.375, 185.0, 185.625, 186.25, 186.875, 187.5, 188.125, 188.75, 189.375, 190.0, 190.625, 191.25, 191.875, 192.5, 193.125, 193.75, 194.375, 195.0, 195.625, 196.25, 196.875, 197.5, 198.125, 198.75, 199.375, 200.0, 200.625, 201.25, 201.875, 202.5, 203.125, 203.75, 204.375, 205.0, 205.625, 206.25, 206.875, 207.5, 208.125, 208.75, 209.375, 210.0, 210.625, 211.25, 211.875, 212.5, 213.125, 213.75, 214.375, 215.0, 215.625, 216.25, 216.875, 217.5, 218.125, 218.75, 219.375, 220.0, 220.625, 221.25, 221.875, 222.5, 223.125, 223.75, 224.375, 225.0, 225.625, 226.25, 226.875, 227.5, 228.125, 228.75, 229.375, 230.0, 230.625, 231.25, 231.875, 232.5, 233.125, 233.75, 234.375, 235.0, 235.625, 236.25, 236.875, 237.5, 238.125, 238.75, 239.375, 240.0, 240.625, 241.25, 241.875, 242.5, 243.125, 243.75, 244.375, 245.0, 245.625, 246.25, 246.875, 247.5, 248.125, 248.75, 249.375, 250.0, 250.625, 251.25, 251.875, 252.5, 253.125, 253.75, 254.375, 255.0, 255.625, 256.25, 256.875, 257.5, 258.125, 258.75, 259.375, 260.0, 260.625, 261.25, 261.875, 262.5, 263.125, 263.75, 264.375, 265.0, 265.625, 266.25, 266.875, 267.5, 268.125, 268.75, 269.375, 270.0, 270.625, 271.25, 271.875, 272.5, 273.125, 273.75, 274.375, 275.0, 275.625, 276.25, 276.875, 277.5, 278.125, 278.75, 279.375, 280.0, 280.625, 281.25, 281.875, 282.5, 283.125, 283.75, 284.375, 285.0, 285.625, 286.25, 286.875, 287.5, 288.125, 288.75, 289.375, 290.0, 290.625, 291.25, 291.875, 292.5, 293.125, 293.75, 294.375, 295.0, 295.625, 296.25, 296.875, 297.5, 298.125, 298.75, 299.375, 300.0, 300.625, 301.25, 301.875, 302.5, 303.125, 303.75, 304.375, 305.0, 305.625, 306.25, 306.875, 307.5, 308.125, 308.75, 309.375, 310.0, 310.625, 311.25, 311.875, 312.5, 313.125, 313.75, 314.375, 315.0, 315.625, 316.25, 316.875, 317.5, 318.125, 318.75, 319.375, 320.0, 320.625, 321.25, 321.875, 322.5, 323.125, 323.75, 324.375, 325.0, 325.625, 326.25, 326.875, 327.5, 328.125, 328.75, 329.375, 330.0, 330.625, 331.25, 331.875, 332.5, 333.125, 333.75, 334.375, 335.0, 335.625, 336.25, 336.875, 337.5, 338.125, 338.75, 339.375, 340.0, 340.625, 341.25, 341.875, 342.5, 343.125, 343.75, 344.375, 345.0, 345.625, 346.25, 346.875, 347.5, 348.125, 348.75, 349.375, 350.0};

    int initial_v_idx = 10; //12.5 m/s
    int n_times = 10;

    auto started_cpu = std::chrono::high_resolution_clock::now();
    cpu_prob_mem cpu_p_mem(a_options,v_options,s_options,n_times);
    auto done_cpu = std::chrono::high_resolution_clock::now();
    std::cout << "Total time for memory alloc on CPU... = " << std::chrono::duration_cast<std::chrono::microseconds>(done_cpu-started_cpu).count()/1000.0 << " ms " << std::endl;

    std::vector<double> cpu_runtimes;
    for(int num_tries=0; num_tries<10; ++num_tries) {
        auto started_cpu = std::chrono::high_resolution_clock::now();
        speed_dp_serial(cpu_p_mem,a_options,v_options,s_options,n_times,initial_v_idx);
        auto done_cpu = std::chrono::high_resolution_clock::now();
        double rt = std::chrono::duration_cast<std::chrono::microseconds>(done_cpu-started_cpu).count()/1000.0;
        cpu_runtimes.push_back(rt);
        std::cout << "Total time for search on CPU... = " << rt << " ms " << std::endl;
    }

    //setup memory
    auto started_gpu = std::chrono::high_resolution_clock::now();
    prob_mem p_mem = setup_memory(a_options,v_options,s_options,n_times);
    auto done_gpu = std::chrono::high_resolution_clock::now();
    std::cout << "Total time for memory alloc GPU... = " << std::chrono::duration_cast<std::chrono::microseconds>(done_gpu-started_gpu).count()/1000.0 << " ms " << std::endl;

    //First run, needs to compile to executable code on the gpu...
    std::vector<double> gpu_runtimes;
    for(int num_tries=0; num_tries<10; ++num_tries) {
        auto started_gpu = std::chrono::high_resolution_clock::now();
        speed_dp(p_mem,a_options,v_options,s_options,n_times,initial_v_idx,num_tries>-1);
        auto done_gpu = std::chrono::high_resolution_clock::now();
        double rt = std::chrono::duration_cast<std::chrono::microseconds>(done_gpu-started_gpu).count()/1000.0;
        gpu_runtimes.push_back(rt);
        if(num_tries>-1)
            std::cout << "Total time for search on GPU... = " << rt << " ms " << std::endl;
    }

    clear_memory(p_mem);

    std::cout << "MEAN runtime cpu: " << std::accumulate(cpu_runtimes.begin(), cpu_runtimes.end(), 0.0)/10.0 << std::endl;
    std::cout << "MEAN runtime gpu: " << std::accumulate(gpu_runtimes.begin(), gpu_runtimes.end(), 0.0)/10.0 << std::endl;

    return 0;
}
