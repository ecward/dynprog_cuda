#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <vector>
#include <iostream>

#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <chrono>

#include "helper_cuda.h"

/*
 * We need to exploit the problem structure to make it fast!
 *
 * For any specific speed, we cannot reach the same (s,v) pair, so
 * we can start a bunch of threads in parallell (number of s values x number of a values)
 *
 *
 *
 */

struct move_options {
    float * a_opt;
    float * v_opt;
    float * s_opt;
    int A_SZ;
    int V_SZ;
    int S_SZ;
};

float const COST_INFEASIBLE = 999999.9f;

//Maybe the very large c2g and from matrices really
//screws up hitting in the cache...

__global__ void make_move(float * const c2g,
                          int   * const from,
                          int const v_idx,
                          int const t_idx,
                          move_options const * const move_opt)
{

    //2d grid over s,a
    int s_idx = blockIdx.x*blockDim.x  + threadIdx.x;
    int a_idx = blockIdx.y*blockDim.y  + threadIdx.y;

    float const v_end = move_opt->v_opt[v_idx];
    float const s_end = move_opt->s_opt[s_idx];

    if(s_idx < move_opt->S_SZ && a_idx < move_opt->A_SZ) {
        //Previous state!
        //s_start = s_end - v*Dt - a*Dt*Dt/2
        //v_start = v_end - a*DT
        float s_start = move_opt->s_opt[s_idx] - move_opt->v_opt[v_idx] - move_opt->a_opt[a_idx]/2;
        float v_start = move_opt->v_opt[v_idx] - move_opt->a_opt[a_idx];

        float const delta_s   = move_opt->s_opt[1]-move_opt->s_opt[0];
        float const delta_v   = move_opt->v_opt[1]-move_opt->v_opt[0];
        int s_start_idx = s_start/delta_s;
        int v_start_idx = v_start/delta_v;
        if(v_start_idx < 0 || s_start_idx < 0 ||
                v_start_idx >= move_opt->V_SZ || s_start_idx >= move_opt->S_SZ) {
            //Nop
        } else {
            //check if cost is lower, then update it
            float acc_cost   = 0.5*move_opt->a_opt[a_idx]*move_opt->a_opt[a_idx];
            //v_cost has to be on end speed!
            float v_cost     = 0.2*(v_end-10.0)*(v_end-10.0);
            float delta_cost = acc_cost + v_cost;

            //Hard coded obstacle... (at t=3, s=[20,30] is in collision)
            if(t_idx == 3 && s_end >= 20.0 && s_end <= 30.0) {
                delta_cost = COST_INFEASIBLE;
            }


            //c2g (t,s,v)
            int c2g_idx_curr  = (t_idx  )*(move_opt->S_SZ*move_opt->V_SZ) + s_start_idx*move_opt->V_SZ + v_start_idx;
            int c2g_idx_start = (t_idx-1)*(move_opt->S_SZ*move_opt->V_SZ) + s_start_idx*move_opt->V_SZ + v_start_idx;

            if(delta_cost + c2g[c2g_idx_curr] < c2g[c2g_idx_start]) {
                c2g[c2g_idx_start]  = delta_cost + c2g[c2g_idx_curr];
                from[c2g_idx_start] = s_idx*move_opt->V_SZ + v_idx;
            }

        }
    }
}

extern
int speed_dp(std::vector<float> const & a_options,
             std::vector<float> const & v_options,
             std::vector<float> const & s_options,
             int n_times,
             int initial_v_idx) {

    if(s_options.size() > 1024 || a_options.size() > 1024) {
        std::cerr << "Maximum dimensions exceeded" << std::endl;
        return -1;
    }

    if(initial_v_idx < 0 || initial_v_idx >= v_options.size()) {
        std::cerr << "Invalid inital speed " << std::endl;
        return -1;
    }


    // initialise CUDA timing
    float milli;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    cudaEventRecord(start);

    ///@todo c2g doesn't need to include more than two timesteps...
    //allocate and set data
    float * dev_c2g;   //(t,s,v)
    int   * dev_from;  //(t,s,v)
    float * dev_a_opt;
    float * dev_v_opt;
    float * dev_s_opt;

    int sz = n_times*v_options.size()*s_options.size();
    checkCudaErrors( cudaMalloc((void **)&dev_c2g,sz*sizeof(float)) );
    checkCudaErrors( cudaMalloc((void **)&dev_from,sz*sizeof(int)) );

    //At last time all costs are zero, otherwise, initialize to COST_INFEASIBLE..
    thrust::device_ptr<float> dev_ptr(dev_c2g);
    thrust::fill(dev_ptr, dev_ptr + sz, COST_INFEASIBLE);
    thrust::fill(dev_ptr+(n_times-1)*(v_options.size()*s_options.size()),dev_ptr+sz,0.0);

    //cudaMemset only works for int values...
    //checkCudaErrors( cudaMemset(dev_c2g,COST_INFEASIBLE,sz*sizeof(float)) );
    //checkCudaErrors( cudaMemset(dev_c2g+(n_times-1)*(v_options.size()*s_options.size()),0.0,(v_options.size()*s_options.size())*sizeof(float)) );

    checkCudaErrors( cudaMemset(dev_from,-1,sz*sizeof(int)) );

    checkCudaErrors( cudaMalloc((void **)&dev_a_opt,(a_options.size())*sizeof(float)) );
    checkCudaErrors( cudaMemcpy(dev_a_opt,a_options.data(),a_options.size()*sizeof(float),cudaMemcpyHostToDevice) );

    checkCudaErrors( cudaMalloc((void **)&dev_v_opt,(v_options.size())*sizeof(float)) );
    checkCudaErrors( cudaMemcpy(dev_v_opt,v_options.data(),v_options.size()*sizeof(float),cudaMemcpyHostToDevice) );

    checkCudaErrors( cudaMalloc((void **)&dev_s_opt,(s_options.size())*sizeof(float)) );
    checkCudaErrors( cudaMemcpy(dev_s_opt,s_options.data(),s_options.size()*sizeof(float),cudaMemcpyHostToDevice) );

    //Struct on device
    move_options   host_opt;
    move_options * dev_opt_ptr;
    cudaMalloc((void**)&dev_opt_ptr,sizeof(move_options));
    //raw ptrs
    host_opt.a_opt = dev_a_opt;
    host_opt.v_opt = dev_v_opt;
    host_opt.s_opt = dev_s_opt;
    host_opt.A_SZ = a_options.size();
    host_opt.S_SZ = s_options.size();
    host_opt.V_SZ = v_options.size();
    //copy data to device
    checkCudaErrors( cudaMemcpy(dev_opt_ptr,&host_opt,sizeof(move_options),cudaMemcpyHostToDevice) );

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milli, start, stop);
    std::cout << "Device memory allocation in " << milli << " ms " << std::endl;

    ///@todo this can probably be done better...
    dim3 block_dim(host_opt.S_SZ,1);
    dim3 grid_dim(1,host_opt.A_SZ);

    //These loops have to be done in sequence
    for(int t_idx=n_times-1; t_idx>0; t_idx--) {
        for(int v_idx=0; v_idx<host_opt.V_SZ; ++v_idx) {
            make_move<<<grid_dim,block_dim>>>(dev_c2g,dev_from,v_idx,t_idx,dev_opt_ptr);
        }
    }

    //Backtrack
    auto started = std::chrono::high_resolution_clock::now();
    float * host_c2g  = new float[sz];
    int   * host_from = new int[sz];

    cudaMemcpy(host_c2g,dev_c2g,sz*sizeof(float),cudaMemcpyDeviceToHost);
    cudaMemcpy(host_from,dev_from,sz*sizeof(int),cudaMemcpyDeviceToHost);


    //by definition we start at s=0
    //find the state starts at initial_v_idx

    //c2g/from (t,s,v)
    float cost = host_c2g[0*(host_opt.S_SZ*host_opt.V_SZ) + 0*host_opt.V_SZ + initial_v_idx];
    std::cout << "Optimal cost = " << cost << std::endl;

    std::cout << "optimal speed prof: ";
    std::cout << "(" << s_options[0] << ", " << v_options[initial_v_idx] << "); ";
    int idx_nxt = host_from[initial_v_idx];
    for(int t_idx=1; t_idx<n_times; ++t_idx) {
        //unwind index
        int s_idx = idx_nxt/host_opt.V_SZ;
        int v_idx = idx_nxt-(s_idx*host_opt.V_SZ);
        std::cout << "(" << s_options[s_idx] << ", " << v_options[v_idx] << "); ";
        idx_nxt   = host_from[t_idx*(host_opt.S_SZ*host_opt.V_SZ) + s_idx*host_opt.V_SZ + v_idx];
    }
    std::cout << std::endl;

    //cleanup
    delete[] host_c2g;
    delete[] host_from;

    auto done = std::chrono::high_resolution_clock::now();
    std::cout << "Backtrack time = " << std::chrono::duration_cast<std::chrono::milliseconds>(done-started).count() << " ms " << std::endl;

    cudaEventRecord(start);

    checkCudaErrors( cudaFree(dev_opt_ptr));
    checkCudaErrors( cudaFree(dev_s_opt));
    checkCudaErrors( cudaFree(dev_v_opt));
    checkCudaErrors( cudaFree(dev_a_opt));
    checkCudaErrors( cudaFree(dev_from));
    checkCudaErrors( cudaFree(dev_c2g));

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milli, start, stop);
    std::cout << "Cleanup device memory in " << milli << " ms " << std::endl;

    return 0;
}

