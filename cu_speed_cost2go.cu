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
#include "dp_utils.h"

/*
 * We need to exploit the problem structure to make it fast!
 *
 * For any specific speed, we cannot reach the same (s,v) pair, so
 * we can start a bunch of threads in parallell (number of s values x number of a values)
 *
 *
 *
 */



float const COST_INFEASIBLE = 999999.9f;

__host__ __device__ int get_c2g_idx(int t_idx, int s_idx, int v_idx,
                                    move_options const * const move_opt)
{
    return t_idx*(move_opt->V_SZ*move_opt->S_SZ) +
            v_idx*(move_opt->S_SZ) +
            s_idx;
}

__host__ __device__  float norm_cdf(float x)
{
    // constants
    float a1 =  0.254829592;
    float a2 = -0.284496736;
    float a3 =  1.421413741;
    float a4 = -1.453152027;
    float a5 =  1.061405429;
    float p  =  0.3275911;

    // Save the sign of x
    int sign = 1;
    if (x < 0)
        sign = -1;
    x = fabs(x)/sqrt(2.0);

    // A&S formula 7.1.26
    float t = 1.0/(1.0 + p*x);
    float y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*exp(-x*x);

    return 0.5*(1.0 + sign*y);
}

float call_norm_cdf(float x) {
    return norm_cdf(x);
}


__global__ void make_move(float * const c2g,
                          int   * const from,
                          int const v_idx,
                          int const t_idx,
                          move_options const * const move_opt)
{
    /* Memory access pattern

      Threads will have consecutive s-values

      Let's organize c2g and from like:

      t x v x s

     */
    int s_idx = blockIdx.x*blockDim.x  + threadIdx.x;
    int a_idx = blockIdx.y*blockDim.y  + threadIdx.y;

    float const v_end = move_opt->v_opt[v_idx];
    float const s_end = move_opt->s_opt[s_idx];

    if(s_idx < move_opt->S_SZ && a_idx < move_opt->A_SZ) {
        //Previous state!
        //s_start = s_end - v_start*Dt - a*Dt*Dt/2
        //v_start = v_end - a*DT
        float v_start = move_opt->v_opt[v_idx] - move_opt->a_opt[a_idx];
        float s_start = move_opt->s_opt[s_idx] - v_start - move_opt->a_opt[a_idx]/2;

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


            //Cost for being close to a Normally distriubted obstacle
            // @ s=40,sigma=2.0
            float const d_mean   = s_end-40.0f;
            float const d_sigma  = 2.0f;
            // cost ~ how much probabily mass is within 2.0 from 0.0
            // int_{-inf}^2 - inf_{-inf}^-2
            delta_cost += norm_cdf((2.0f-d_mean)/d_sigma) - norm_cdf((-2.0f-d_mean)/d_sigma);



            //Hard coded obstacle... (at t=3, s=[20,30] is in collision)
            if(t_idx == 3 && s_end >= 20.0 && s_end <= 30.0) {
                delta_cost = COST_INFEASIBLE;
            }


            //c2g (t,v,s)
            int c2g_idx_curr  = get_c2g_idx(t_idx,s_idx,v_idx,move_opt);
            int c2g_idx_start = get_c2g_idx(t_idx-1,s_start_idx,v_start_idx,move_opt);


            if( (delta_cost + c2g[c2g_idx_curr]) < c2g[c2g_idx_start]) {


                c2g[c2g_idx_start]  = delta_cost + c2g[c2g_idx_curr];
                from[c2g_idx_start] = s_idx*move_opt->V_SZ + v_idx;
            }

        }
    }
}

extern
int init_card(int argc, char const **argv) {
    //inline int findCudaDevice(int argc, const char **argv)
    return findCudaDevice(argc, argv);
}


extern prob_mem setup_memory(std::vector<float> const & a_options,
                             std::vector<float> const & v_options,
                             std::vector<float> const & s_options,
                             int n_times)
{
    ///@todo c2g doesn't need to include more than two timesteps, and we can swap buffers
    //allocate and set data

    prob_mem p_mem;

    //memory allocation really only has to happen once, then we can re-run over and over
    //as long as we re-set the memory....
    //What's so strange is that we don't get any cost for this the second time...
    //even though we delete and do Malloc again...

    //t x v x s
    int sz = n_times*v_options.size()*s_options.size();
    checkCudaErrors( cudaMalloc((void **)&p_mem.dev_c2g,sz*sizeof(float)) );
    checkCudaErrors( cudaMalloc((void **)&p_mem.dev_from,sz*sizeof(int)) );

    checkCudaErrors( cudaMalloc((void **)&p_mem.dev_a_opt,(a_options.size())*sizeof(float)) );
    checkCudaErrors( cudaMemcpy(p_mem.dev_a_opt,a_options.data(),a_options.size()*sizeof(float),cudaMemcpyHostToDevice) );

    checkCudaErrors( cudaMalloc((void **)&p_mem.dev_v_opt,(v_options.size())*sizeof(float)) );
    checkCudaErrors( cudaMemcpy(p_mem.dev_v_opt,v_options.data(),v_options.size()*sizeof(float),cudaMemcpyHostToDevice) );

    checkCudaErrors( cudaMalloc((void **)&p_mem.dev_s_opt,(s_options.size())*sizeof(float)) );
    checkCudaErrors( cudaMemcpy(p_mem.dev_s_opt,s_options.data(),s_options.size()*sizeof(float),cudaMemcpyHostToDevice) );

    //Struct on device
    move_options   host_opt;
    move_options * dev_opt_ptr;
    cudaMalloc((void**)&dev_opt_ptr,sizeof(move_options));
    //raw ptrs
    host_opt.a_opt = p_mem.dev_a_opt;
    host_opt.v_opt = p_mem.dev_v_opt;
    host_opt.s_opt = p_mem.dev_s_opt;
    host_opt.A_SZ = a_options.size();
    host_opt.S_SZ = s_options.size();
    host_opt.V_SZ = v_options.size();
    //copy data to device
    checkCudaErrors( cudaMemcpy(dev_opt_ptr,&host_opt,sizeof(move_options),cudaMemcpyHostToDevice) );

    p_mem.dev_opt_ptr = dev_opt_ptr;
    p_mem.host_opt    = host_opt;
    return p_mem;

}

extern void clear_memory(prob_mem & p_mem) {
    checkCudaErrors( cudaFree(p_mem.dev_opt_ptr));
    checkCudaErrors( cudaFree(p_mem.dev_s_opt));
    checkCudaErrors( cudaFree(p_mem.dev_v_opt));
    checkCudaErrors( cudaFree(p_mem.dev_a_opt));
    checkCudaErrors( cudaFree(p_mem.dev_from));
    checkCudaErrors( cudaFree(p_mem.dev_c2g));
}


extern
int speed_dp(prob_mem & p_mem,
             std::vector<float> const & a_options,
             std::vector<float> const & v_options,
             std::vector<float> const & s_options,
             int n_times,
             int initial_v_idx,
             bool print) {

    if(s_options.size() > 1024 || a_options.size() > 1024) {
        std::cerr << "Maximum dimensions exceeded" << std::endl;
        return -1;
    }

    if(initial_v_idx < 0 || initial_v_idx >= v_options.size()) {
        std::cerr << "Invalid inital speed " << std::endl;
        return -1;
    }


    int sz = n_times*v_options.size()*s_options.size();

    //At last time all costs are zero, otherwise, initialize to COST_INFEASIBLE..
    thrust::device_ptr<float> dev_ptr(p_mem.dev_c2g);
    thrust::fill(dev_ptr, dev_ptr + sz, COST_INFEASIBLE);
    thrust::fill(dev_ptr+(n_times-1)*(v_options.size()*s_options.size()),dev_ptr+sz,0.0);

    checkCudaErrors( cudaMemset(p_mem.dev_from,-1,sz*sizeof(int)) );


    ///@todo this can probably be done better...

    //about 6-8 ms with this
    dim3 block_dim(p_mem.host_opt.S_SZ,1);
    dim3 grid_dim(1,p_mem.host_opt.A_SZ);

    //Let's try with 512 threads/block instead
    //Pretty much the same...
    //dim3 block_dim(512,1);
    //dim3 grid_dim(2,host_opt.A_SZ);

    //same..
    //dim3 block_dim(512,2);
    //dim3 grid_dim(2,ceil(host_opt.A_SZ/2.0));


    //These loops have to be done in sequence
    for(int t_idx=n_times-1; t_idx>0; t_idx--) {
        for(int v_idx=0; v_idx<p_mem.host_opt.V_SZ; ++v_idx) {
            make_move<<<grid_dim,block_dim>>>(p_mem.dev_c2g,p_mem.dev_from,v_idx,t_idx,p_mem.dev_opt_ptr);
        }
    }

    //Backtrack
    float * host_c2g  = new float[sz];
    int   * host_from = new int[sz];

    cudaMemcpy(host_c2g,p_mem.dev_c2g,sz*sizeof(float),cudaMemcpyDeviceToHost);
    cudaMemcpy(host_from,p_mem.dev_from,sz*sizeof(int),cudaMemcpyDeviceToHost);


    //by definition we start at s=0
    //find the state starts at initial_v_idx

    //c2g/from (t,s,v)
    int c2g_idx = get_c2g_idx(0,0,initial_v_idx,&p_mem.host_opt);
    float cost = host_c2g[c2g_idx];
    if(print) std::cout << "Optimal cost = " << cost << std::endl;
    if(print) std::cout << "optimal speed prof: ";
    if(print) std::cout << "(" << s_options[0] << ", " << v_options[initial_v_idx] << "); ";
    int idx_nxt = host_from[c2g_idx];
    //std::cout << "idx_nxt = " << idx_nxt << " ";
    for(int t_idx=1; t_idx<n_times; ++t_idx) {
        //unwind index
        int s_idx = idx_nxt/p_mem.host_opt.V_SZ;
        int v_idx = idx_nxt-(s_idx*p_mem.host_opt.V_SZ);
        if(print) std::cout << "(" << s_options[s_idx] << ", " << v_options[v_idx] << "); ";
        c2g_idx   = get_c2g_idx(t_idx,s_idx,v_idx,&p_mem.host_opt);
        idx_nxt   = host_from[c2g_idx];
        //std::cout << "idx_nxt = " << idx_nxt << " ";
    }
    if(print) std::cout << std::endl;

    //cleanup
    delete[] host_c2g;
    delete[] host_from;

    //for printing
    //cudaDeviceReset();

    return 0;
}



