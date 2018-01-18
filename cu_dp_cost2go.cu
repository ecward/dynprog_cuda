#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <vector>
#include <istream>
#include <algorithm>
#include <chrono>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>

#include "helper_cuda.h"

/* We start from last timestep and go backwards.
 *
 * Each state in the current timestep has a set of moves
 * that get it to the previous timestep with a DELTA_COST
 * added to the cost-to-go (0 for the last timestep).
 *
 * We can process all the states fur the current timestep,
 * and all the moves in parallell, if we have an array
 * for each state in previous timesstep (of size num states)
 * which stores the cost, and from which state we came from.
 *
 * We then need to do a reduction on this array, for each state
 * in previous timestep before we can start the procedure again.
 *
 *
 *       t-1                                    t
 *
 *       c2g_t-1_i  <- min <-- + c_i,i   --  c2g_t_i
 *                        \
 *       c2g_t-1_i+1       \__ + c_i+1,i --  c2g_t_i+1
 *
 *       t-1                                    t
 *
 *       c2g_t-1_i      [ c_i,i   + c2g_t_i     <--
 *                                              <--
 *                        c_i+1,i + c2g_t_i+1 ] <--
 *
 *
 *
 * How moves are defined:
 *
 * if s_{t-1,i} -> s_{t,j} is possible it has cost < MAX_MOVE_COST
 * otherwise                           it has cost = INFEASIBLE_COST
 *
 * What moves are possible are problem specific
 *
 * -> Problem is to find the speed profile of a vehicle
 *    with constraints on "s" for specific times (collisions with
 *    other vehicles), constraints on min/max_speed, min/max_acc
 *
 *    States encode s,v pairs.
 *    Let's use a = -2.5,-1.25,0,1.25,2.5
 *    and t_max   = 10
 *    Delta t     = 1.0
 *
 *    then we have fixed number of possible
 *    speeds, positions based on max_speed (min_speed = 0)
 */
//std::vector<float> const a_options = {-2.5,  -1.25,  0.0,    1.25,  2.5};
//std::vector<float> const v_options = {0.0, 1.25, 2.5, 3.75, 5.0, 6.25, 7.5, 8.75, 10.0, 11.25, 12.5, 13.75, 15.0, 16.25, 17.5, 18.75, 20.0, 21.25, 22.5, 23.75, 25.0, 26.25, 27.5, 28.75, 30.0, 31.25, 32.5, 33.75, 35.0};
//std::vector<float> const s_options = {0.0, 0.625, 1.25, 1.875, 2.5, 3.125, 3.75, 4.375, 5.0, 5.625, 6.25, 6.875, 7.5, 8.125, 8.75, 9.375, 10.0, 10.625, 11.25, 11.875, 12.5, 13.125, 13.75, 14.375, 15.0, 15.625, 16.25, 16.875, 17.5, 18.125, 18.75, 19.375, 20.0, 20.625, 21.25, 21.875, 22.5, 23.125, 23.75, 24.375, 25.0, 25.625, 26.25, 26.875, 27.5, 28.125, 28.75, 29.375, 30.0, 30.625, 31.25, 31.875, 32.5, 33.125, 33.75, 34.375, 35.0, 35.625, 36.25, 36.875, 37.5, 38.125, 38.75, 39.375, 40.0, 40.625, 41.25, 41.875, 42.5, 43.125, 43.75, 44.375, 45.0, 45.625, 46.25, 46.875, 47.5, 48.125, 48.75, 49.375, 50.0, 50.625, 51.25, 51.875, 52.5, 53.125, 53.75, 54.375, 55.0, 55.625, 56.25, 56.875, 57.5, 58.125, 58.75, 59.375, 60.0, 60.625, 61.25, 61.875, 62.5, 63.125, 63.75, 64.375, 65.0, 65.625, 66.25, 66.875, 67.5, 68.125, 68.75, 69.375, 70.0, 70.625, 71.25, 71.875, 72.5, 73.125, 73.75, 74.375, 75.0, 75.625, 76.25, 76.875, 77.5, 78.125, 78.75, 79.375, 80.0, 80.625, 81.25, 81.875, 82.5, 83.125, 83.75, 84.375, 85.0, 85.625, 86.25, 86.875, 87.5, 88.125, 88.75, 89.375, 90.0, 90.625, 91.25, 91.875, 92.5, 93.125, 93.75, 94.375, 95.0, 95.625, 96.25, 96.875, 97.5, 98.125, 98.75, 99.375, 100.0, 100.625, 101.25, 101.875, 102.5, 103.125, 103.75, 104.375, 105.0, 105.625, 106.25, 106.875, 107.5, 108.125, 108.75, 109.375, 110.0, 110.625, 111.25, 111.875, 112.5, 113.125, 113.75, 114.375, 115.0, 115.625, 116.25, 116.875, 117.5, 118.125, 118.75, 119.375, 120.0, 120.625, 121.25, 121.875, 122.5, 123.125, 123.75, 124.375, 125.0, 125.625, 126.25, 126.875, 127.5, 128.125, 128.75, 129.375, 130.0, 130.625, 131.25, 131.875, 132.5, 133.125, 133.75, 134.375, 135.0, 135.625, 136.25, 136.875, 137.5, 138.125, 138.75, 139.375, 140.0, 140.625, 141.25, 141.875, 142.5, 143.125, 143.75, 144.375, 145.0, 145.625, 146.25, 146.875, 147.5, 148.125, 148.75, 149.375, 150.0, 150.625, 151.25, 151.875, 152.5, 153.125, 153.75, 154.375, 155.0, 155.625, 156.25, 156.875, 157.5, 158.125, 158.75, 159.375, 160.0, 160.625, 161.25, 161.875, 162.5, 163.125, 163.75, 164.375, 165.0, 165.625, 166.25, 166.875, 167.5, 168.125, 168.75, 169.375, 170.0, 170.625, 171.25, 171.875, 172.5, 173.125, 173.75, 174.375, 175.0, 175.625, 176.25, 176.875, 177.5, 178.125, 178.75, 179.375, 180.0, 180.625, 181.25, 181.875, 182.5, 183.125, 183.75, 184.375, 185.0, 185.625, 186.25, 186.875, 187.5, 188.125, 188.75, 189.375, 190.0, 190.625, 191.25, 191.875, 192.5, 193.125, 193.75, 194.375, 195.0, 195.625, 196.25, 196.875, 197.5, 198.125, 198.75, 199.375, 200.0, 200.625, 201.25, 201.875, 202.5, 203.125, 203.75, 204.375, 205.0, 205.625, 206.25, 206.875, 207.5, 208.125, 208.75, 209.375, 210.0, 210.625, 211.25, 211.875, 212.5, 213.125, 213.75, 214.375, 215.0, 215.625, 216.25, 216.875, 217.5, 218.125, 218.75, 219.375, 220.0, 220.625, 221.25, 221.875, 222.5, 223.125, 223.75, 224.375, 225.0, 225.625, 226.25, 226.875, 227.5, 228.125, 228.75, 229.375, 230.0, 230.625, 231.25, 231.875, 232.5, 233.125, 233.75, 234.375, 235.0, 235.625, 236.25, 236.875, 237.5, 238.125, 238.75, 239.375, 240.0, 240.625, 241.25, 241.875, 242.5, 243.125, 243.75, 244.375, 245.0, 245.625, 246.25, 246.875, 247.5, 248.125, 248.75, 249.375, 250.0, 250.625, 251.25, 251.875, 252.5, 253.125, 253.75, 254.375, 255.0, 255.625, 256.25, 256.875, 257.5, 258.125, 258.75, 259.375, 260.0, 260.625, 261.25, 261.875, 262.5, 263.125, 263.75, 264.375, 265.0, 265.625, 266.25, 266.875, 267.5, 268.125, 268.75, 269.375, 270.0, 270.625, 271.25, 271.875, 272.5, 273.125, 273.75, 274.375, 275.0, 275.625, 276.25, 276.875, 277.5, 278.125, 278.75, 279.375, 280.0, 280.625, 281.25, 281.875, 282.5, 283.125, 283.75, 284.375, 285.0, 285.625, 286.25, 286.875, 287.5, 288.125, 288.75, 289.375, 290.0, 290.625, 291.25, 291.875, 292.5, 293.125, 293.75, 294.375, 295.0, 295.625, 296.25, 296.875, 297.5, 298.125, 298.75, 299.375, 300.0, 300.625, 301.25, 301.875, 302.5, 303.125, 303.75, 304.375, 305.0, 305.625, 306.25, 306.875, 307.5, 308.125, 308.75, 309.375, 310.0, 310.625, 311.25, 311.875, 312.5, 313.125, 313.75, 314.375, 315.0, 315.625, 316.25, 316.875, 317.5, 318.125, 318.75, 319.375, 320.0, 320.625, 321.25, 321.875, 322.5, 323.125, 323.75, 324.375, 325.0, 325.625, 326.25, 326.875, 327.5, 328.125, 328.75, 329.375, 330.0, 330.625, 331.25, 331.875, 332.5, 333.125, 333.75, 334.375, 335.0, 335.625, 336.25, 336.875, 337.5, 338.125, 338.75, 339.375, 340.0, 340.625, 341.25, 341.875, 342.5, 343.125, 343.75, 344.375, 345.0, 345.625, 346.25, 346.875, 347.5, 348.125, 348.75, 349.375, 350.0};

//This will give a tmp array of about 180 MB... (num_states**2 = (V*S)*(V*S))
//But we COULD limit our parallelism by only considering
//a fixed number of previous states, K, before doing the reduction:
//tmp arr size = V*S*K*sizeof(float)


int const WARP_SIZE = 32; //Let's just assume this will always be true...

float const COST_INFEASIBLE = 999999.9f;

struct move_options {
    float * a_opt;
    float * v_opt;
    float * s_opt;
    int A_SZ;
    int V_SZ;
    int S_SZ;
};

///@todo think about memory access patterns...
/// this is SUPER SLOW
__global__ void reset_tmp_arr(float * const tmp_arr,
                              move_options const * const move_opt) {
    //int n_states = s_options.size()*v_options.size();
    //tmp array is n_states X n_states
    //             S_prevxV_prevxSxV

    //we should get these from thread indexing...
    int s_idx      = blockIdx.x*blockDim.x  + threadIdx.x;
    int v_idx      = blockIdx.y*blockDim.y  + threadIdx.y;
    int s_idx_prev = blockIdx.z*blockDim.z  + threadIdx.z;


    //let each thread reset all cost for previous speeds
    int idx_curr    = s_idx*move_opt->V_SZ       + v_idx;

    if(s_idx < move_opt->S_SZ && v_idx < move_opt->V_SZ && s_idx_prev < move_opt->S_SZ) {

        for(int v_idx_prev=0; v_idx_prev<move_opt->V_SZ; ++v_idx_prev) {
            int idx_prev    = s_idx_prev*move_opt->V_SZ + v_idx_prev;

            int tmp_arr_idx = idx_prev*(move_opt->S_SZ*move_opt->V_SZ) + idx_curr;
            tmp_arr[tmp_arr_idx] = COST_INFEASIBLE;
        }
    } else {
        //NOP
    }
}

//Working with huge tmp array is too SLOW!
//Need to use __shared__ memory for the blocks or something...
__global__ void make_move_naive(float * const tmp_arr,
                          float const * const c2g_arr,
                          move_options const * const move_opt) {


    int s_idx = blockIdx.x*blockDim.x  + threadIdx.x;
    int v_idx = blockIdx.y*blockDim.y  + threadIdx.y;
    int a_idx = blockIdx.z*blockDim.z  + threadIdx.z;

    if(s_idx < move_opt->S_SZ && v_idx < move_opt->V_SZ && a_idx < move_opt->A_SZ) {

        //Previous state!
        //s_start = s_end - v*Dt - a*Dt*Dt/2
        //v_start = v_end - a*DT
        float s_start = move_opt->s_opt[s_idx] - move_opt->v_opt[v_idx] - move_opt->a_opt[a_idx]/2;
        float v_start = move_opt->v_opt[v_idx] - move_opt->a_opt[a_idx];

        //figure out idx in tmp_arr (like row major matrix with s_idx as row and v_idx as col)
        float const delta_s   = move_opt->s_opt[1]-move_opt->s_opt[0];
        float const delta_v   = move_opt->v_opt[1]-move_opt->v_opt[0];
        int s_start_idx = s_start/delta_s;
        int v_start_idx = v_start/delta_v;

        //Move from (i,t-1) -> (j,t)
        int idx_prev    = s_start_idx*move_opt->V_SZ + v_start_idx;
        int idx_curr    = s_idx*move_opt->V_SZ       + v_idx;
        //tmp arr indexed by prev_state,curr_state
        int tmp_arr_idx = idx_prev*(move_opt->S_SZ*move_opt->V_SZ) + idx_curr;
        //Negative speeds/Negative positions not allowed
        if(v_start_idx < 0 || s_start_idx < 0 ||
                v_start_idx >= move_opt->V_SZ || s_start_idx >= move_opt->S_SZ) {
            //Nop
        } else {

            //Just cost for acc...
            tmp_arr[tmp_arr_idx] =
                    0.5*move_opt->a_opt[a_idx]*move_opt->a_opt[a_idx] + c2g_arr[idx_curr];
        }
    }
}


//reduction on tmp_arr to get minimum
__global__ void best_cost_for_state(int s_idx_curr,
                                    int v_idx_curr,
                                    float const * const tmp_arr,
                                    float * const c2g_arr,
                                    move_options const * const move_opt) {

    extern  __shared__  float temp[];

    //reduction on ONE curr state (min over all prev states)
    //tmp array (idx_prev,idx_nxt)
    int idx_curr = s_idx_curr*move_opt->V_SZ + v_idx_curr;
    int idx_prev = threadIdx.x + blockIdx.x*blockDim.x;
    int idx      = idx_prev*(move_opt->S_SZ*move_opt->V_SZ) + idx_curr;


    //tmp_arr must have idx_curr in a valid range that is a power of 2!!!
    //pad with infeasible!!!

    //warp_min will contain the min of all threads in this warp after suffle
    float warp_min = COST_INFEASIBLE;
    if(idx_prev < move_opt->V_SZ*move_opt->S_SZ) {
        warp_min = tmp_arr[idx];
    }

    for(int mask = WARP_SIZE/2; mask>0; mask >>=1) {
        //we get data from thread with lane_id = my_lane_id XOR laneMask
        float shfl_data = __shfl_xor(warp_min,mask);
        warp_min = fminf(warp_min,shfl_data);
    }

    //now all threads have the partial min of their warp
    //let the first thread in the warp fill in the shared memory for the block
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    if(lane_id == 0) {
        temp[warp_id] = warp_min;
    }
    __syncthreads();


    if(warp_id == 0) {
        float min = temp[lane_id];
        for(int mask = WARP_SIZE/2; mask>0; mask >>=1) {
            float shfl_data = __shfl_xor(min,mask);
            min = fminf(min,shfl_data);
        }
        if(threadIdx.x == 0) {
            //fill in output
            c2g_arr[idx_curr] = min;
        }
    }

}

extern
int simple_dp(std::vector<float> const & a_options,
              std::vector<float> const & v_options,
              std::vector<float> const & s_options,
              int n_times) {

    // initialise CUDA timing
    float milli;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    ///@todo also add backtrace infromation, not just cost two go

    int n_states = s_options.size()*v_options.size();
    //allocate memory

    cudaEventRecord(start);

    //SxT table
    float * dev_c2g;
    checkCudaErrors( cudaMalloc((void **)&dev_c2g,(n_states*n_times)*sizeof(float)) );
    //SxS temporary arrays
    float * dev_tmp_arr;
    checkCudaErrors( cudaMalloc((void **)&dev_tmp_arr,(n_states*n_states)*sizeof(float)) );

    //This gives errors for some reason :(
    /*
    thrust::device_vector<float> D_a_opt(a_options.begin(),a_options.end());
    thrust::device_vector<float> D_v_opt(v_options.begin(),v_options.end());
    thrust::device_vector<float> D_s_opt(s_options.begin(),s_options.end());
    */
    float * dev_a_opt;
    checkCudaErrors( cudaMalloc((void **)&dev_a_opt,(a_options.size())*sizeof(float)) );
    checkCudaErrors( cudaMemcpy(dev_a_opt,a_options.data(),a_options.size()*sizeof(float),cudaMemcpyHostToDevice) );

    float * dev_v_opt;
    checkCudaErrors( cudaMalloc((void **)&dev_v_opt,(v_options.size())*sizeof(float)) );
    checkCudaErrors( cudaMemcpy(dev_v_opt,v_options.data(),v_options.size()*sizeof(float),cudaMemcpyHostToDevice) );

    float * dev_s_opt;
    checkCudaErrors( cudaMalloc((void **)&dev_s_opt,(s_options.size())*sizeof(float)) );
    checkCudaErrors( cudaMemcpy(dev_s_opt,s_options.data(),s_options.size()*sizeof(float),cudaMemcpyHostToDevice) );





    //Struct on device...
    move_options   host_opt;
    move_options * dev_opt_ptr;
    cudaMalloc((void**)&dev_opt_ptr,sizeof(move_options));
    //raw ptrs
    host_opt.a_opt = dev_a_opt;//thrust::raw_pointer_cast(D_a_opt.data());
    host_opt.v_opt = dev_v_opt;//thrust::raw_pointer_cast(D_v_opt.data());
    host_opt.s_opt = dev_s_opt;//thrust::raw_pointer_cast(D_s_opt.data());
    host_opt.A_SZ = a_options.size();
    host_opt.S_SZ = s_options.size();
    host_opt.V_SZ = v_options.size();
    //copy data to host
    checkCudaErrors( cudaMemcpy(dev_opt_ptr,&host_opt,sizeof(move_options),cudaMemcpyHostToDevice) );

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milli, start, stop);

    std::cout << "Device memory allocation in " << milli << " ms " << std::endl;

    /*
     dim3 block_dim(128,1,1);
     dim3 grid_dim(10,1,1);
     kernel<<<grid_dim,block_dim>>>(...);

      __device__
      int getGlobalIdx_3D_3D(){
          int blockId = blockIdx.x + blockIdx.y * gridDim.x
                        + gridDim.x * gridDim.y * blockIdx.z;
          int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
                        + (threadIdx.z * (blockDim.x * blockDim.y))
                        + (threadIdx.y * blockDim.x) + threadIdx.x;
          return threadId;
      }

      //Let's use the following convention
      __device__
      int global_x = blockIdx.x*blockDim.x  + threadIdx.x
      int global_y = blockIdx.y*blockDim.y  + threadIdx.y
      int global_z = blockIdx.z*blockDim.z  + threadIdx.z
    */

    std::cout << "S_SZ = " << host_opt.S_SZ << " V_SZ = " << host_opt.V_SZ << std::endl;

    //three dimensional indexing...
    //   s_idx x  v_idx  x s_idx_prev
    //
    // Each block can have at most 1024 threads

    if(host_opt.S_SZ > 1024 || host_opt.V_SZ > 1024 || host_opt.A_SZ > 1024) {
        std::cerr << "Maximum dimensions exceeded" << std::endl;
        return -1;
    }

    /* This wont work...
    int block_d0_sz = host_opt_ptr.S_SZ;
    int block_d1_sz = std::min(host_opt_ptr.V_SZ,(int)ceil(1024.0/block_d0_sz));
    int block_d2_sz = std::min(host_opt_ptr.S_SZ,(int)ceil(1024.0/(block_d0_sz*block_d1_sz)));
    */
    int block_d0_sz = host_opt.S_SZ;  //host_opt_ptr.S_SZ/2;
    int block_d1_sz = 1;                  //1024/block_d0_sz;
    int block_d2_sz = 1;

    std::cout << "block dims = " << block_d0_sz << " x " << block_d1_sz << " x " << block_d2_sz
              << " (total = " << block_d0_sz*block_d1_sz*block_d2_sz << ") " << std::endl;

    int grid_d0_sz = ceil(1.0*host_opt.S_SZ/block_d0_sz);
    int grid_d1_sz = ceil(1.0*host_opt.V_SZ/block_d1_sz);
    int grid_d2_sz = ceil(1.0*host_opt.S_SZ/block_d2_sz);

    std::cout << "grid dims = " << grid_d0_sz << " x " << grid_d1_sz << " x " << grid_d2_sz
              << " total = " << grid_d0_sz*grid_d1_sz*grid_d2_sz << " x " << block_d0_sz*block_d1_sz*block_d2_sz
              << " = " << grid_d0_sz*grid_d1_sz*grid_d2_sz*block_d0_sz*block_d1_sz*block_d2_sz << std::endl;
    std::cout << "for problem of dimensions = " << host_opt.S_SZ << " x " << host_opt.V_SZ << " x " << host_opt.S_SZ
              << " = " << host_opt.S_SZ*host_opt.V_SZ*host_opt.S_SZ << std::endl;

    dim3 grid_dim(grid_d0_sz,grid_d1_sz,grid_d2_sz);
    dim3 block_dim(block_d0_sz,block_d1_sz,block_d2_sz);

    cudaEventRecord(start);

    reset_tmp_arr<<<grid_dim,block_dim>>>(dev_tmp_arr,dev_opt_ptr);
    getLastCudaError("reset_tmp_arr execution failed\n");

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milli, start, stop);
    std::cout << "Reset tmp arr in " << milli << " ms " << std::endl;

    //Let's do it with thrust instead!
//    cudaEventRecord(start);
//    thrust::device_vector<float> dev_tmp_vec(n_states*n_states);
//    thrust::fill(dev_tmp_vec.begin(),dev_tmp_vec.end(),COST_INFEASIBLE);
//    cudaEventRecord(stop);
//    cudaEventSynchronize(stop);
//    cudaEventElapsedTime(&milli, start, stop);
//    std::cout << "Reset tmp arr (thrust) in " << milli << " ms " << std::endl;

    ///@todo just checking it worked!
    /*
    float * host_tmp_arr = new float[(n_states*n_states)];
    cudaMemcpy(host_tmp_arr,dev_tmp_arr,(n_states*n_states)*sizeof(float),cudaMemcpyDeviceToHost);
    int err_cnt   = 0;
    float max_err = 0.0;
    for(int i=0; i<n_states; ++i) {
        for(int j=0; j<n_states; ++j) {
            float err = fabs(host_tmp_arr[n_states*i + j]-COST_INFEASIBLE);
            if(err > 1.0e-10) {
                err_cnt++;
            }
            if(err > max_err) {
                max_err = err;
            }
        }
    }
    std::cout << "err_cnt = " << err_cnt << " max_err = " << max_err << std::endl;

    std::vector<float> host_tmp_vec(dev_tmp_vec.size());
    thrust::copy(dev_tmp_vec.begin(),dev_tmp_vec.end(),host_tmp_vec.begin());
    err_cnt = 0;
    max_err = 0.0f;
    for(int i=0; i<host_tmp_vec.size(); ++i) {
        float err = fabs(host_tmp_vec[i]-COST_INFEASIBLE);
        if(err > 1.0e-10) {
            err_cnt++;
        }
        if(err > max_err) {
            max_err = err;
        }
    }
    std::cout << "err_cnt = " << err_cnt << " max_err = " << max_err << std::endl;

    delete[] host_tmp_arr;
    */

    block_d0_sz = host_opt.S_SZ;  //host_opt_ptr.S_SZ/2;
    block_d1_sz = 1;                  //1024/block_d0_sz;
    block_d2_sz = 1;

    std::cout << "block dims = " << block_d0_sz << " x " << block_d1_sz << " x " << block_d2_sz
              << " (total = " << block_d0_sz*block_d1_sz*block_d2_sz << ") " << std::endl;

    grid_d0_sz = ceil(1.0*host_opt.S_SZ/block_d0_sz);
    grid_d1_sz = ceil(1.0*host_opt.V_SZ/block_d1_sz);
    grid_d2_sz = ceil(1.0*host_opt.A_SZ/block_d2_sz);

    std::cout << "grid dims = " << grid_d0_sz << " x " << grid_d1_sz << " x " << grid_d2_sz
              << " total = " << grid_d0_sz*grid_d1_sz*grid_d2_sz << " x " << block_d0_sz*block_d1_sz*block_d2_sz
              << " = " << grid_d0_sz*grid_d1_sz*grid_d2_sz*block_d0_sz*block_d1_sz*block_d2_sz << std::endl;
    std::cout << "for problem of dimensions = " << host_opt.S_SZ << " x " << host_opt.V_SZ << " x " << host_opt.A_SZ
              << " = " << host_opt.S_SZ*host_opt.V_SZ*host_opt.A_SZ << std::endl;

    dim3 grid_dim_search(grid_d0_sz,grid_d1_sz,grid_d2_sz);
    dim3 block_dim_search(block_d0_sz,block_d1_sz,block_d2_sz);

    cudaEventRecord(start);
    make_move_naive<<<grid_dim_search,block_dim_search>>>(dev_tmp_arr,
                                                    dev_c2g,
                                                    dev_opt_ptr);
    getLastCudaError("make_move execution failed\n");

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milli, start, stop);
    std::cout << "Make move in " << milli << " ms " << std::endl;


    int shared_mem_size = sizeof(float) * WARP_SIZE;
    int num_threads = 1024;
    int num_blocks  = ceil( (host_opt.V_SZ*host_opt.S_SZ)/1024.0 );

    std::cout << "num_t x num_blocks = " << num_threads << " x " << num_blocks << " = "
              << num_threads*num_blocks << " for problem of size = " << host_opt.V_SZ*host_opt.S_SZ << std::endl;



    /*
    auto started = std::chrono::high_resolution_clock::now();

    std::vector<float> runtimes;
    for(int s_idx_curr=0; s_idx_curr<host_opt.S_SZ; ++s_idx_curr) {
        for(int v_idx_curr=0; v_idx_curr<host_opt.V_SZ; ++v_idx_curr) {
            cudaEventRecord(start);
            best_cost_for_state<<<num_blocks,num_threads,shared_mem_size>>>(s_idx_curr,v_idx_curr,
                                                                            dev_tmp_arr,dev_c2g,dev_opt_ptr);
            getLastCudaError("best_cost_for_state execution failed\n");
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&milli, start, stop);
            runtimes.push_back(milli);
        }
    }
    //Max runtime
    std::cout << "max runtime = " << *std::max_element(runtimes.begin(),runtimes.end()) << " ms " << std::endl;
    //Total
    std::cout << "Total runtime = " <<  std::accumulate(runtimes.begin(), runtimes.end(), 0.0) << " ms " << std::endl;

    auto done = std::chrono::high_resolution_clock::now();
    std::cout << "Total time for loops... = " << std::chrono::duration_cast<std::chrono::milliseconds>(done-started).count() << " ms " << std::endl;
    */
    //about 458 ms....

    auto started = std::chrono::high_resolution_clock::now();
    //Let's create a bunch of streams....
    std::vector<cudaStream_t> streams(100);
    for(auto & stream : streams) {
        cudaStreamCreate(&stream);
    }
    for(int s_idx_curr=0; s_idx_curr<host_opt.S_SZ; ++s_idx_curr) {
        for(int v_idx_curr=0; v_idx_curr<host_opt.V_SZ; ++v_idx_curr) {
            size_t stream_idx = (s_idx_curr*host_opt.V_SZ+ v_idx_curr) % streams.size();
            best_cost_for_state<<<num_blocks,num_threads,shared_mem_size,streams.at(stream_idx)>>>(s_idx_curr,v_idx_curr,
                                                                                                   dev_tmp_arr,dev_c2g,dev_opt_ptr);
        }
    }
    auto done = std::chrono::high_resolution_clock::now();
    std::cout << "Total time for loops... = " << std::chrono::duration_cast<std::chrono::milliseconds>(done-started).count() << " ms " << std::endl;
    //10 streams -> 384
    //20 streams -> 355,340
    //100 streams -> 182,186,184,186
    //200 streams -> 190
    //cleanup

    cudaEventRecord(start);

    cudaFree(dev_opt_ptr);
    cudaFree(dev_s_opt);
    cudaFree(dev_v_opt);
    cudaFree(dev_a_opt);
    cudaFree(dev_tmp_arr);
    cudaFree(dev_c2g);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milli, start, stop);
    std::cout << "Cleanup device memory in " << milli << " ms " << std::endl;

    cudaDeviceReset();
    return 0;


}
