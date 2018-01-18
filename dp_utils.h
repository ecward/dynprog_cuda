#pragma once

struct move_options {
    float * a_opt;
    float * v_opt;
    float * s_opt;
    int A_SZ;
    int V_SZ;
    int S_SZ;
};

struct prob_mem {
    float * dev_c2g;   //(t,v,s)
    int   * dev_from;  //(t,v,s)

    float * dev_a_opt;
    float * dev_v_opt;
    float * dev_s_opt;

    move_options * dev_opt_ptr;
    move_options   host_opt;
};

float call_norm_cdf(float x);

