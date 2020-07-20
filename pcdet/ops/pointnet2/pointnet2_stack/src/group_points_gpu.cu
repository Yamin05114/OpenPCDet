/*
Stacked-batch-data version of point grouping, modified from the original implementation of official PointNet++ codes.
Written by Shaoshuai Shi
All Rights Reserved 2019-2020.
*/


#include <stdio.h>
#include <stdlib.h>

#include "cuda_utils.h"
#include "group_points_gpu.h"


__global__ void group_points_grad_kernel_stack(int B, int M, int C, int N, int nsample,
    const float *grad_out, const int *idx, const int *idx_batch_cnt, const int *features_batch_cnt, float *grad_features) {
    // :param grad_out: (M1 + M2 ..., C, nsample) tensor of the gradients of the output from forward
    // :param idx: (M1 + M2 ..., nsample) tensor containing the indicies of features to group with
    // :param idx_batch_cnt: (batch_size) [M1 + M2 ...] tensor containing the indicies of features to group with
    // :param features_batch_cnt: (batch_size) [N1 + N2 ...] tensor containing the indicies of features to group with
    // :return:
    //     grad_features: (N1 + N2 ..., C) gradient of the features
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int sample_idx = index % nsample;
    int C_idx = (index / nsample) % C;
    int pt_idx = (index / nsample / C);

    if (pt_idx >= M || C_idx >= C || sample_idx >= nsample) return;

    int bs_idx = 0, pt_cnt = idx_batch_cnt[0];
    for (int k = 1; k < B; k++){
        if (pt_idx < pt_cnt) break;
        pt_cnt += idx_batch_cnt[k];
        bs_idx = k;
    }

    int features_batch_start_idx = 0;
    for (int k = 0; k < bs_idx; k++) features_batch_start_idx += features_batch_cnt[k];

    grad_out += pt_idx * C * nsample + C_idx * nsample + sample_idx;
    idx += pt_idx * nsample + sample_idx;
    grad_features += (features_batch_start_idx + idx[0]) * C + C_idx;

    atomicAdd(grad_features, grad_out[0]);
}

void group_points_grad_kernel_launcher_stack(int B, int M, int C, int N, int nsample,
    const float *grad_out, const int *idx, const int *idx_batch_cnt, const int *features_batch_cnt, float *grad_features) {
    // :param grad_out: (M1 + M2 ..., C, nsample) tensor of the gradients of the output from forward
    // :param idx: (M1 + M2 ..., nsample) tensor containing the indicies of features to group with
    // :param idx_batch_cnt: (batch_size) [M1 + M2 ...] tensor containing the indicies of features to group with
    // :param features_batch_cnt: (batch_size) [N1 + N2 ...] tensor containing the indicies of features to group with
    // :return:
    //     grad_features: (N1 + N2 ..., C) gradient of the features

    cudaError_t err;
    // dim3 blocks(DIVUP(npoints * nsample, THREADS_PER_BLOCK), c, b);  // blockIdx.x(col), blockIdx.y(row)
    dim3 blocks(DIVUP(M * C * nsample, THREADS_PER_BLOCK));  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    group_points_grad_kernel_stack<<<blocks, threads>>>(B, M, C, N, nsample, grad_out, idx, idx_batch_cnt, features_batch_cnt, grad_features);

    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}


__global__ void group_points_kernel_stack(int B, int M, int C, int nsample,
    const float *features, const int *features_batch_cnt, const int *idx, const int *idx_batch_cnt, float *out) {
    // M, nsample => M, C, nsample
    // :param features: (N1 + N2 ..., C) tensor of features to group, sparse tensor跟spconv一样
    // :param features_batch_cnt: (batch_size) [N1 + N2 ...] tensor containing the indicies of features to group with
    //                            用来标记batch idx
    // :param idx: (M1 + M2 ..., nsample) tensor containing the indicies of features to group with
    //             ballquery函数的返回值，用来给每个pcl点选择voxel_center点
    // :param idx_batch_cnt: (batch_size) [M1 + M2 ...] tensor containing the indicies of features to group with，标记batch
    // :return:
    //     output: (M1 + M2, C, nsample) tensor，把c长度feature附在query上。
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int sample_idx = index % nsample;
    // index / nsample
    int C_idx = (index / nsample) % C;  // 一个thread只操作一个channel
    int pt_idx = (index / nsample / C);  // 每个点有c*nsample个元素
    
    // 上限M个点，C个channel，nsample个neighbor
    if (pt_idx >= M || C_idx >= C || sample_idx >= nsample) return;
    
    // 当前点所在batch idx
    int bs_idx = 0, pt_cnt = idx_batch_cnt[0];
    for (int k = 1; k < B; k++){
        if (pt_idx < pt_cnt) break;
        pt_cnt += idx_batch_cnt[k];
        bs_idx = k;
    }
    
    // feature在当前batch idx第一个位置
    int features_batch_start_idx = 0;
    for (int k = 0; k < bs_idx; k++) features_batch_start_idx += features_batch_cnt[k];
    features += features_batch_start_idx * C;

    idx += pt_idx * nsample + sample_idx;
    // 最后一个维度最先增长。
    int in_idx = idx[0] * C + C_idx;
    int out_idx = pt_idx * C * nsample + C_idx * nsample + sample_idx;

    out[out_idx] = features[in_idx];
}


void group_points_kernel_launcher_stack(int B, int M, int C, int nsample,
    const float *features, const int *features_batch_cnt, const int *idx, const int *idx_batch_cnt, float *out) {
    // :param features: (N1 + N2 ..., C) tensor of features to group
    // :param features_batch_cnt: (batch_size) [N1 + N2 ...] tensor containing the indicies of features to group with
    // :param idx: (M1 + M2 ..., nsample) tensor containing the indicies of features to group with
    // :param idx_batch_cnt: (batch_size) [M1 + M2 ...] tensor containing the indicies of features to group with
    // :return:
    //     output: (M1 + M2, C, nsample) tensor

    cudaError_t err;
    dim3 blocks(DIVUP(M * C * nsample, THREADS_PER_BLOCK));  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    group_points_kernel_stack<<<blocks, threads>>>(B, M, C, nsample, features, features_batch_cnt, idx, idx_batch_cnt, out);
    // cudaDeviceSynchronize();  // for using printf in kernel function
    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}
