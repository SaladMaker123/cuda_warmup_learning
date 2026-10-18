#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>

#include "../common/cuda_check.h"


// GPU Kernel：串行依赖访问
__global__ void pointerChase(
    const unsigned int* arr,
    unsigned int steps,
    unsigned int* result)
    {
    unsigned int idx = 0;

    for (unsigned int i = 0; i < steps; ++i) {
        idx = arr[idx];
    }

    result[0] = idx;
}

int main()
{
    // 每组工作集的大小，单位 KiB
    const size_t sizesKB[] = {4, 64, 512, 8192};

    // 每次测试执行的依赖读取次数
    const unsigned int minSteps = 1000000;

    for (size_t sizeKB : sizesKB) {

        size_t bytes = sizeKB * 1024;
        size_t N = bytes / sizeof(unsigned int);
        unsigned int steps = minSteps;

    if (N > steps) {
        steps = static_cast<unsigned int>(N);
    }

        // 生成随机排列
        std::vector<unsigned int> order(N);
        std::iota(order.begin(), order.end(), 0);

        std::mt19937 rng(42);
        std::shuffle(order.begin(), order.end(), rng);

        // 构建一个完整的循环访问链
        std::vector<unsigned int> h_arr(N);

        for (size_t i = 0; i < N; ++i) {
            h_arr[order[i]] = order[(i + 1) % N];
        }

        unsigned int* d_arr = nullptr;
        unsigned int* d_result = nullptr;

        CUDA_CHECK(cudaMalloc(&d_arr, bytes));
        CUDA_CHECK(cudaMalloc(&d_result, sizeof(unsigned int)));

        CUDA_CHECK(cudaMemcpy(
            d_arr,
            h_arr.data(),
            bytes,
            cudaMemcpyHostToDevice
        ));

        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        // 预热
        pointerChase<<<1, 1>>>(d_arr, steps, d_result);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // 开始计时
       // 每组工作集重复测试 5 次
    const int repeats = 5;

    // 保存 5 次测量得到的延迟
    std::vector<double> latencies;

    for (int r = 0; r < repeats; ++r) {

        CUDA_CHECK(cudaEventRecord(start));

        pointerChase<<<1, 1>>>(d_arr, steps, d_result);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;

        CUDA_CHECK(cudaEventElapsedTime(
            &ms, start, stop
        ));

        double ns = ms * 1e6 / steps;

        latencies.push_back(ns);
    }

    // 将 5 次测量结果从小到大排序
    std::sort(latencies.begin(), latencies.end());

    // 取排序后中间位置的值
    double nsPerLoad = latencies[repeats / 2];

        std::printf(
            "Size: %6zu KiB | Latency: %.2f ns\n",
            sizeKB,
            nsPerLoad
        );
        std::printf("Steps: %u\n", steps);

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));

        CUDA_CHECK(cudaFree(d_arr));
        CUDA_CHECK(cudaFree(d_result));
    }

    return 0;
}