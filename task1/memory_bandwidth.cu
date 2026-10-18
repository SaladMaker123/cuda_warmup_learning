#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include "../common/cuda_check.h"

//GPU kernel:每个线程复制一个数组元素
__global__ void copyKernel(
    const float* src,
    float* dst,
    size_t N
)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N)
    {
        dst[idx] = src[idx];
    }
}

int main(int argc, char* argv[])
{
    //每个数组的大小：256MiB
    const size_t bytes = 256ULL * 1024 * 1024;
    //计算数组元素的数量
    const size_t N = bytes/sizeof(float);
    //声明两个GPU指针
    float* d_src = nullptr;
    float* d_dst = nullptr;
    //在GPU上分配两个数组
    CUDA_CHECK(cudaMalloc(&d_src,bytes));
    CUDA_CHECK(cudaMalloc(&d_dst,bytes));
    //暂时只测试内存分配
    std::printf("Array size: %zu MiB\n",bytes/(1024 * 1024));
    std::printf("Numbers of elements: %zu\n",N);

    //在CPU上创建源数组
    std::vector<float> h_src(N);
    //初始化
    for(size_t i = 0;i < N;i++)
    {
        h_src[i] = static_cast<float>(i%1000);
    }
    CUDA_CHECK(cudaMemcpy(
    d_src,
    h_src.data(),
    bytes,
    cudaMemcpyHostToDevice
    ));

    // 每个 Block 使用 256 个线程
    // 默认使用 256 Threads/Block
    int threads = 256;

    // 如果用户提供命令行参数，就读取该参数
    if (argc >= 2) {
        threads = std::atoi(argv[1]);
    }

    // 只允许测试这四种 Block 大小
    if (threads != 64 &&
        threads != 128 &&
        threads != 256 &&
        threads != 512) {

        std::printf("Invalid threads per block!\n");
        return 1;
    }

    // 计算需要多少个 Block
    const int blocks = static_cast<int>(
    (N + threads - 1) / threads
    );
    std::printf(
            "Threads per Block: %d | Blocks: %d\n",
            threads,
            blocks
        );
    // 启动 GPU Copy Kernel
    copyKernel<<<blocks, threads>>>(d_src, d_dst, N);

    // 检查 Kernel 启动与执行错误
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());


    // 用于检查复制结果的 CPU 变量
    float first = 0.0f;
    float second = 0.0f;
    float last = 0.0f;

    // 读取 d_dst 的第一个元素
    CUDA_CHECK(cudaMemcpy(
        &first,
        d_dst,
        sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    // 读取 d_dst 的第二个元素
    CUDA_CHECK(cudaMemcpy(
        &second,
        d_dst + 1,
        sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    // 读取 d_dst 的最后一个元素
    CUDA_CHECK(cudaMemcpy(
        &last,
        d_dst + (N - 1),
        sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    // 比较源数组和目标数组
    bool ok =
        (first == h_src[0]) &&
        (second == h_src[1]) &&
        (last == h_src[N - 1]);

    std::printf(
        "First: %.1f | Second: %.1f | Last: %.1f\n",
        first, second, last
    );

    std::printf(
        "Sample verification: %s\n",
        ok ? "PASS" : "FAIL"
    );

    //检查整个数组
    // 1. 在 CPU 上创建目标数组
    std::vector<float> h_dst(N);

    // 2. 将 GPU 目标数组完整复制回 CPU
    CUDA_CHECK(cudaMemcpy(
        h_dst.data(),
        d_dst,
        bytes,
        cudaMemcpyDeviceToHost
    ));

    // 3. 逐个比较两个数组
    bool allCorrect = true;

    for (size_t i = 0; i < N; ++i) {

        if (h_dst[i] != h_src[i]) {
            allCorrect = false;

            std::printf(
                "Mismatch at index %zu\n",
                i
            );

            break;
        }
    }

    // 4. 输出完整验证结果
    std::printf(
        "Full verification: %s\n",
        allCorrect ? "PASS" : "FAIL"
    );
  
    cudaEvent_t start, stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // 2. 重复测量 5 次
    const int repeats = 5;

    std::vector<double> bandwidths;

    for (int r = 0; r < repeats; ++r) {

        // 开始计时
        CUDA_CHECK(cudaEventRecord(start));

        // 启动 Copy Kernel
        copyKernel<<<blocks, threads>>>(d_src, d_dst, N);

        CUDA_CHECK(cudaGetLastError());

        // 结束计时
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        // 获取本次执行时间
        float ms = 0.0f;

        CUDA_CHECK(cudaEventElapsedTime(
            &ms, start, stop
        ));

        // 计算本次有效带宽，单位 GB/s
        double bandwidthGBs = (2.0 * bytes) / (ms * 1e6);

        bandwidths.push_back(bandwidthGBs);

        std::printf(
            "Run %d: %.3f ms | %.2f GB/s\n",
            r + 1,
            ms,
            bandwidthGBs
        );
    }

    // 3. 将 5 次带宽结果排序
    std::sort(bandwidths.begin(), bandwidths.end());

    // 4. 取中位数
    double medianBandwidth = bandwidths[repeats / 2];

    std::printf(
        "Median Bandwidth: %.2f GB/s\n",
        medianBandwidth
    );

    // 5. 释放 CUDA Event
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    //释放GPU内存
    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaFree(d_dst));
    return 0;
}