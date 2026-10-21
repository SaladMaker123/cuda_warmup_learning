
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdio>
#include <vector>

#include "../common/cuda_check.h"


__global__ void cacheReadKernel(
    const float* src,
    float* partial,
    size_t N,
    int repeats)
{
    // 当前线程的全局编号
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    // 整个 Grid 的线程总数
    size_t stride = gridDim.x * blockDim.x;

    // 每次内层循环最多处理 4 个元素
    size_t jump = stride * 4;

    // 4 个独立累加器
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;

    // 重复读取数组
    for (int r = 0; r < repeats; ++r) {

        // 每次处理当前线程负责的 4 个位置
        for (size_t i = tid; i < N; i += jump) {

            sum0 += src[i];

            if (i + stride < N) {
                sum1 += src[i + stride];
            }

            if (i + 2 * stride < N) {
                sum2 += src[i + 2 * stride];
            }

            if (i + 3 * stride < N) {
                sum3 += src[i + 3 * stride];
            }
        }
    }

    // 合并当前线程的 4 个累加结果
    float sum = sum0 + sum1 + sum2 + sum3;

    // 将结果保存到 GPU 数组中
    partial[tid] = sum;
}


int main()
{
    // ==============================
    // 1. 设置实验参数
    // ==============================

    const size_t N = 512 * 1024 / sizeof(float);

    const int blockSize = 256;
    const int gridSize = 32;
    const int repeats = 100;

    const size_t totalThreads = gridSize * blockSize;

    // ==============================
    // 2. 分配 GPU 内存
    // ==============================

    float* d_src = nullptr;
    float* d_partial = nullptr;

    CUDA_CHECK(cudaMalloc(
        &d_src,
        N * sizeof(float)
    ));

    CUDA_CHECK(cudaMalloc(
        &d_partial,
        totalThreads * sizeof(float)
    ));

    // ==============================
    // 3. 初始化 CPU 源数组
    // ==============================

    std::vector<float> h_src(N);

    for (size_t i = 0; i < N; ++i) {
        h_src[i] = 1.0f;
    }

    // ==============================
    // 4. CPU -> GPU
    // ==============================

    CUDA_CHECK(cudaMemcpy(
        d_src,
        h_src.data(),
        N * sizeof(float),
        cudaMemcpyHostToDevice
    ));

    // ==============================
    // 5. 创建 CUDA Event
    // ==============================

    cudaEvent_t start, stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // ==============================
    // 6. Warmup：预热 10 次
    // ==============================

    const int warmup = 10;

    for (int w = 0; w < warmup; ++w) {

        cacheReadKernel<<<gridSize, blockSize>>>(
            d_src,
            d_partial,
            N,
            repeats
        );
    }

    CUDA_CHECK(cudaGetLastError());

    // 等待全部预热 Kernel 完成
    CUDA_CHECK(cudaDeviceSynchronize());

    // ==============================
    // 7. 正式计时 10 次
    // ==============================

    const int trials = 10;

    float totalMs = 0.0f;

    for (int t = 0; t < trials; ++t) {

        // 记录开始事件
        CUDA_CHECK(cudaEventRecord(start));

        // 启动 GPU Kernel
        cacheReadKernel<<<gridSize, blockSize>>>(
            d_src,
            d_partial,
            N,
            repeats
        );

        CUDA_CHECK(cudaGetLastError());

        // 记录结束事件
        CUDA_CHECK(cudaEventRecord(stop));

        // 等待结束事件完成
        CUDA_CHECK(cudaEventSynchronize(stop));

        // 获取本次计时结果
        float elapsedMs = 0.0f;

        CUDA_CHECK(cudaEventElapsedTime(
            &elapsedMs,
            start,
            stop
        ));

        // 累加时间
        totalMs += elapsedMs;

        // 输出本次时间
        std::printf(
            "Trial %d: %.6f ms\n",
            t + 1,
            elapsedMs
        );
    }

    // ==============================
    // 8. 计算平均执行时间
    // ==============================

    float avgMs = totalMs / trials;

    std::printf(
        "Average time: %.6f ms\n",
        avgMs
    );

    // 销毁 CUDA Event
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // ==============================
    // 9. GPU -> CPU
    // ==============================

    std::vector<float> h_partial(totalThreads);

    CUDA_CHECK(cudaMemcpy(
        h_partial.data(),
        d_partial,
        totalThreads * sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    // ==============================
    // 10. 输出各线程计算结果
    // ==============================

    for (size_t tid = 0; tid < 8; ++tid) {

        std::printf(
            "Thread %zu: sum = %.1f\n",
            tid,
            h_partial[tid]
        );
    }

    // ==============================
    // 11. CPU 通用结果验证
    // ==============================

    std::vector<float> expected(totalThreads, 0.0f);

    // 根据 Grid-Stride 分工计算预期结果
    for (size_t i = 0; i < N; ++i) {

        size_t tid = i % totalThreads;

        expected[tid] += h_src[i] * repeats;
    }

    bool passed = true;

    for (size_t i = 0; i < totalThreads; ++i) {

        if (h_partial[i] != expected[i]) {

            passed = false;

            std::printf(
                "Mismatch at thread %zu: GPU = %.1f, CPU = %.1f\n",
                i,
                h_partial[i],
                expected[i]
            );
        }
    }

    if (passed) {
        std::printf("Verification PASSED!\n");
    } else {
        std::printf("Verification FAILED!\n");
    }

    // ==============================
    // 12. 释放 GPU 内存
    // ==============================

    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaFree(d_partial));

    return passed ? 0 : 1;
}
