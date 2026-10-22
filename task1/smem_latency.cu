#include <cuda_runtime.h>
#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <numeric>
#include <random>
#include <vector>

#include "../common/cuda_check.h"

constexpr unsigned int CHASE_UNROLL = 8;
constexpr unsigned int STEPS = 1000000;

constexpr int SHORT_WARMUP_STEPS = 16;
constexpr int FULL_WARMUP = 10;
constexpr int TRIALS = 10;
constexpr int BATCH = 3;

// Shared Memory Pointer Chasing Kernel
__global__ void sharedMemoryPointerChaseKernel(
    const unsigned int* globalChain,
    unsigned int* sink,
    size_t n,
    unsigned int steps)
{
    extern __shared__ unsigned int smem[];

    // 将访问链从 Global Memory 复制到 Shared Memory
    for (size_t i = threadIdx.x;
         i < n;
         i += blockDim.x)
    {
        smem[i] = globalChain[i];
    }

    // 确保整个访问链初始化完成
    __syncthreads();

    // 只让线程 0 执行串行依赖读取
    if (threadIdx.x == 0)
    {
        unsigned int idx = 0;

        for (unsigned int i = 0; i < steps;
             i += CHASE_UNROLL)
        {
            #pragma unroll
            for (unsigned int j = 0;
                 j < CHASE_UNROLL;
                 ++j)
            {
                idx = smem[idx];
            }
        }

        sink[0] = idx;
    }
}

// 构造覆盖全部元素的单一随机访问环
void buildRandomCycle(
    std::vector<unsigned int>& chain,
    unsigned int seed)
{
    const size_t n = chain.size();

    std::vector<unsigned int> perm(n);
    std::iota(perm.begin(), perm.end(), 0u);

    std::mt19937 rng(seed);
    std::shuffle(perm.begin(), perm.end(), rng);

    for (size_t i = 0; i + 1 < n; ++i)
    {
        chain[perm[i]] = perm[i + 1];
    }

    chain[perm[n - 1]] = perm[0];
}

// 测试一种工作集大小
bool runBenchmark(
    int worksetKiB,
    const cudaDeviceProp& prop)
{
    const size_t smemBytes =
        static_cast<size_t>(worksetKiB) * 1024;

    const size_t n =
        smemBytes / sizeof(unsigned int);

    std::printf("\n========== %d KiB ==========\n",
                worksetKiB);

    // 检查设备是否支持该工作集
    if (smemBytes >
        static_cast<size_t>(prop.sharedMemPerBlockOptin))
    {
        std::printf("SKIPPED: exceeds opt-in limit.\n");
        return true;
    }

    // 超过默认上限时，申请更大的动态共享内存上限
    if (smemBytes >
        static_cast<size_t>(prop.sharedMemPerBlock))
    {
        CUDA_CHECK(cudaFuncSetAttribute(
            sharedMemoryPointerChaseKernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(smemBytes)
        ));
    }

    // 1. CPU 构造随机访问链
    std::vector<unsigned int> h_chain(n);

    buildRandomCycle(
        h_chain,
        12345u + static_cast<unsigned int>(worksetKiB)
    );

    // 2. CPU 计算预期结果
    unsigned int expected = 0;

    for (unsigned int i = 0; i < STEPS; ++i)
    {
        expected = h_chain[expected];
    }

    // 3. 分配 GPU 内存
    unsigned int* d_chain = nullptr;
    unsigned int* d_sink = nullptr;

    CUDA_CHECK(cudaMalloc(
        &d_chain,
        n * sizeof(unsigned int)
    ));

    CUDA_CHECK(cudaMalloc(
        &d_sink,
        sizeof(unsigned int)
    ));

    // 4. 将访问链复制到 GPU
    CUDA_CHECK(cudaMemcpy(
        d_chain,
        h_chain.data(),
        n * sizeof(unsigned int),
        cudaMemcpyHostToDevice
    ));

    // 5. 创建 CUDA Events
    cudaEvent_t start, stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // 6. 短预热：16 次依赖读取
    sharedMemoryPointerChaseKernel
        <<<1, 128, smemBytes>>>(
            d_chain, d_sink, n, SHORT_WARMUP_STEPS
        );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // 7. 完整 Kernel 预热：10 次，不计时
    for (int w = 0; w < FULL_WARMUP; ++w)
    {
        sharedMemoryPointerChaseKernel
            <<<1, 128, smemBytes>>>(
                d_chain, d_sink, n, STEPS
            );
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // 8. 正式测量：10 组，每组 3 次 Kernel
    double totalMs = 0.0;

    for (int t = 0; t < TRIALS; ++t)
    {
        CUDA_CHECK(cudaEventRecord(start));

        for (int b = 0; b < BATCH; ++b)
        {
            sharedMemoryPointerChaseKernel
                <<<1, 128, smemBytes>>>(
                    d_chain, d_sink, n, STEPS
                );
        }

        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventSynchronize(stop));

        float batchMs = 0.0f;

        CUDA_CHECK(cudaEventElapsedTime(
            &batchMs, start, stop
        ));

        const double kernelMs = batchMs / BATCH;

        totalMs += kernelMs;

        std::printf(
            "Trial %d: %.6f ms/kernel\n",
            t + 1, kernelMs
        );
    }

    const double avgMs = totalMs / TRIALS;

    // 9. 读取 GPU 最终结果
    unsigned int h_result = 0;

    CUDA_CHECK(cudaMemcpy(
        &h_result,
        d_sink,
        sizeof(unsigned int),
        cudaMemcpyDeviceToHost
    ));

    const bool passed = (h_result == expected);

    // 10. 计算有效平均延迟
    const double latencyNs =
        avgMs * 1.0e6 / STEPS;

    // clockRate 的单位为 kHz，转换为 MHz
    int clockRateKHz = 0;

    CUDA_CHECK(cudaDeviceGetAttribute(
        &clockRateKHz,
        cudaDevAttrClockRate,
        0
    ));

    const double clockMHz =
    static_cast<double>(clockRateKHz) / 1000.0;
    const double estimatedCycles =
        latencyNs * clockMHz / 1000.0;

    std::printf("\nWorking set: %d KiB\n", worksetKiB);
    std::printf("Average kernel time: %.6f ms\n", avgMs);
    std::printf("Effective latency: %.4f ns/load\n", latencyNs);
    std::printf("Estimated cycles/load: %.2f\n",
                estimatedCycles);

    std::printf("GPU result: %u\n", h_result);
    std::printf("CPU expected: %u\n", expected);

    std::printf(
        "Verification: %s\n",
        passed ? "PASSED" : "FAILED"
    );

    // 11. 释放资源
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaFree(d_chain));
    CUDA_CHECK(cudaFree(d_sink));

    return passed;
}

int main()
{
    cudaDeviceProp prop{};

    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    std::printf("GPU: %s\n", prop.name);
    std::printf("Compute Capability: %d.%d\n",
                prop.major, prop.minor);

    const int worksets[] = {16, 24, 32, 48, 64};

    bool allPassed = true;

    for (int kb : worksets)
    {
        if (!runBenchmark(kb, prop))
        {
            allPassed = false;
        }
    }

    std::printf(
        "\nOverall result: %s\n",
        allPassed ? "PASSED" : "FAILED"
    );

    return allPassed ? 0 : 1;
}
