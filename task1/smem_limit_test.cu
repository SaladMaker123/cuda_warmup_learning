#include <cuda_runtime.h>
#include <cstddef>
#include <cstdio>

#include "../common/cuda_check.h"

// 使用动态共享内存进行简单测试
__global__ void sharedMemoryTest(
    unsigned long long* result,
    size_t n)
{
    // 声明动态共享内存
    extern __shared__ unsigned int smem[];

    // 每个线程负责初始化一部分元素
    for (size_t i = threadIdx.x;
         i < n;
         i += blockDim.x)
    {
        smem[i] = static_cast<unsigned int>(i);
    }

    // 确保所有元素初始化完成
    __syncthreads();

    // 由线程 0 计算总和
    if (threadIdx.x == 0)
    {
        unsigned long long sum = 0;

        for (size_t i = 0; i < n; ++i)
        {
            sum += smem[i];
        }

        *result = sum;
    }
}

int main()
{
    // 1. 查询 GPU 属性
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    std::printf("GPU: %s\n", prop.name);

    std::printf(
        "Default shared memory limit: %zu KiB\n",
        static_cast<size_t>(prop.sharedMemPerBlock) / 1024
    );

    std::printf(
        "Opt-in shared memory limit: %zu KiB\n\n",
        static_cast<size_t>(prop.sharedMemPerBlockOptin) / 1024
    );

    // 2. 创建结果存储空间
    unsigned long long* d_result = nullptr;

    CUDA_CHECK(cudaMalloc(
        &d_result,
        sizeof(unsigned long long)
    ));

    // 3. 依次测试五种工作集
    const int worksetKiB[] = {16, 24, 32, 48, 64};
    bool allPassed = true;

    for (int kb : worksetKiB)
    {
        const size_t smemBytes =
            static_cast<size_t>(kb) * 1024;

        const size_t n =
            smemBytes / sizeof(unsigned int);

        std::printf("Testing %d KiB...\n", kb);

        // 4. 检查是否超过 opt-in 上限
        if (smemBytes >
            static_cast<size_t>(prop.sharedMemPerBlockOptin))
        {
            std::printf("SKIPPED: exceeds opt-in limit.\n\n");
            allPassed = false;
            continue;
        }

        // 5. 超过默认上限时，申请更大的动态共享内存
        if (smemBytes >
            static_cast<size_t>(prop.sharedMemPerBlock))
        {
            CUDA_CHECK(cudaFuncSetAttribute(
                sharedMemoryTest,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(smemBytes)
            ));
        }

        // 6. 启动 Kernel
        sharedMemoryTest<<<1, 128, smemBytes>>>(
            d_result,
            n
        );

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // 7. 将结果复制回 CPU
        unsigned long long h_result = 0;

        CUDA_CHECK(cudaMemcpy(
            &h_result,
            d_result,
            sizeof(h_result),
            cudaMemcpyDeviceToHost
        ));

        // 8. CPU 验证
        const unsigned long long expected =
            static_cast<unsigned long long>(n) * (n - 1) / 2;

        std::printf("GPU result: %llu\n", h_result);
        std::printf("CPU expected: %llu\n", expected);

        const bool passed = (h_result == expected);

        std::printf(
            "Verification: %s\n\n",
            passed ? "PASSED" : "FAILED"
        );

        if (!passed)
        {
            allPassed = false;
        }
    }

    // 9. 释放资源
    CUDA_CHECK(cudaFree(d_result));

    std::printf(
        "Overall result: %s\n",
        allPassed ? "PASSED" : "FAILED"
    );

    return allPassed ? 0 : 1;
}