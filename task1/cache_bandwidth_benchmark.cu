#include <cuda_runtime.h>
#include <cstddef>
#include <cstdio>
#include <vector>

#include "../common/cuda_check.h"
__device__ __forceinline__ float loadCA(const float* address)
{
    float value;

    asm volatile(
        "ld.global.ca.f32 %0, [%1];"
        : "=f"(value)
        : "l"(address)
    );

    return value;
}
//四累加器缓存读取kernel
__global__ void cacheReadKernelCA(
    const float* src,
    float* partial,
    size_t N,
    int repeats
)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;
    size_t jump = stride * 4;

    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;

    for(int r = 0;r<repeats;++r)
    {
        for(size_t i = tid;i<N;i += jump)
        {
            sum0 += loadCA(src + i);
            if(i + stride < N){
                sum1 += loadCA(src + i + stride);
            }
            if(i + 2 * stride < N){
                sum2 += loadCA(src + i + 2 * stride);
            }
            if(i + 3 * stride < N){
                sum3 += loadCA(src + i + 3 * stride);
            }
        }
    }
    partial[tid] = sum0 + sum1 + sum2 + sum3;

}

//对指定数组大小进行一次完整benchmark
void runBenchmark(size_t bytes)
{
    const int blockSize = 256;
    const int gridSize = 32;
    const int repeats = 100;

    const int warmup = 10;
    const int trials = 10;
    const int batch = 20;

    const size_t N = bytes/sizeof(float);
    const size_t totalThreads = gridSize * blockSize;

    //cpu数据初始化
    std::vector<float> h_src(N,1.0f);
    std::vector<float> h_partial(totalThreads);

    // 分配GPU内存
    float* d_src = nullptr;
    float* d_partial = nullptr;

    CUDA_CHECK(cudaMalloc(&d_src,bytes));
    CUDA_CHECK(cudaMalloc(&d_partial,totalThreads * sizeof(float)));
    // 3. CPU -> GPU
    CUDA_CHECK(cudaMemcpy(d_src,h_src.data(),bytes,cudaMemcpyHostToDevice));
    //创建event
    cudaEvent_t start,stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    //预热，不计时

    for(int w = 0;w < warmup;w++)
    {
        cacheReadKernelCA<<<gridSize,blockSize>>>(
            d_src,d_partial,N,repeats
        );
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    //正式测量
    double totalMs = 0.0;

    for(int t = 0;t<trials;t++)
    {
        //开始记录点
        CUDA_CHECK(cudaEventRecord(start));
        //每组启动batch次kernel
        for(int b = 0; b<batch;b++)
        {
            cacheReadKernelCA<<<gridSize,blockSize>>>(
            d_src,d_partial,N,repeats
        );
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventSynchronize(stop));

        float batchMs = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(
            &batchMs, start, stop
        ));
        // 一组 batch 次 Kernel 的平均时间
        double kernelMs = batchMs / batch;

        totalMs += kernelMs;
        std::printf(
            "Trial %d: %.6f ms/kernel\n",
            t + 1, kernelMs
        );
        

    }
    double avgMs = totalMs / trials;

    // 7. GPU -> CPU
    CUDA_CHECK(cudaMemcpy(
        h_partial.data(),
        d_partial,
        totalThreads * sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    //cpu验证
    std::vector<float> expected(totalThreads,0.0f);
    for(size_t i = 0;i<N;i++)
    {
        size_t tid = i % totalThreads;
        expected[tid] += h_src[i] * repeats;
     
    }
    bool passed = true;

    for (size_t tid = 0; tid < totalThreads; ++tid) {
        if (h_partial[tid] != expected[tid]) {
            passed = false;

            std::printf(
                "Mismatch at thread %zu: GPU=%.1f CPU=%.1f\n",
                tid, h_partial[tid], expected[tid]
            );
        }
    }

    // 9. 计算逻辑读取吞吐量
    double logicalBytes =
        static_cast<double>(bytes) * repeats;

    double bandwidthGBs =
        logicalBytes / (avgMs * 1.0e6);

    std::printf("\nWorking set: %.2f KiB\n",
                bytes / 1024.0);

    std::printf("Average kernel time: %.6f ms\n", avgMs);

    std::printf("Effective read throughput: %.2f GB/s\n",
                bandwidthGBs);

    std::printf("Thread 0 sum: %.1f\n", h_partial[0]);

    std::printf(
        "Verification %s\n\n",
        passed ? "PASSED!" : "FAILED!"
    );

    // 10. 释放资源
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaFree(d_partial));
}

int main()
{
    // 64 KiB
    runBenchmark(64 * 1024);

    // 512 KiB
    runBenchmark(512 * 1024);

    // 8 MiB
    runBenchmark(8 * 1024 * 1024);

    return 0;
}
