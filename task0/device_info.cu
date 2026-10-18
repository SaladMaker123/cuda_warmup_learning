#include <cuda_runtime.h>
#include <cstdio>
#include "../common/cuda_check.h"

int main()
{
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    std::printf("GPU Count: %d\n",deviceCount);
    if (deviceCount == 0)
    {
        return 1;
    }

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop,0));
    std::printf("\n===== GPU Information =====\n");
    std::printf("GPU Name: %s\n",prop.name);
    std::printf("Compute Capability:%d.%d\n",prop.major,prop.minor);
    std::printf("SM Count: %d\n",prop.multiProcessorCount);
    std::printf("Warp Size: %d\n",prop.warpSize);
    std::printf("Max Threads Per Block: %d\n",prop.maxThreadsPerBlock);
    std::printf("Max Threads Per SM: %d\n",prop.maxThreadsPerMultiProcessor);
    std::printf("Registers per SM: %d\n",prop.regsPerMultiprocessor);
    std::printf("Shared Memory per Block: %zu bytes\n",prop.sharedMemPerBlock);
    std::printf("Shared Memory per SM: %zu bytes\n",prop.sharedMemPerMultiprocessor);
    std::printf("Global Memory: %.2f GiB\n",
                static_cast<double>(prop.totalGlobalMem)
                / (1024.0 * 1024.0 * 1024.0));
    std::printf("Memory Bus Width: %d bits\n",
                prop.memoryBusWidth);
    int memoryClockRate = 0;

    CUDA_CHECK(cudaDeviceGetAttribute(
    &memoryClockRate,
    cudaDevAttrMemoryClockRate,
    0
));

    std::printf("Memory Clock: %d kHz\n",
            memoryClockRate);

    return 0;

}