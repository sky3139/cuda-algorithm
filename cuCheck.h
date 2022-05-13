// Define this to turn on error checking
#pragma once
#include <iostream>

#define CUDA_ERROR_CHECK

#define CK(err) __CK(err, __FILE__, __LINE__)
#define CudaCheckError() __cudaCheckError(__FILE__, __LINE__)

inline void __CK(cudaError err, const char *file, const int line)
{
#ifdef CUDA_ERROR_CHECK
    if (cudaSuccess != err)
    {
        fprintf(stderr, "CK() failed at %s:%i : %s\n",
                file, line, cudaGetErrorString(err));
        exit(-1);
    }
#endif

    return;
}

inline void __cudaCheckError(const char *file, const int line)
{
#ifdef CUDA_ERROR_CHECK
    cudaError err = cudaGetLastError();
    if (cudaSuccess != err)
    {
        fprintf(stderr, "cudaCheckError() failed at %s:%i : %s\n",
                file, line, cudaGetErrorString(err));
        exit(-1);
    }

    // More careful checking. However, this will affect performance.
    // Comment away if needed.
    err = cudaDeviceSynchronize();
    if (cudaSuccess != err)
    {
        fprintf(stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
                file, line, cudaGetErrorString(err));
        exit(-1);
    }
#endif

    return;
}
#include <chrono>
#include <sys/time.h>
double seconds()
{
    struct timeval time;

    /* 获取时间，理论到us */
    gettimeofday(&time, NULL);
    // printf("s: %ld, ms: %ld\n", time.tv_sec, );
    // std::chrono::duration_cast<std::chrono::microseconds>();
    return (time.tv_sec * 1000 + time.tv_usec / 1000);
}