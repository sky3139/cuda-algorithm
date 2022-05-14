#pragma once
#include <cuda.h>
#include <cassert>
#include <iostream>
#include "./cuCheck.h"
template <class T>
struct Point
{
    T x, y, z;
    Point() : x(0), y(0), z(0) {}
    Point(T x, T y, T z) : x(x), y(y), z(z) {}
    T crass(const Point p, const Point q, const Point r)
    {
        return (q.x - p.x) * (r.y - q.y) - (q.y - p.y) * (r.x - q.x);
    }
};
typedef Point<float> Pointf;
typedef Point<int> Pointi;

template <class T>
struct cuBase
{
    T *devPtr;
    uint64_t capacity;
    uint64_t size;
    //拷贝构造函数
    cuBase(const cuBase &lth) : devPtr(lth.devPtr), capacity(lth.capacity), size(lth.size){};
    // struct cuBase copy()
    // {
    //     cuBase b;
    //     //   :  devPtr(lth.devPtr), capacity(lth.capacity), size(lth.size)
    //     return cuBase;
    // }
};
template <class T>
struct cuVector
{
public:
    cuBase<T> *cb;
    __host__ cuVector(int capacity)
    {

        CK(cudaMallocManaged((void **)&cb, sizeof(cuBase<T>)));
        cb->capacity = capacity;
        cb->size = 0;
        CK(cudaMallocManaged((void **)&(cb->devPtr), sizeof(T) * cb->capacity));
        CK(cudaMemset(cb->devPtr, 0, sizeof(T) * cb->capacity));
    }
    __host__ cuVector(int capacity, T val)
    {
        CK(cudaMallocManaged((void **)&cb, sizeof(cuBase<T>)));
        cb->capacity = capacity;
        cb->size = capacity;
        CK(cudaMallocManaged((void **)&(cb->devPtr), sizeof(T) * cb->capacity));
        CK(cudaMemset(cb->devPtr, val, sizeof(T) * cb->capacity));
    }
    //拷贝构造函数
    __host__ __device__ cuVector(const cuVector &lth)
    {
        // cb->devPtr = lth.cb->devPtr;
        // cb->size = lth.cb->size;
        // cb->capacity = lth.cb->capacity;
        cb = lth.cb;
    }
    __host__ void release()
    {
        cudaFree(cb);
    }
    ~cuVector()
    {
        cudaFree(cb->devPtr);
        // cudaFree(cb);
    }
    __host__ __device__ void push_back(T val)
    { //多线程时有问题，需要使用原子数
        assert(cb->size < cb->capacity);
        cb->devPtr[cb->size++] = val;
    }
    __host__ __device__ inline T &operator[](size_t x)
    {
        return *(cb->devPtr + x);
    }
    __host__ __device__ inline size_t size()
    {
        return cb->size;
    }
    __host__ __device__ inline size_t capacity()
    {
        return cb->capacity;
    }
    __host__ __device__ inline T &back()
    {
        return *(cb->devPtr + cb->size - 1);
    }
    __host__ __device__ inline void pop_back()
    {
        assert(cb->size != 0);
        cb->size--;
    }
    void print()
    {
        std::cout << size() << " " << capacity() << std::endl;
    }
    T *begin() { return cb->devPtr; };
    T *end() { return cb->devPtr + cb->capacity; };
};
template <class T>
class cuVector2D : public cuVector<T>
{
public:
    size_t rows, cols;
    cuVector2D(int rows, int cols,T val) : cuVector<T>(rows * cols,val), rows(rows), cols(cols)
    {
        // CK(cudaMallocManaged((void **)&devPtr, sizeof(T) * size));
        // CK(cudaMemset(devPtr, 0, sizeof(T) * size));
    }
    cuVector2D(int size)
    {
        //
    }
    inline T *operator()(size_t row, size_t col)
    {
        // if (rows < pitch)
        //     return mat[x];
        return ((cuVector<T>::cb->devPtr + cols * row + col));
    }
    __host__ __device__ inline T *operator[](size_t row)
    {
        // if (rows < pitch)
        //     return mat[x];
        return cuVector<T>::cb->devPtr + cols * row;
    }
};
template <class T>
struct Patch
{
    T *devPtr;
    size_t pitch = 0;
    size_t rows, cols;
    Patch()
    {
    }
    Patch(int rows, int cols) : rows(rows), cols(cols)
    {
        CK(cudaMallocPitch((void **)&devPtr, &pitch, cols * sizeof(float), rows));
        CK(cudaMemset2D(devPtr, pitch, 0, sizeof(T) * cols, rows));
    }
    ~Patch()
    {
        cudaFree(devPtr);
    }
    inline T operator()(size_t rows, size_t cols)
    {
        // if (rows < pitch)
        //     return mat[x];
        return *((devPtr + rows * pitch + cols));
    }
};