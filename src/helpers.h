#pragma once
#include <cuda_runtime.h>


struct RayState {
    float3 position;
    float3 velocity;
};

__host__ __device__ inline float3 operator*(float3 a, float s);
__host__ __device__ inline float dot(float3 a, float3 b);
__host__ __device__ inline float3 normalize(float3 v);
__host__ __device__ inline float3 cross(float3 a, float3 b);
__host__ __device__ inline float3 operator+(float3 a, float3 b);
__host__ __device__ inline float length(float3 v);
