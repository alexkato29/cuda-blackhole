#pragma once
#include <math.h>
#include <cuda_runtime.h>


// TODO: Add CUDA error checking/handling

struct RayState {
    float3 position;
    float3 velocity;
};

__host__ __device__ inline float3 operator*(float3 a, float s) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}

__host__ __device__ inline float3 operator+(float3 a, float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ __device__ inline float3 operator-(float3 a, float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__host__ __device__ inline float dot(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ inline float3 cross(float3 a, float3 b) {
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

__device__ inline float length(float3 v) {
    return sqrtf(dot(v, v));
}

__device__ inline float length_squared(float3 v) {
    return dot(v, v);
}

__host__ __device__ inline float3 normalize(float3 v) {
    // This is technically less precise than 1.0f/magnitude, so keep that in mind
    return v * rsqrtf(dot(v, v));
}

__device__ inline float2 uv_mapping(float3 dir) {
    // For texture mapping, the radius is irrelevant. Also, dir is normalized.
    // https://drakeor.com/2023/04/27/equirectangular-to-skybox-projection/
    // https://en.wikipedia.org/wiki/List_of_common_coordinate_transformations?ref=drakeor.com#From_spherical_coordinates
    float phi = atan2f(dir.y, dir.x);
    float theta = acosf(dir.z);

    float u = (phi + M_PI) / (2.0f * M_PI);
    float v = theta / M_PI;

    return make_float2(u, v);
}
