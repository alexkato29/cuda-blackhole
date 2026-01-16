#include <cmath>
#include <cuda_runtime.h>
#include "helpers.h"


// TODO: Add CUDA error checking/handling

__host__ __device__ inline float3 operator*(float3 a, float s) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}

__host__ __device__ inline float3 operator+(float3 a, float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ __device__ inline float length(float3 v) {
	return sqrtf(dot(v, v));
}

__host__ __device__ inline float dot(float3 a, float3 b) {
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ inline float3 normalize(float3 v) {
	// This is technically less precise than 1.0f/magnitude, so keep that in mind
    return v * rsqrtf(dot(v, v));
}

__host__ __device__ inline float3 cross(float3 a, float3 b) {
	return make_float3(
		a.y * b.z - a.z * b.y,
		a.z * b.x - a.x * b.z,
		a.x * b.y - a.y * b.x
	);
}
