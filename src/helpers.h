#pragma once
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
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

__device__ inline float2 uv_mapping(float3 velocity) {
    // For texture mapping, the radius is irrelevant. Also, dir is normalized.
    // https://drakeor.com/2023/04/27/equirectangular-to-skybox-projection/
    // https://en.wikipedia.org/wiki/List_of_common_coordinate_transformations?ref=drakeor.com#From_spherical_coordinates
    float3 dir = normalize(velocity);

    // Note: atan2 is not arctan^2. It's a distinct function.
    float phi = atan2f(dir.y, dir.x);
    float theta = acosf(dir.z);

    float u = (phi + M_PI) / (2.0f * M_PI);
    float v = theta / M_PI;

    return make_float2(u, v);
}

struct Config {
    int output_width;
    int output_height;
    float fov_degrees;
    float schwarzchild_radius;
    float blackhole_x, blackhole_y, blackhole_z;
    int num_iterations;
    float step_size;
    char* input_path;
    char* output_path;
};

inline bool load_config(const char* filepath, Config* config) {
    config->output_width = 3840;
    config->output_height = 2160;
    config->fov_degrees = 60.0f;
    config->schwarzchild_radius = 1.0f;
    config->blackhole_x = 0.0f;
    config->blackhole_y = 20.0f;
    config->blackhole_z = 10.0f;
    config->num_iterations = 100000;
    config->step_size = 0.01f;
    config->input_path = strdup("images/input.jpg");
    config->output_path = strdup("images/output.jpg");

    FILE* file = fopen(filepath, "r");
    if (!file) {
        return false;
    }

    char line[256];
    while (fgets(line, sizeof(line), file)) {
        if (line[0] == '#' || line[0] == '\n' || line[0] == '\r') continue;

        char key[128], value[128];
        if (sscanf(line, "%127[^=]=%127s", key, value) == 2) {
            char* k = key;
            char* k_end = k + strlen(k) - 1;
            while (k_end > k && (*k_end == ' ' || *k_end == '\t' || *k_end == '\n' || *k_end == '\r')) {
                *k_end = '\0';
                k_end--;
            }

            if (strcmp(k, "output_width") == 0) config->output_width = atoi(value);
            else if (strcmp(k, "output_height") == 0) config->output_height = atoi(value);
            else if (strcmp(k, "fov_degrees") == 0) config->fov_degrees = atof(value);
            else if (strcmp(k, "schwarzchild_radius") == 0) config->schwarzchild_radius = atof(value);
            else if (strcmp(k, "blackhole_x") == 0) config->blackhole_x = atof(value);
            else if (strcmp(k, "blackhole_y") == 0) config->blackhole_y = atof(value);
            else if (strcmp(k, "blackhole_z") == 0) config->blackhole_z = atof(value);
            else if (strcmp(k, "num_iterations") == 0) config->num_iterations = atoi(value);
            else if (strcmp(k, "step_size") == 0) config->step_size = atof(value);
            else if (strcmp(k, "input_path") == 0) config->input_path = strdup(value);
            else if (strcmp(k, "output_path") == 0) config->output_path = strdup(value);
        }
    }

    fclose(file);
    return true;
}
