#pragma once

void raytrace_blackhole(
    float* d_input,
    int input_width,
    int input_height,
    float* d_output,
    int output_width,
    int output_height,
    int channels,
    float fov_degrees,
    float schwarzchild_radius
);
