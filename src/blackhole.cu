#include <cuda_runtime.h>
#include "helpers.h"
#include "blackhole.h"

#define BLACK make_float3(0.0f, 0.0f, 0.0f)


__global__ void blackhole(
	float *input,
	float *output,
	int width,
	int height,
	int channels,
	float3 camera_front,
	float3 camera_up,
	float3 camera_left,
	float3 camera_position,
	float tan_fov,
	int num_iterations,
	float step_size,
	float schwarzchild_radius
) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int global_idx = (y * width + x) * 3;

	if (x >= width || y >= height) return;

	float screen_x = ((float) x / width - 0.5f) * tan_fov;
	float screen_y = ((float) y / height - 0.5f) * tan_fov * ((float) height / width);
	float3 view_direction = camera_front + camera_left * screen_x + camera_up * screen_y;
	view_direction = normalize(view_direction);

	RayState r;
	r.position = camera_position;
	r.velocity = view_direction;

	bool hit_event_horizon = false;

	for (int n = 0; n < num_iterations; n++) {
		float old_radius = length(r.position);
		r.position += r.velocity * step_size;
		float new_radius = length(r.position);

		if (new_radius < schwarzchild_radius && old_radius >= schwarzchild_radius) {
			hit_event_horizon = true;
			break;
		}

		if (new_radius > 1000.0f) break;
	}

	float3 color;
	if (hit_event_horizon) {
		color = BLACK;
	} else {
		color.x = input[global_idx + 0];
		color.y = input[global_idx + 1];
		color.z = input[global_idx + 2];
	}

	output[global_idx + 0] = color.x;
	output[global_idx + 1] = color.y;
	output[global_idx + 2] = color.z;
}

void raytrace_blackhole(
	float* d_input,
	float* d_output,
	int width,
	int height,
	int channels
) {
	float3 camera_position = make_float3(0.0f, 1.0f, -20.0f);
	float3 facing = make_float3(0.0f, 0.0f, 0.0f);

	float3 camera_front = normalize(camera_position - facing);
	float3 camera_left = cross(camera_front, make_float3(0.0f, 1.0f, 0.0f));
	float3 camera_up = cross(camera_front, camera_left);

	// For now, these are just arbitrary
	float tan_fov = 1.0f;
	int num_iterations = 500;
	float step_size = 0.08f;

	float schwarzchild_radius = 1.0f;

	dim3 threads(16, 16);
	dim3 blocks((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y);

	blackhole<<<blocks, threads>>>(
		d_input,
		d_output,
		width,
		height,
		channels,
		camera_front,
		camera_up,
		camera_left,
		camera_position,
		tan_fov,
		num_iterations,
		step_size,
		schwarzchild_radius
	);
	cudaDeviceSynchronize();
}
