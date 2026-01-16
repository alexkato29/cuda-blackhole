#include <cuda_runtime.h>
#include <math.h>
#include "helpers.h"
#include "blackhole.h"

#define BLACK make_float3(0.0f, 0.0f, 0.0f)


__global__ void blackhole(
	float *input,
	int input_width,
	int input_height,
	float *output,
	int output_width,
	int output_height,
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
	int global_idx = (y * output_width + x) * 3;

	if (x >= output_width || y >= output_height) return;

	float screen_x = ((float) x / output_width - 0.5f) * tan_fov;
	float screen_y = ((float) y / output_height - 0.5f) * tan_fov * ((float) output_height / output_width);
	float3 view_direction = camera_front + camera_left * screen_x + camera_up * screen_y;
	view_direction = normalize(view_direction);

	RayState ray;
	ray.position = camera_position;
	ray.velocity = view_direction;

	bool hit_event_horizon = false;

	for (int n = 0; n < num_iterations; n++) {
		float old_radius = length(ray.position);
		ray.position = ray.position + ray.velocity * step_size;
		float new_radius = length(ray.position);

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
		float3 dir = normalize(ray.velocity);

		// Convert to polar coordinates. We don't care about radius, we assume infinte distance.
		// Just want to know what the ray would hit if it went forever.
		float phi = atan2f(dir.y, dir.x);
		float theta = acosf(dir.z);

		float u = (phi + M_PI) / (2.0f * M_PI);
		float v = theta / M_PI;

		int px = (int)(u * input_width) % input_width;
		int py = (int)(v * input_height) % input_height;

		int input_idx = (py * input_width + px) * channels;
		color.x = input[input_idx + 0];
		color.y = input[input_idx + 1];
		color.z = input[input_idx + 2];
	}

	output[global_idx + 0] = color.x;
	output[global_idx + 1] = color.y;
	output[global_idx + 2] = color.z;
}

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
) {
	float3 camera_position = make_float3(0.0f, 1.0f, -45.0f);
	float3 facing = make_float3(0.0f, 0.0f, 0.0f);

	float3 camera_front = normalize(facing - camera_position);
	float3 camera_left = normalize(cross(camera_front, make_float3(0.0f, 1.0f, 0.0f)));
	float3 camera_up = normalize(cross(camera_front, camera_left));

	float fov_radians = fov_degrees * M_PI / 180.0f;
	float tan_fov = 2.0f * tanf(fov_radians / 2.0f);

	// For now, these are just arbitrary
	int num_iterations = 100000;
	float step_size = 0.01f;

	dim3 threads(16, 16);
	dim3 blocks((output_width + threads.x - 1) / threads.x, (output_height + threads.y - 1) / threads.y);

	blackhole<<<blocks, threads>>>(
		d_input,
		input_width,
		input_height,
		d_output,
		output_width,
		output_height,
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

