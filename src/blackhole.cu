#include <stdio.h>

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
	float3 blackhole_position,
	float schwarzchild_radius
) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int global_idx = (y * output_width + x) * 3;

	if (x >= output_width || y >= output_height) return;

	/*
	A few notes:

	1. Unscaled screen_x/y go from -0.5 to 0.5. Multiplying the leftmost vector (-0.5 screen_x)
	by the camera_left (negative X direction) vector will have us look right! Hence the 
	subtraction to keep orientation as expected.

	2. FOV is actually super interesting. I just expected that high FOV = fisheye, but that's 
	not the case. For massive tan_fov, ray_direction values will asymptotically hang at near
	perpendicular angles to the front of the camera. As a result, we should see some tunneling 
	effect that looks like Hyperspace in Star Wars.
	*/
	float screen_x = ((float) x / output_width - 0.5f) * tan_fov;
	float screen_y = ((float) y / output_height - 0.5f) * tan_fov * ((float) output_height / output_width);
	float3 ray_direction = camera_front - camera_left * screen_x - camera_up * screen_y;
	ray_direction = normalize(ray_direction);

	// To save time, we will say that if you're already as far as the camera you are good.
	// When the camera starts deep in the gravity well, visuals are already very odd since
	// every geodesic is massively warped, so you should be backed up anyway.
	float early_stop_radius = length_squared(camera_position - blackhole_position);

	RayState ray;
	ray.position = camera_position;
	ray.velocity = ray_direction;

	bool hit_event_horizon = false;

	float3 angular_momentum = cross(ray.position - blackhole_position, ray.velocity);
	float h2 = length_squared(angular_momentum);

	for (int n = 0; n < num_iterations; n++) {		
		float old_radius = length_squared(ray.position - blackhole_position);
		// For now we simulate the light rays rather than using the field equations.
		// It's a simplified and (much faster) approach. But ultimately, if I have time
		// to properly learn GR I think it'd be great to actually use Einstein's Field Equations.
		ray =  rk2_step_midpoint(ray, h2, step_size, blackhole_position);
		float new_radius = length_squared(ray.position - blackhole_position);

		if (new_radius < schwarzchild_radius && old_radius >= schwarzchild_radius) {
			hit_event_horizon = true;
			break;
		}

		if (new_radius > early_stop_radius) break;
		// TODO: figure out why this early stopping condition makes rendering look off
		// if (dot(old_velocity, ray.velocity) == 1.0f && new_radius > old_radius) break;
	}

	float3 color;
	if (hit_event_horizon) {
		color = BLACK;
	} else {
		float2 uv = uv_mapping(ray.velocity);
		int px = (int)(uv.x * input_width);
		int py = (int)(uv.y * input_height);

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
	float schwarzchild_radius,
	float3 blackhole_position,
	int num_iterations,
	float step_size
) {
	// It just makes it simple to keep this at the origin.
	float3 camera_position = make_float3(0.0f, 0.0f, 0.0f);

	float3 camera_front = normalize(blackhole_position - camera_position);
	float3 camera_left = normalize(cross(make_float3(0.0f, 0.0f, 1.0f), camera_front));
	float3 camera_up = normalize(cross(camera_front, camera_left));

	// Note: tan_fov can be negative depending on FoV. Abs value prevents flipping the image.
	float fov_radians = fov_degrees * M_PI / 180.0f;
	float tan_fov = fabsf(2.0f * tanf(fov_radians / 2.0f));

	printf("\nFront: %f, %f, %f\n", camera_front.x, camera_front.y, camera_front.z);
	printf("Left: %f, %f, %f\n", camera_left.x, camera_left.y, camera_left.z);
	printf("Up: %f, %f, %f\n", camera_up.x, camera_up.y, camera_up.z);
	printf("Tan FoV: %f\n", tan_fov);

	dim3 threads(16, 16);
	dim3 blocks((output_width + threads.x - 1) / threads.x, (output_height + threads.y - 1) / threads.y);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
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
		blackhole_position,
		schwarzchild_radius
	);
	cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("Kernel Time: %f ms\n", ms);
}

