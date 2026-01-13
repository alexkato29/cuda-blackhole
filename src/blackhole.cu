#include "raytrace.h"

__global__ void blackhole(float *input, float *output, int width, int height, int channels) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height) return;

	int idx = (y * width + x) * channels;

	for (int c = 0; c < channels; c++) {
		output[idx + c] = input[idx + c] * 0.5;
	}
}
