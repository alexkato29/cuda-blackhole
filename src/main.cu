#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "blackhole.h"


void uchar_to_float(const unsigned char *input, float *output, size_t size) {
	for (size_t i = 0; i < size; i++) {
		output[i] = input[i] / 255.0f;
	}
}

void float_to_uchar(const float *input, unsigned char *output, size_t size) {
	for (size_t i = 0; i < size; i++) {
		float val = input[i] * 255.0f;
		output[i] = (unsigned char)(val < 0 ? 0 : (val > 255 ? 255 : val));
	}
}


int main(int argc, char **argv) {
	if (argc != 3) {
		fprintf(stderr, "Invalid Usage. Example: %s <input_image> <output_image>\n", argv[0]);
		return EXIT_FAILURE;
	}

	const char *input_path = argv[1];
	const char *output_path = argv[2];

	printf("Loading image from: %s\n", input_path);

	int width, height, channels;
	unsigned char *image = stbi_load(input_path, &width, &height, &channels, 0);

	if (!image) {
		fprintf(stderr, "Error: Failed to load image '%s'\n", input_path);
		return EXIT_FAILURE;
	}

	printf("Image loaded! %dx%d with %d channels\n", width, height, channels);

	size_t img_size = width * height * channels;
	size_t uchar_bytes = img_size * sizeof(unsigned char);
	size_t float_bytes = img_size * sizeof(float);

	float *h_input_float = nullptr;
	cudaMallocHost(&h_input_float, float_bytes);
	uchar_to_float(image, h_input_float, img_size);
	stbi_image_free(image);

	float *h_output_float = nullptr;
	cudaMallocHost(&h_output_float, float_bytes);

	float *d_input = nullptr, *d_output = nullptr;
	cudaMalloc(&d_input, float_bytes);
	cudaMalloc(&d_output, float_bytes);

	cudaMemcpy(d_input, h_input_float, float_bytes, cudaMemcpyHostToDevice);

	dim3 threads(16, 16);
	dim3 blocks((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y);
	blackhole<<<blocks, threads>>>(d_input, d_output, width, height, channels);
	cudaDeviceSynchronize();

	cudaMemcpy(h_output_float, d_output, float_bytes, cudaMemcpyDeviceToHost);

	cudaFree(d_input);
	cudaFree(d_output);

	unsigned char *h_output_uchar = (unsigned char *)malloc(uchar_bytes);
	float_to_uchar(h_output_float, h_output_uchar, img_size);

	printf("Writing image to: %s\n", output_path);
	if (!stbi_write_png(output_path, width, height, channels, h_output_uchar, width * channels)) {
		fprintf(stderr, "Error: Failed to write image '%s'\n", output_path);
		free(h_output_uchar);
		cudaFreeHost(h_input_float);
		cudaFreeHost(h_output_float);
		return EXIT_FAILURE;
	}

	printf("Successfully wrote (engulfed) image!\n");

	free(h_output_uchar);
	cudaFreeHost(h_input_float);
	cudaFreeHost(h_output_float);

	return EXIT_SUCCESS;
}
