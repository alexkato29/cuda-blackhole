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
	// Todo: FOV, output resolution, and blackhole params are input parameters
	const char *input_path = argv[1];
	const char *output_path = argv[2];

	const int output_width = 1920;
	const int output_height = 1080;
	const float fov_degrees = 60.0f;

	const float schwarzchild_radius = 1.0f;

	printf("Loading image from: %s\n", input_path);

	// Channels is technically dynamically declared, but we can sorta always assume it's 3
	int input_width, input_height, channels;
	unsigned char *image = stbi_load(input_path, &input_width, &input_height, &channels, 0);

	if (!image) {
		fprintf(stderr, "Error: Failed to load image '%s'\n", input_path);
		return EXIT_FAILURE;
	}

	printf("Image loaded! %dx%d with %d channels\n", input_width, input_height, channels);

	size_t input_img_size = input_width * input_height * channels;
	size_t input_float_bytes = input_img_size * sizeof(float);

	size_t output_render_size = output_width * output_height * channels;
	size_t output_uchar_bytes = output_render_size * sizeof(unsigned char);
	size_t output_float_bytes = output_render_size * sizeof(float);

	float *h_input_float = nullptr;
	cudaMallocHost(&h_input_float, input_float_bytes);
	uchar_to_float(image, h_input_float, input_img_size);
	stbi_image_free(image);

	float *h_output_float = nullptr;
	cudaMallocHost(&h_output_float, output_float_bytes);

	float *d_input = nullptr, *d_output = nullptr;
	cudaMalloc(&d_input, input_float_bytes);
	cudaMalloc(&d_output, output_float_bytes);

	cudaMemcpy(d_input, h_input_float, input_float_bytes, cudaMemcpyHostToDevice);

	raytrace_blackhole(
		d_input, 
		input_width,
		input_height,
		d_output, 
		output_width,
		output_height,
		channels,
		fov_degrees,
		schwarzchild_radius
	);	

	cudaMemcpy(h_output_float, d_output, output_float_bytes, cudaMemcpyDeviceToHost);

	cudaFree(d_input);
	cudaFree(d_output);

	unsigned char *h_output_uchar = (unsigned char *)malloc(output_uchar_bytes);
	float_to_uchar(h_output_float, h_output_uchar, output_render_size);

	printf("Writing rendered image to: %s\n", output_path);
	if (!stbi_write_png(output_path, output_width, output_height, channels, h_output_uchar, output_width * channels)) {
		fprintf(stderr, "Error: Failed to write image '%s'\n", output_path);
		free(h_output_uchar);
		cudaFreeHost(h_input_float);
		cudaFreeHost(h_output_float);
		return EXIT_FAILURE;
	}

	printf("Successfully rendered (engulfed) image!\n");

	free(h_output_uchar);
	cudaFreeHost(h_input_float);
	cudaFreeHost(h_output_float);

	return EXIT_SUCCESS;
}
