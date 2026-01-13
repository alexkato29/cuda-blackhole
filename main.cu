#include <stdio.h>
#include <stdlib.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"


int main(int argc, char **argv) {
	if (argc != 3) {
		fprintf(stderr, "Invalid Usage. Example: %s <input_image> <output_image>\n", argv[0]);
		return EXIT_FAILURE;
	}

	const char *input_path = argv[1];
	const char *output_path = argv[2];

	printf("Loading image from: %s\n", input_path);

	int width, height, channels;
	unsigned char *h_input = stbi_load(input_path, &width, &height, &channels, 0);

	if (!h_input) {
		fprintf(stderr, "Error: Failed to load image '%s'\n", input_path);
		return EXIT_FAILURE;
	}

	printf("Image loaded! %dx%d with %d channels\n", width, height, channels);

	stbi_image_free(h_input);

	return EXIT_SUCCESS;
}
