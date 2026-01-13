# Claude generated makefile, to be honest...

# Directories
SRC_DIR = src
BUILD_DIR = build

# Compiler settings
NVCC = nvcc
CC = gcc

# Compiler flags
NVCC_FLAGS = -O2 -arch=sm_75 -I$(SRC_DIR)
CFLAGS = -O2 -c

# Target executable
TARGET = blackhole

# Source files
CU_SOURCES = $(SRC_DIR)/main.cu $(SRC_DIR)/raytrace.cu

# Object files
CU_OBJECTS = $(patsubst $(SRC_DIR)/%.cu,$(BUILD_DIR)/%.o,$(CU_SOURCES))

# Default target
all: $(TARGET)

# Link object files into executable
$(TARGET): $(CU_OBJECTS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^

# Compile CUDA source files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Create build directory
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Run the program (requires input/output image paths)
run: $(TARGET)
	@echo "Usage: ./$(TARGET) <input_image> <output_image>"
	@echo "Example: ./$(TARGET) input.png output.png"

# Clean build artifacts
clean:
	rm -f $(TARGET) $(CU_OBJECTS)
	rm -rf $(BUILD_DIR)

# Download stb headers if not present
setup:
	@if [ ! -f $(SRC_DIR)/stb_image.h ]; then \
		echo "Downloading stb_image.h..."; \
		curl -o $(SRC_DIR)/stb_image.h https://raw.githubusercontent.com/nothings/stb/master/stb_image.h; \
		echo "stb_image.h downloaded successfully!"; \
	else \
		echo "stb_image.h already exists."; \
	fi
	@if [ ! -f $(SRC_DIR)/stb_image_write.h ]; then \
		echo "Downloading stb_image_write.h..."; \
		curl -o $(SRC_DIR)/stb_image_write.h https://raw.githubusercontent.com/nothings/stb/master/stb_image_write.h; \
		echo "stb_image_write.h downloaded successfully!"; \
	else \
		echo "stb_image_write.h already exists."; \
	fi

.PHONY: all run clean setup

