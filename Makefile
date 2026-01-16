# Directories
SRC_DIR = src
BUILD_DIR = build

# Compiler
NVCC = nvcc

# Compiler flags
NVCC_FLAGS = -O2 -arch=sm_75 -I$(SRC_DIR)

# Target executable
TARGET = blackhole

# Source files
CU_SOURCES = $(SRC_DIR)/main.cu $(SRC_DIR)/blackhole.cu $(SRC_DIR)/helpers.cu

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

.PHONY: all clean setup
