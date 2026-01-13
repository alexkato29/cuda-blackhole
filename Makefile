# Claude generated makefile, to be honest...

# Compiler settings
NVCC = nvcc
CC = gcc

# Compiler flags
NVCC_FLAGS = -O2 -arch=sm_75
CFLAGS = -O2 -c

# Target executable
TARGET = blackhole

# Source files
CU_SOURCES = main.cu
C_SOURCES = stb_image.c

# Object files
CU_OBJECTS = $(CU_SOURCES:.cu=.o)
C_OBJECTS = $(C_SOURCES:.c=.o)

# Default target
all: $(TARGET)

# Link object files into executable
$(TARGET): $(CU_OBJECTS) $(C_OBJECTS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^

# Compile CUDA source files
%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Compile C source files
%.o: %.c
	$(CC) $(CFLAGS) $< -o $@

# Run the program (requires input/output image paths)
run: $(TARGET)
	@echo "Usage: ./$(TARGET) <input_image> <output_image>"
	@echo "Example: ./$(TARGET) input.png output.png"

# Clean build artifacts
clean:
	rm -f $(TARGET) $(CU_OBJECTS) $(C_OBJECTS)

# Download stb_image.h if not present
setup:
	@if [ ! -f stb_image.h ]; then \
		echo "Downloading stb_image.h..."; \
		curl -o stb_image.h https://raw.githubusercontent.com/nothings/stb/master/stb_image.h; \
		echo "stb_image.h downloaded successfully!"; \
	else \
		echo "stb_image.h already exists."; \
	fi

.PHONY: all run clean setup
