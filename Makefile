CXX = g++
CXXFLAGS = -O3 -Wall -shared -std=c++11 -fPIC
TARGET = quad example
SRC = quad.cpp example.cpp

BUILD_DIR := build
SRC_DIR := src
PYBIND11_INCLUDES := $(shell python3 -m pybind11 --includes)
OPENCV_CFLAGS := $(shell pkg-config --cflags opencv4)
OPENCV_LIBS := $(shell pkg-config --libs opencv4)

# Generate object file names for each source file
OBJ = $(patsubst %.cpp,$(BUILD_DIR)/%.o,$(SRC))

# Build all targets
all: $(addprefix $(BUILD_DIR)/, $(TARGET))

# Build the shared library
$(BUILD_DIR)/%: $(BUILD_DIR)/%.o
	$(CXX) $(CXXFLAGS) $< -o $@$(shell python3-config --extension-suffix) $(OPENCV_LIBS)

# Build object files from source files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(OPENCV_CFLAGS) -c -o $@ $< $(PYBIND11_INCLUDES)

# Create the build directory if it doesn't exist
$(BUILD_DIR):
	mkdir -p $@

# Clean up generated files
clean:
	rm -rf $(BUILD_DIR)

.PHONY: all clean
