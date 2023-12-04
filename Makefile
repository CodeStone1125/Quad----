CXX = g++
CXXFLAGS = -O3 -Wall -shared -std=c++17 -fPIC
TARGET = quad
SRC = quad.cpp
TEST_SRC = quad.cpp

BUILD_DIR := build
SRC_DIR := src
TEST_DIR := tests
PYBIND11_INCLUDES := $(shell python3 -m pybind11 --includes)
OPENCV_CFLAGS := $(shell pkg-config --cflags opencv4)
OPENCV_LIBS := $(shell pkg-config --libs opencv4)

OBJ = $(patsubst %.cpp,$(BUILD_DIR)/%.o,$(SRC))

all: $(addprefix $(BUILD_DIR)/, $(TARGET))

$(BUILD_DIR)/$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) $< -o $@$(shell python3-config --extension-suffix) $(OPENCV_LIBS)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(OPENCV_CFLAGS) -c -o $@ $< $(PYBIND11_INCLUDES)

$(BUILD_DIR):
	mkdir -p $@

clean:
	rm -rf $(BUILD_DIR)
	rm -rf *png *jpg


.PHONY: all clean test
