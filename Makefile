
NVCC = nvcc
TARGET = image_processing
SOURCE = image_processing.cu

CXXFLAGS = -std=c++17
OPENCV_FLAGS = $(shell pkg-config --cflags --libs opencv4)

all:
	$(NVCC) $(CXXFLAGS) -o $(TARGET) $(SOURCE) $(OPENCV_FLAGS)

clean:
	rm -f $(TARGET)
