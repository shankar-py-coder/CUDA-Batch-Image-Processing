#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t error = (call);                                        \
        if (error != cudaSuccess) {                                        \
            std::cerr << "CUDA Error: " << cudaGetErrorString(error)       \
                      << " at line " << __LINE__ << std::endl;              \
            std::exit(EXIT_FAILURE);                                       \
        }                                                                  \
    } while (0)

// Converts a colour image into grayscale.
__global__ void GrayscaleKernel(
    const unsigned char* input,
    unsigned char* output,
    int width,
    int height,
    int channels) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int totalPixels = width * height;

    if (index < totalPixels) {
        int inputIndex = index * channels;

        unsigned char blue = input[inputIndex];
        unsigned char green = input[inputIndex + 1];
        unsigned char red = input[inputIndex + 2];

        output[index] = static_cast<unsigned char>(
            0.114f * blue + 0.587f * green + 0.299f * red);
    }
}

// Applies a 3x3 Gaussian blur filter.
__global__ void GaussianKernel(
    const unsigned char* input,
    unsigned char* output,
    int width,
    int height) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        const float weights[3][3] = {
            {1, 2, 1},
            {2, 4, 2},
            {1, 2, 1}
        };

        float sum = 0.0f;

        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                int pixelX = max(0, min(x + dx, width - 1));
                int pixelY = max(0, min(y + dy, height - 1));

                sum += input[pixelY * width + pixelX] *
                       weights[dy + 1][dx + 1];
            }
        }

        output[y * width + x] =
            static_cast<unsigned char>(sum / 16.0f);
    }
}

// Detects edges using horizontal and vertical Sobel operators.
__global__ void EdgeKernel(
    const unsigned char* input,
    unsigned char* output,
    int width,
    int height) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        const int sobelX[3][3] = {
            {-1, 0, 1},
            {-2, 0, 2},
            {-1, 0, 1}
        };

        const int sobelY[3][3] = {
            {-1, -2, -1},
            {0, 0, 0},
            {1, 2, 1}
        };

        int gradientX = 0;
        int gradientY = 0;

        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                int pixelX = max(0, min(x + dx, width - 1));
                int pixelY = max(0, min(y + dy, height - 1));

                int pixel = input[pixelY * width + pixelX];

                gradientX += pixel * sobelX[dy + 1][dx + 1];
                gradientY += pixel * sobelY[dy + 1][dx + 1];
            }
        }

        int magnitude = abs(gradientX) + abs(gradientY);

        output[y * width + x] =
            static_cast<unsigned char>(min(magnitude, 255));
    }
}

int main(int argc, char* argv[]) {
    std::string inputFolder =
        argc > 1 ? argv[1] : "/content/input_images";

    std::string outputFolder =
        argc > 2 ? argv[2] : "/content/output_images";

    std::vector<cv::String> imageFiles;

    cv::glob(inputFolder + "/*.jpg", imageFiles, false);

    if (imageFiles.empty()) {
        std::cerr << "No JPG images found in: "
                  << inputFolder << std::endl;
        return EXIT_FAILURE;
    }

    std::filesystem::create_directories(outputFolder);

    int processedImages = 0;

    for (const auto& imageFile : imageFiles) {
        cv::Mat image = cv::imread(imageFile, cv::IMREAD_COLOR);

        if (image.empty()) {
            std::cerr << "Could not read: " << imageFile << std::endl;
            continue;
        }

        // Ensure image data is stored continuously in memory.
        if (!image.isContinuous()) {
            image = image.clone();
        }

        int width = image.cols;
        int height = image.rows;
        int totalPixels = width * height;
        size_t colorBytes =
            static_cast<size_t>(totalPixels) * 3;
        size_t grayBytes =
            static_cast<size_t>(totalPixels);

        cv::Mat gray(height, width, CV_8UC1);
        cv::Mat blur(height, width, CV_8UC1);
        cv::Mat edges(height, width, CV_8UC1);

        unsigned char* deviceInput = nullptr;
        unsigned char* deviceGray = nullptr;
        unsigned char* deviceBlur = nullptr;
        unsigned char* deviceEdges = nullptr;

        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void**>(&deviceInput), colorBytes));

        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void**>(&deviceGray), grayBytes));

        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void**>(&deviceBlur), grayBytes));

        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void**>(&deviceEdges), grayBytes));

        CUDA_CHECK(cudaMemcpy(
            deviceInput, image.data, colorBytes,
            cudaMemcpyHostToDevice));

        // Launch grayscale kernel.
        int threads = 256;
        int blocks = (totalPixels + threads - 1) / threads;

        GrayscaleKernel<<<blocks, threads>>>(
            deviceInput, deviceGray, width, height, 3);

        CUDA_CHECK(cudaGetLastError());

        // Launch Gaussian blur and Sobel kernels.
        dim3 blockSize(16, 16);
        dim3 gridSize(
            (width + 15) / 16,
            (height + 15) / 16);

        GaussianKernel<<<gridSize, blockSize>>>(
            deviceGray, deviceBlur, width, height);

        CUDA_CHECK(cudaGetLastError());

        EdgeKernel<<<gridSize, blockSize>>>(
            deviceBlur, deviceEdges, width, height);

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Copy processed images back to CPU memory.
        CUDA_CHECK(cudaMemcpy(
            gray.data, deviceGray, grayBytes,
            cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaMemcpy(
            blur.data, deviceBlur, grayBytes,
            cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaMemcpy(
            edges.data, deviceEdges, grayBytes,
            cudaMemcpyDeviceToHost));

        std::string imageName =
            std::to_string(processedImages + 1);

        bool graySaved = cv::imwrite(
            outputFolder + "/" + imageName + "_gray.jpg", gray);

        bool blurSaved = cv::imwrite(
            outputFolder + "/" + imageName + "_blur.jpg", blur);

        bool edgesSaved = cv::imwrite(
            outputFolder + "/" + imageName + "_edges.jpg", edges);

        // Release GPU memory.
        CUDA_CHECK(cudaFree(deviceInput));
        CUDA_CHECK(cudaFree(deviceGray));
        CUDA_CHECK(cudaFree(deviceBlur));
        CUDA_CHECK(cudaFree(deviceEdges));

        if (!graySaved || !blurSaved || !edgesSaved) {
            std::cerr << "Could not save all outputs for "
                      << imageFile << std::endl;
            continue;
        }

        ++processedImages;

        std::cout << "Processed image "
                  << processedImages << "/"
                  << imageFiles.size() << std::endl;
    }

    std::cout << "\nTotal images processed: "
              << processedImages << std::endl;

    return EXIT_SUCCESS;
}
