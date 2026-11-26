
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <iostream>
#include <string>
#include <vector>
#include <filesystem>

using namespace std;
using namespace cv;

// CUDA Kernel 1: Grayscale Conversion
__global__ void grayscaleKernel(
    unsigned char* input,
    unsigned char* output,
    int width,
    int height,
    int channels) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = width * height;

    if (idx < total) {
        int i = idx * channels;

        unsigned char b = input[i];
        unsigned char g = input[i + 1];
        unsigned char r = input[i + 2];

        output[idx] = 0.114f * b + 0.587f * g + 0.299f * r;
    }
}

// CUDA Kernel 2: Gaussian Blur
__global__ void gaussianKernel(
    unsigned char* input,
    unsigned char* output,
    int width,
    int height) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        float weights[3][3] = {
            {1, 2, 1},
            {2, 4, 2},
            {1, 2, 1}
        };

        float sum = 0.0f;

        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                int px = min(max(x + dx, 0), width - 1);
                int py = min(max(y + dy, 0), height - 1);

                sum += input[py * width + px] *
                       weights[dy + 1][dx + 1];
            }
        }

        output[y * width + x] = (unsigned char)(sum / 16.0f);
    }
}

// CUDA Kernel 3: Edge Detection
__global__ void edgeKernel(
    unsigned char* input,
    unsigned char* output,
    int width,
    int height) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int gx = 0;
        int gy = 0;

        int sx[3][3] = {
            {-1, 0, 1},
            {-2, 0, 2},
            {-1, 0, 1}
        };

        int sy[3][3] = {
            {-1, -2, -1},
            {0, 0, 0},
            {1, 2, 1}
        };

        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                int px = min(max(x + dx, 0), width - 1);
                int py = min(max(y + dy, 0), height - 1);

                int pixel = input[py * width + px];

                gx += pixel * sx[dy + 1][dx + 1];
                gy += pixel * sy[dy + 1][dx + 1];
            }
        }

        int magnitude = abs(gx) + abs(gy);
        output[y * width + x] =
            (unsigned char)min(magnitude, 255);
    }
}

int main(int argc, char* argv[]) {

    string inputFolder = argc > 1 ? argv[1] : "/content/input_images";
    string outputFolder = argc > 2 ? argv[2] : "/content/output_images";

    vector<String> files;
    glob(inputFolder + "/*.jpg", files, false);

    if (files.empty()) {
        cerr << "No JPG images found!" << endl;
        return 1;
    }

    std::filesystem::create_directories(outputFolder);

    int processed = 0;

    for (const auto& file : files) {

        Mat image = imread(file, IMREAD_COLOR);

        if (image.empty()) {
            continue;
        }

        int width = image.cols;
        int height = image.rows;
        int total = width * height;

        Mat gray(height, width, CV_8UC1);
        Mat blur(height, width, CV_8UC1);
        Mat edges(height, width, CV_8UC1);

        unsigned char *d_input, *d_gray, *d_blur, *d_edges;

        cudaMalloc(&d_input, total * 3);
        cudaMalloc(&d_gray, total);
        cudaMalloc(&d_blur, total);
        cudaMalloc(&d_edges, total);

        cudaMemcpy(d_input, image.data, total * 3,
                   cudaMemcpyHostToDevice);

        int threads = 256;
        int blocks = (total + threads - 1) / threads;

        grayscaleKernel<<<blocks, threads>>>(
            d_input, d_gray, width, height, 3);

        dim3 block2D(16, 16);
        dim3 grid2D(
            (width + 15) / 16,
            (height + 15) / 16);

        gaussianKernel<<<grid2D, block2D>>>(
            d_gray, d_blur, width, height);

        edgeKernel<<<grid2D, block2D>>>(
            d_blur, d_edges, width, height);

        cudaDeviceSynchronize();

        cudaMemcpy(gray.data, d_gray, total,
                   cudaMemcpyDeviceToHost);

        cudaMemcpy(blur.data, d_blur, total,
                   cudaMemcpyDeviceToHost);

        cudaMemcpy(edges.data, d_edges, total,
                   cudaMemcpyDeviceToHost);

        string name = to_string(++processed);

        imwrite(outputFolder + "/" + name + "_gray.jpg", gray);
        imwrite(outputFolder + "/" + name + "_blur.jpg", blur);
        imwrite(outputFolder + "/" + name + "_edges.jpg", edges);

        cudaFree(d_input);
        cudaFree(d_gray);
        cudaFree(d_blur);
        cudaFree(d_edges);

        cout << "Processed image " << processed
             << "/" << files.size() << endl;
    }

    cout << "\nTotal images processed: " << processed << endl;

    return 0;
}
