# GPU Accelerated Batch Image Processing Using CUDA

## Project Overview

This project demonstrates GPU-accelerated batch image processing using NVIDIA CUDA C++.

It processes a collection of computer mouse images using three parallel image-processing operations: grayscale conversion, Gaussian blur, and Sobel edge detection.

The image-processing computations are performed on an NVIDIA GPU, while OpenCV is used for image loading and saving.

## Technologies Used

- CUDA C++ 13.0
- NVIDIA Tesla T4 GPU
- OpenCV 4.6.0
- Google Colab
- C++17
- Makefile and Bash

## Image Processing Operations

1. **Grayscale Conversion:** Converts colour images into grayscale using the weighted BGR intensity formula.
2. **Gaussian Blur:** Applies a 3 × 3 Gaussian filter to smooth images and reduce noise.
3. **Sobel Edge Detection:** Detects image boundaries using horizontal and vertical Sobel operators.

## Dataset

- Dataset category: Computer mouse images
- Number of input images: 100
- Image format: JPG
- Each image is processed using all three CUDA operations.

The original input images are not included in this repository. Users should provide their own JPG images in the `input_images` folder.

## Project Structure

```text
CUDA-Batch-Image-Processing/
├── image_processing.cu
├── README.md
├── Makefile
├── run.sh
├── input_images/          # Add JPG input images here
└── output_images/         # Generated automatically
```

## Requirements

- NVIDIA GPU with CUDA support
- CUDA Toolkit and NVCC compiler
- OpenCV development libraries
- C++17-compatible compiler
- Make and Bash

## How to Run

### 1. Prepare the input folder

Create a folder named `input_images` and place your JPG images inside it.

### 2. Compile the project

```bash
make
```

### 3. Run the program

```bash
bash run.sh input_images output_images
```

You can also run the compiled program directly:

```bash
./image_processing input_images output_images
```

The program accepts two command-line arguments:

- First argument: Input image folder
- Second argument: Output image folder

### 4. View the results

The program creates three output images for each successfully processed input image:

- `_gray.jpg` – Grayscale result
- `_blur.jpg` – Gaussian blur result
- `_edges.jpg` – Sobel edge detection result

## Execution Results

The project was executed in Google Colab using an NVIDIA Tesla T4 GPU.

| Metric | Result |
|---|---:|
| Input images processed | 100 |
| Grayscale images generated | 100 |
| Gaussian blur images generated | 100 |
| Edge detection images generated | 100 |
| Total output images | 300 |

## Execution Evidence

The `execution_evidence.zip` file contains the generated output images and an execution summary.

## Learning Outcomes

- Understanding GPU parallel processing using CUDA.
- Implementing custom CUDA kernels.
- Using CUDA memory allocation and data transfer.
- Applying grayscale, Gaussian blur, and Sobel edge detection.
- Integrating CUDA C++ with OpenCV.
- Compiling and executing CUDA applications in Google Colab.

## Challenges Faced

- Setting up the CUDA environment in Google Colab.
- Integrating OpenCV with CUDA C++.
- Resolving compilation and execution issues.
- Managing GPU memory and transferring image data.
- Processing multiple images in a single program execution.

## Conclusion

This project demonstrates how CUDA can be used to accelerate image-processing operations through GPU parallelism. It successfully processed 100 computer mouse images and generated 300 processed outputs using three custom CUDA kernels.
