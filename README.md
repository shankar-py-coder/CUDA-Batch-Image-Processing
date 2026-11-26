
# GPU Accelerated Batch Image Processing Using CUDA

## Project Description
This project uses NVIDIA CUDA C++ to process 100 mouse images in batches using GPU parallel computing.

## Technologies Used
- CUDA C++ 13.0
- NVIDIA Tesla T4 GPU
- OpenCV 4.6.0
- Google Colab

## Image Processing Operations

1. **Grayscale Conversion:** Converts colour images into grayscale images.
2. **Gaussian Blur:** Applies a 3x3 Gaussian filter to smooth images and reduce noise.
3. **Edge Detection:** Uses the Sobel operator to identify image boundaries.

## Dataset
The project uses 100 JPG images of computer mice.

## Project Structure

```text
CUDA-Batch-Image-Processing/
├── image_processing.cu
├── README.md
├── Makefile
├── run.sh
├── input_images/
└── output_images/
