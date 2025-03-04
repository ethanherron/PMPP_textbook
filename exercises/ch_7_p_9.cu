#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Error checking macro
#define CHECK_CUDA_ERROR(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

// Define constant memory for 2D filter
#define MAX_FILTER_SIZE 25  // Support up to 5x5 filter
__constant__ float F_2D[MAX_FILTER_SIZE][MAX_FILTER_SIZE];

// Define constant memory for 3D filter
#define MAX_3D_FILTER_SIZE 9  // Support up to 9x9x9 filter
__constant__ float F_3D[MAX_3D_FILTER_SIZE][MAX_3D_FILTER_SIZE][MAX_3D_FILTER_SIZE];

// 2D convolution kernel
__global__ void convolution_2D_const_mem_kernel(float *N, float *P, int r,
                                                int width, int height) {
    int outCol = blockIdx.x * blockDim.x + threadIdx.x;
    int outRow = blockIdx.y * blockDim.y + threadIdx.y;

    // Check bounds
    if (outCol >= width || outRow >= height)
        return;

    float Pvalue = 0.0f;
    for (int fRow = 0; fRow < 2 * r + 1; fRow++) {
        for (int fCol = 0; fCol < 2 * r + 1; fCol++) {
            int inRow = outRow - r + fRow;
            int inCol = outCol - r + fCol;
            if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width) {
                Pvalue += F_2D[fRow][fCol] * N[inRow * width + inCol];
            }
        }
    }

    P[outRow * width + outCol] = Pvalue;
}


// 3D convolution kernel 
__global__ void convolution_3D_const_mem_kernel(float *N, float *P, int r,
                                                int width, int height, int depth) {
    int outCol = blockIdx.x * blockDim.x + threadIdx.x;
    int outRow = blockIdx.y * blockDim.y + threadIdx.y;
    int outDepth = blockIdx.z * blockDim.z + threadIdx.z;
    
    // Check bounds
    if (outCol >= width || outRow >= height || outDepth >= depth)
        return;
        
    float Pvalue = 0.0f;
    for (int fRow = 0; fRow < 2 * r + 1; fRow++) {
        for (int fCol = 0; fCol < 2 * r + 1; fCol++) {
            for (int fDepth = 0; fDepth < 2 * r + 1; fDepth++) {
                int inRow = outRow - r + fRow;
                int inCol = outCol - r + fCol;
                int inDepth = outDepth - r + fDepth;
                if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width && inDepth >= 0 && inDepth < depth) {
                    Pvalue += F_3D[fRow][fCol][fDepth] * N[inDepth + inCol * depth + inRow * width * depth];
                }
            }
        }
    }

    // Correct 3D output indexing
    P[outDepth + outCol * depth + outRow * width * depth] = Pvalue;
}


// Test function for 3D convolution
void test_3D_convolution() {
    // Define dimensions
    int width = 8;
    int height = 8;
    int depth = 8;
    int filterRadius = 1;
    int filterSize = 2 * filterRadius + 1;
    
    // Allocate host memory
    size_t inputSize = width * height * depth * sizeof(float);
    size_t filterSize3D = filterSize * filterSize * filterSize * sizeof(float);
    size_t outputSize = width * height * depth * sizeof(float);
    
    float *h_input = (float*)malloc(inputSize);
    float *h_filter = (float*)malloc(filterSize3D);
    float *h_output = (float*)malloc(outputSize);
    float *h_verify = (float*)malloc(outputSize);
    
    // Initialize input with simple pattern
    for (int i = 0; i < width * height * depth; i++) {
        h_input[i] = (float)(i % 10);
    }
    
    // Initialize filter (simple box filter)
    for (int i = 0; i < filterSize * filterSize * filterSize; i++) {
        h_filter[i] = 1.0f / (filterSize * filterSize * filterSize);
    }
    
    // Copy filter to constant memory
    float h_filter_3D[MAX_3D_FILTER_SIZE][MAX_3D_FILTER_SIZE][MAX_3D_FILTER_SIZE] = {0};
    for (int i = 0; i < filterSize; i++) {
        for (int j = 0; j < filterSize; j++) {
            for (int k = 0; k < filterSize; k++) {
                h_filter_3D[i][j][k] = h_filter[i * filterSize * filterSize + j * filterSize + k];
            }
        }
    }
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(F_3D, h_filter_3D, sizeof(h_filter_3D)));
    
    // Allocate device memory
    float *d_input, *d_output;
    CHECK_CUDA_ERROR(cudaMalloc(&d_input, inputSize));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, outputSize));
    
    // Copy data to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input, inputSize, cudaMemcpyHostToDevice));
    
    // Define grid and block dimensions
    dim3 blockDim(8, 8, 8);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x,
                 (height + blockDim.y - 1) / blockDim.y,
                 (depth + blockDim.z - 1) / blockDim.z);
    
    // Launch kernel
    convolution_3D_const_mem_kernel<<<gridDim, blockDim>>>(d_input, d_output, 
                                                      filterRadius, width, height, depth);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    
    // Copy result back to host
    CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, outputSize, cudaMemcpyDeviceToHost));
    
    // Verify results (simple check)
    printf("First few output values:\n");
    for (int i = 0; i < 10; i++) {
        printf("output[%d] = %f\n", i, h_output[i]);
    }
    
    // Clean up
    free(h_input);
    free(h_filter);
    free(h_output);
    free(h_verify);
    cudaFree(d_input);
    cudaFree(d_output);
}

// Test function for 2D convolution
void test_2D_convolution() {
    // Define dimensions
    int width = 8;
    int height = 8;
    int filterRadius = 1;
    int filterSize = 2 * filterRadius + 1;
    
    // Allocate host memory
    size_t inputSize = width * height * sizeof(float);
    size_t filterSize2D = filterSize * filterSize * sizeof(float);
    size_t outputSize = width * height * sizeof(float);
    
    float *h_input = (float*)malloc(inputSize);
    float *h_filter = (float*)malloc(filterSize2D);
    float *h_output = (float*)malloc(outputSize);
    
    // Initialize input with simple pattern
    for (int i = 0; i < width * height; i++) {
        h_input[i] = (float)(i % 10);
    }
    
    // Initialize filter (simple box filter)
    for (int i = 0; i < filterSize * filterSize; i++) {
        h_filter[i] = 1.0f / (filterSize * filterSize);
    }
    
    // Copy filter to constant memory
    float h_filter_2D[MAX_FILTER_SIZE][MAX_FILTER_SIZE] = {0};
    for (int i = 0; i < filterSize; i++) {
        for (int j = 0; j < filterSize; j++) {
            h_filter_2D[i][j] = h_filter[i * filterSize + j];
        }
    }
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(F_2D, h_filter_2D, sizeof(h_filter_2D)));
    
    // Allocate device memory
    float *d_input, *d_output;
    CHECK_CUDA_ERROR(cudaMalloc(&d_input, inputSize));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, outputSize));
    
    // Copy data to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input, inputSize, cudaMemcpyHostToDevice));
    
    // Define grid and block dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x,
                 (height + blockDim.y - 1) / blockDim.y);
    
    // Launch kernel
    convolution_2D_const_mem_kernel<<<gridDim, blockDim>>>(d_input, d_output, 
                                                      filterRadius, width, height);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    
    // Copy result back to host
    CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, outputSize, cudaMemcpyDeviceToHost));
    
    // Verify results (simple check)
    printf("First few 2D output values:\n");
    for (int i = 0; i < 10; i++) {
        printf("output[%d] = %f\n", i, h_output[i]);
    }
    
    // Clean up
    free(h_input);
    free(h_filter);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);
}

int main() {
    test_2D_convolution();
    test_3D_convolution();
    printf("Convolution tests completed!\n");
    return 0;
}