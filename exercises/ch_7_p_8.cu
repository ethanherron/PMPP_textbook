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

// 2D convolution kernel (fixed)
__global__ void convolution_2D_basic_kernel(float *N, float *F, float *P,
                                           int r, int width, int height) {
    int outCol = blockIdx.x * blockDim.x + threadIdx.x;
    int outRow = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (outCol >= width || outRow >= height)
        return;

    float Pvalue = 0.0f;
    int filterSize = 2 * r + 1;
    
    for (int fRow = 0; fRow < filterSize; fRow++) {
        for (int fCol = 0; fCol < filterSize; fCol++) {
            int inRow = outRow - r + fRow;
            int inCol = outCol - r + fCol;
            if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width) {
                // Correct 1D indexing for filter F
                int fIndex = fRow * filterSize + fCol;
                Pvalue += F[fIndex] * N[inRow * width + inCol];
            }
        }
    }

    P[outRow * width + outCol] = Pvalue;
}

// 3D convolution kernel (fixed)
__global__ void convolution_3D_basic_kernel(float *N, float *F, float *P,
                                           int r, int width, int height, int depth) {
    int outCol = blockIdx.x * blockDim.x + threadIdx.x;
    int outRow = blockIdx.y * blockDim.y + threadIdx.y;
    int outDepth = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (outCol >= width || outRow >= height || outDepth >= depth)
        return;

    float Pvalue = 0.0f;
    int filterSize = 2 * r + 1;
    
    for (int fDepth = 0; fDepth < filterSize; fDepth++) {
        for (int fRow = 0; fRow < filterSize; fRow++) {
            for (int fCol = 0; fCol < filterSize; fCol++) {
                int inRow = outRow - r + fRow;
                int inCol = outCol - r + fCol;
                int inDepth = outDepth - r + fDepth;
                
                if (inRow >= 0 && inRow < height && 
                    inCol >= 0 && inCol < width && 
                    inDepth >= 0 && inDepth < depth) {
                    
                    // Correct 1D indexing for filter F
                    int fIndex = fDepth * filterSize * filterSize + 
                                fRow * filterSize + fCol;
                    
                    // Correct 3D indexing for input N
                    int nIndex = inDepth + 
                                inCol * depth + 
                                inRow * width * depth;
                    
                    Pvalue += F[fIndex] * N[nIndex];
                }
            }
        }
    }

    // Calculate output index
    int pIndex = outDepth + outCol * depth + outRow * width * depth;
    P[pIndex] = Pvalue;
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
    
    // Allocate device memory
    float *d_input, *d_filter, *d_output;
    CHECK_CUDA_ERROR(cudaMalloc(&d_input, inputSize));
    CHECK_CUDA_ERROR(cudaMalloc(&d_filter, filterSize3D));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, outputSize));
    
    // Copy data to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input, inputSize, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_filter, h_filter, filterSize3D, cudaMemcpyHostToDevice));
    
    // Define grid and block dimensions
    dim3 blockDim(8, 8, 8);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x,
                 (height + blockDim.y - 1) / blockDim.y,
                 (depth + blockDim.z - 1) / blockDim.z);
    
    // Launch kernel
    convolution_3D_basic_kernel<<<gridDim, blockDim>>>(d_input, d_filter, d_output, 
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
    cudaFree(d_filter);
    cudaFree(d_output);
}

int main() {
    test_3D_convolution();
    printf("3D convolution test completed!\n");
    return 0;
}