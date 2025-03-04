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

// Define filter radius and tile dimensions
#define FILTER_RADIUS 1
#define IN_TILE_DIM 32
#define OUT_TILE_DIM ((IN_TILE_DIM) - 2 * (FILTER_RADIUS))

// Define constant memory for 2D filter
__constant__ float F_c_2D[2 * FILTER_RADIUS + 1][2 * FILTER_RADIUS + 1];

__global__ void convolution_tiled_2D_const_mem_kernel(float *N, float *P,
                                                      int width, int height) {
    int col = blockIdx.x * OUT_TILE_DIM + threadIdx.x - FILTER_RADIUS;
    int row = blockIdx.y * OUT_TILE_DIM + threadIdx.y - FILTER_RADIUS;

    // Loading input tile
    __shared__ float N_s[IN_TILE_DIM][IN_TILE_DIM];

    if (row >= 0 && row < height && col >= 0 && col < width) {
        N_s[threadIdx.y][threadIdx.x] = N[row * width + col];
    } else {
        N_s[threadIdx.y][threadIdx.x] = 0.0;
    }

    __syncthreads();

    // Calculating output elements
    int tileCol = threadIdx.x - FILTER_RADIUS;
    int tileRow = threadIdx.y - FILTER_RADIUS;

    // Turning off the threads at the edges of the block
    if (col >= 0 && col < width && row >= 0 && row < height) {
        if (tileCol >= 0 && tileCol < OUT_TILE_DIM && tileRow >= 0 && tileRow < OUT_TILE_DIM) {
            float Pvalue = 0.0f;
            for (int fRow = 0; fRow < 2 * FILTER_RADIUS + 1; fRow++) {
                for (int fCol = 0; fCol < 2 * FILTER_RADIUS + 1; fCol++) {
                    Pvalue += F_c_2D[fRow][fCol] * N_s[tileRow + fRow][tileCol + fCol];
                }
            }
            P[row * width + col] = Pvalue;
        }
    }
}


// 3D convolution kernel 
#define IN_TILE_DIM_3D 8
#define OUT_TILE_DIM_3D ((IN_TILE_DIM_3D) - 2 * (FILTER_RADIUS))

// Define constant memory for 3D filter
__constant__ float F_c_3D[2 * FILTER_RADIUS + 1][2 * FILTER_RADIUS + 1][2 * FILTER_RADIUS + 1];

__global__ void convolution_tiled_3D_const_mem_kernel(float *N, float *P,
                                                      int width, int height, int depth) {
    int col = blockIdx.x * OUT_TILE_DIM_3D + threadIdx.x - FILTER_RADIUS;
    int row = blockIdx.y * OUT_TILE_DIM_3D + threadIdx.y - FILTER_RADIUS;
    int z = blockIdx.z * OUT_TILE_DIM_3D + threadIdx.z - FILTER_RADIUS;
    
    // Loading input tile
    __shared__ float N_s[IN_TILE_DIM_3D][IN_TILE_DIM_3D][IN_TILE_DIM_3D];

    if (row >= 0 && row < height && col >= 0 && col < width && z >= 0 && z < depth) {
        N_s[threadIdx.z][threadIdx.y][threadIdx.x] = N[z * width * height + row * width + col];
    } else {
        N_s[threadIdx.z][threadIdx.y][threadIdx.x] = 0.0;
    }

    __syncthreads();

    // Calculating output elements
    int tileCol = threadIdx.x - FILTER_RADIUS;
    int tileRow = threadIdx.y - FILTER_RADIUS;
    int tileDepth = threadIdx.z - FILTER_RADIUS;

    // Turning off the threads at the edges of the block
    if (col >= 0 && col < width && row >= 0 && row < height && z >= 0 && z < depth) {
        if (tileCol >= 0 && tileCol < OUT_TILE_DIM_3D && tileRow >= 0 && tileRow < OUT_TILE_DIM_3D && tileDepth >= 0 && tileDepth < OUT_TILE_DIM_3D) {
            float Pvalue = 0.0f;
            for (int fRow = 0; fRow < 2 * FILTER_RADIUS + 1; fRow++) {
                for (int fCol = 0; fCol < 2 * FILTER_RADIUS + 1; fCol++) {
                    for (int fDepth = 0; fDepth < 2 * FILTER_RADIUS + 1; fDepth++) {
                        Pvalue += F_c_3D[fDepth][fRow][fCol] * N_s[tileDepth + fDepth][tileRow + fRow][tileCol + fCol];
                    }
                }
            }
            P[z * width * height + row * width + col] = Pvalue;
        }
    }
}


// Test function for 3D convolution
void test_3D_convolution() {
    // Define dimensions
    int width = 64;
    int height = 64;
    int depth = 64;
    
    // Allocate host memory
    size_t inputSize = width * height * depth * sizeof(float);
    size_t filterSize3D = (2 * FILTER_RADIUS + 1) * (2 * FILTER_RADIUS + 1) * (2 * FILTER_RADIUS + 1) * sizeof(float);
    size_t outputSize = width * height * depth * sizeof(float);
    
    float *h_input = (float*)malloc(inputSize);
    float *h_filter = (float*)malloc(filterSize3D);
    float *h_output = (float*)malloc(outputSize);
    
    // Initialize input with simple pattern
    for (int i = 0; i < width * height * depth; i++) {
        h_input[i] = (float)(i % 10);
    }
    
    // Initialize filter (simple box filter)
    int filterSize = 2 * FILTER_RADIUS + 1;
    for (int i = 0; i < filterSize * filterSize * filterSize; i++) {
        h_filter[i] = 1.0f / (filterSize * filterSize * filterSize);
    }
    
    // Copy filter to constant memory
    float h_filter_3D[2 * FILTER_RADIUS + 1][2 * FILTER_RADIUS + 1][2 * FILTER_RADIUS + 1] = {0};
    for (int i = 0; i < filterSize; i++) {
        for (int j = 0; j < filterSize; j++) {
            for (int k = 0; k < filterSize; k++) {
                h_filter_3D[i][j][k] = h_filter[i * filterSize * filterSize + j * filterSize + k];
            }
        }
    }
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(F_c_3D, h_filter_3D, sizeof(h_filter_3D)));
    
    // Allocate device memory
    float *d_input, *d_output;
    CHECK_CUDA_ERROR(cudaMalloc(&d_input, inputSize));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, outputSize));
    
    // Copy data to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input, inputSize, cudaMemcpyHostToDevice));
    
    // Define grid and block dimensions for tiled kernel
    dim3 blockDim(IN_TILE_DIM_3D, IN_TILE_DIM_3D, IN_TILE_DIM_3D);
    dim3 gridDim(
        (width + OUT_TILE_DIM_3D - 1) / OUT_TILE_DIM_3D,
        (height + OUT_TILE_DIM_3D - 1) / OUT_TILE_DIM_3D,
        (depth + OUT_TILE_DIM_3D - 1) / OUT_TILE_DIM_3D
    );
    
    // Launch kernel
    convolution_tiled_3D_const_mem_kernel<<<gridDim, blockDim>>>(d_input, d_output, width, height, depth);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    
    // Copy result back to host
    CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, outputSize, cudaMemcpyDeviceToHost));
    
    // Verify results (simple check)
    printf("First few 3D output values:\n");
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

// Test function for 2D convolution
void test_2D_convolution() {
    // Define dimensions
    int width = 64;
    int height = 64;
    
    // Allocate host memory
    size_t inputSize = width * height * sizeof(float);
    size_t filterSize2D = (2 * FILTER_RADIUS + 1) * (2 * FILTER_RADIUS + 1) * sizeof(float);
    size_t outputSize = width * height * sizeof(float);
    
    float *h_input = (float*)malloc(inputSize);
    float *h_filter = (float*)malloc(filterSize2D);
    float *h_output = (float*)malloc(outputSize);
    
    // Initialize input with simple pattern
    for (int i = 0; i < width * height; i++) {
        h_input[i] = (float)(i % 10);
    }
    
    // Initialize filter (simple box filter)
    int filterSize = 2 * FILTER_RADIUS + 1;
    for (int i = 0; i < filterSize * filterSize; i++) {
        h_filter[i] = 1.0f / (filterSize * filterSize);
    }
    
    // Copy filter to constant memory
    float h_filter_2D[2 * FILTER_RADIUS + 1][2 * FILTER_RADIUS + 1] = {0};
    for (int i = 0; i < filterSize; i++) {
        for (int j = 0; j < filterSize; j++) {
            h_filter_2D[i][j] = h_filter[i * filterSize + j];
        }
    }
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(F_c_2D, h_filter_2D, sizeof(h_filter_2D)));
    
    // Allocate device memory
    float *d_input, *d_output;
    CHECK_CUDA_ERROR(cudaMalloc(&d_input, inputSize));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, outputSize));
    
    // Copy data to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input, inputSize, cudaMemcpyHostToDevice));
    
    // Define grid and block dimensions for tiled kernel
    dim3 blockDim(IN_TILE_DIM, IN_TILE_DIM);
    dim3 gridDim(
        (width + OUT_TILE_DIM - 1) / OUT_TILE_DIM,
        (height + OUT_TILE_DIM - 1) / OUT_TILE_DIM
    );
    
    // Launch kernel
    convolution_tiled_2D_const_mem_kernel<<<gridDim, blockDim>>>(d_input, d_output, width, height);
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
    printf("Tiled convolution tests completed!\n");
    return 0;
}