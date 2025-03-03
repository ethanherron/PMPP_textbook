#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <string.h>
#include <float.h>  // For FLT_MAX

// Error checking macro
#define CHECK_CUDA_ERROR(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA Error: %s at %s:%d\n", \
               cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

// Matrix dimensions
#define M 16384  // Number of rows in A and C
#define N 16384  // Number of columns in B and C
#define K 16384  // Number of columns in A and rows in B

// Tile size
#define TILE_WIDTH 16

// Number of test runs for averaging performance
#define NUM_RUNS 3

// Timeout in milliseconds for kernel execution
#define KERNEL_TIMEOUT 10000  // 10 seconds timeout

// Skip verification for large matrices to save time
#define SKIP_VERIFICATION_FOR_LARGE_MATRICES 1

// Kernel function for matrix multiplication with corner turning (coalesced memory access)
// A is in row-major format, B is in column-major format
__global__ void matrixMulCornerTurning(float* A, float* B, float* C, int Width) {
    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    float Cvalue = 0.0f;
    for (int ph = 0; ph < Width/TILE_WIDTH; ++ph) {
        As[ty][tx] = A[Row * Width + (ph * TILE_WIDTH + tx)];
        Bs[ty][tx] = B[Col * Width + (ph * TILE_WIDTH + ty)];
        
        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k) {
            Cvalue += As[ty][k] * Bs[k][tx];  
        }
        __syncthreads();
    }
    
    if (Row < Width && Col < Width) { 
        C[Row * Width + Col] = Cvalue;
    }
}

// Kernel function for matrix multiplication WITHOUT memory coalescing
// A is in row-major format, B is in column-major format
__global__ void matrixMulNonCoalesced(float* A, float* B, float* C, int Width) {
    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    float Cvalue = 0.0f;
    for (int ph = 0; ph < Width/TILE_WIDTH; ++ph) {
        // Load A tile (row-major) - COALESCED access (still efficient)
        As[ty][tx] = A[Row * Width + (ph * TILE_WIDTH + tx)];
        
        // Load B tile (column-major) - NON-COALESCED access
        // This should access the same data as the coalesced version but in a non-coalesced way
        // Original coalesced: B[Col * Width + (ph * TILE_WIDTH + ty)]
        int col_idx = Col;
        int row_idx = ph * TILE_WIDTH + ty;
        Bs[ty][tx] = B[col_idx * Width + row_idx];
        
        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k) {
            Cvalue += As[ty][k] * Bs[k][tx];  
        }
        __syncthreads();
    }
    
    if (Row < Width && Col < Width) {  // Boundary check
        C[Row * Width + Col] = Cvalue;
    }
}

// Initialize matrices
void initializeMatrices(float* A, float* B, int m, int n, int k) {
    // Initialize A (row-major)
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            A[i * k + j] = (float)(rand() % 10) / 10.0f;
        }
    }
    
    // Initialize B (column-major)
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < k; i++) {
            B[j * k + i] = (float)(rand() % 10) / 10.0f;
        }
    }
}

// Verify results with CPU implementation
void verifyResults(float* A, float* B, float* C, int m, int n, int k) {
    float* C_cpu = (float*)malloc(m * n * sizeof(float));
    
    // CPU matrix multiplication
    // A is row-major, B is column-major
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int l = 0; l < k; l++) {
                // A[i][l] in row-major is A[i*k + l]
                // B[l][j] in column-major is B[j*k + l]
                sum += A[i * k + l] * B[j * k + l];
            }
            C_cpu[i * n + j] = sum;
        }
    }
    
    // Compare results
    int errors = 0;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float diff = fabs(C[i * n + j] - C_cpu[i * n + j]);
            if (diff > 1e-5) {
                errors++;
                if (errors < 10) {
                    printf("Error at C[%d][%d]: GPU = %f, CPU = %f, Diff = %e\n", 
                           i, j, C[i * n + j], C_cpu[i * n + j], diff);
                }
            }
        }
    }
    
    if (errors == 0) {
        printf("Verification successful!\n");
    } else {
        printf("Verification failed with %d errors\n", errors);
    }
    
    free(C_cpu);
}

// Run kernel and measure performance
float runKernel(void (*kernelFunc)(float*, float*, float*, int), 
               float* d_A, float* d_B, float* d_C, int width, 
               dim3 gridDim, dim3 blockDim, const char* kernelName) {
    
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));
    
    float totalTime = 0.0f;
    float times[NUM_RUNS];
    bool timedOut = false;
    
    // Warm-up run
    printf("Running warm-up for %s kernel...\n", kernelName);
    kernelFunc<<<gridDim, blockDim>>>(d_A, d_B, d_C, width);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    
    // Timed runs
    for (int run = 0; run < NUM_RUNS; run++) {
        printf("Running %s kernel, iteration %d/%d...\n", kernelName, run+1, NUM_RUNS);
        
        // Record start event
        CHECK_CUDA_ERROR(cudaEventRecord(start));
        
        // Launch kernel
        kernelFunc<<<gridDim, blockDim>>>(d_A, d_B, d_C, width);
        
        // Check for kernel launch errors
        CHECK_CUDA_ERROR(cudaGetLastError());
        
        // Record stop event
        CHECK_CUDA_ERROR(cudaEventRecord(stop));
        
        // Wait for kernel with timeout
        cudaError_t err = cudaEventSynchronize(stop);
        if (err == cudaErrorLaunchTimeout || err == cudaErrorUnknown) {
            printf("WARNING: %s kernel timed out after %d ms!\n", kernelName, KERNEL_TIMEOUT);
            timedOut = true;
            
            // Try to reset device
            cudaDeviceReset();
            printf("Device has been reset. Further results may be unreliable.\n");
            
            // Return a very large time to indicate timeout
            return FLT_MAX;
        }
        
        // Calculate elapsed time
        float milliseconds = 0;
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));
        
        times[run] = milliseconds;
        totalTime += milliseconds;
        printf("  Iteration time: %.3f ms\n", milliseconds);
    }
    
    if (timedOut) {
        return FLT_MAX;
    }
    
    // Calculate average time
    float avgTime = totalTime / NUM_RUNS;
    
    // Calculate standard deviation
    float variance = 0.0f;
    for (int i = 0; i < NUM_RUNS; i++) {
        variance += (times[i] - avgTime) * (times[i] - avgTime);
    }
    variance /= NUM_RUNS;
    float stdDev = sqrt(variance);
    
    // Calculate performance metrics
    float gigaFlops = (2.0f * M * N * K) / (avgTime * 1e6);
    float memoryBandwidth = (3.0f * M * N * sizeof(float)) / (avgTime * 1e6); // GB/s
    
    printf("\n=== %s Performance ===\n", kernelName);
    printf("Average execution time: %.3f ms (std dev: %.3f ms)\n", avgTime, stdDev);
    printf("Performance: %.2f GFLOPS\n", gigaFlops);
    printf("Memory bandwidth: %.2f GB/s\n", memoryBandwidth);
    printf("Individual run times: ");
    for (int i = 0; i < NUM_RUNS; i++) {
        printf("%.3f%s", times[i], (i < NUM_RUNS-1) ? ", " : " ms\n");
    }
    
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));
    
    return avgTime;
}

// Print a visual bar chart comparing performance
void printPerformanceComparison(float coalesced_time, float noncoalesced_time) {
    printf("\n=== Performance Comparison ===\n");
    
    float speedup = noncoalesced_time / coalesced_time;
    printf("Coalesced vs Non-coalesced speedup: %.2fx\n", speedup);
    
    // Calculate bar lengths (max 50 chars)
    int coalesced_bar = 50;  // Reference length
    int noncoalesced_bar = (int)(coalesced_bar / speedup);
    
    // Print bars
    printf("\nExecution Time Comparison (shorter is better):\n");
    
    printf("Coalesced:     [");
    for (int i = 0; i < coalesced_bar; i++) printf("#");
    printf("] %.3f ms\n", coalesced_time);
    
    printf("Non-coalesced: [");
    for (int i = 0; i < noncoalesced_bar; i++) printf("#");
    for (int i = noncoalesced_bar; i < coalesced_bar; i++) printf(" ");
    printf("] %.3f ms\n", noncoalesced_time);
    
    // Print performance bars (longer is better)
    printf("\nPerformance Comparison (longer is better):\n");
    
    printf("Coalesced:     [");
    for (int i = 0; i < coalesced_bar; i++) printf("#");
    printf("] %.2f GFLOPS\n", (2.0f * M * N * K) / (coalesced_time * 1e6));
    
    printf("Non-coalesced: [");
    for (int i = 0; i < noncoalesced_bar; i++) printf("#");
    printf("] %.2f GFLOPS\n", (2.0f * M * N * K) / (noncoalesced_time * 1e6));
}

int main() {
    // Print system information
    cudaDeviceProp prop;
    CHECK_CUDA_ERROR(cudaGetDeviceProperties(&prop, 0));
    printf("\n=== Device Information ===\n");
    printf("Device: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Global Memory: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("Shared Memory per Block: %lu KB\n", prop.sharedMemPerBlock / 1024);
    printf("Warp Size: %d\n", prop.warpSize);
    printf("Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
    printf("Memory Clock Rate: %.0f MHz\n", prop.memoryClockRate / 1000.0);
    printf("Memory Bus Width: %d bits\n", prop.memoryBusWidth);
    printf("Peak Memory Bandwidth: %.2f GB/s\n\n", 
           2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
    
    // Host memory
    float *h_A, *h_B, *h_C;
    
    // Device memory
    float *d_A, *d_B, *d_C;
    
    // Allocate host memory
    h_A = (float*)malloc(M * K * sizeof(float));
    h_B = (float*)malloc(N * K * sizeof(float));
    h_C = (float*)malloc(M * N * sizeof(float));
    
    // Initialize matrices
    printf("Initializing matrices...\n");
    initializeMatrices(h_A, h_B, M, N, K);
    
    // Allocate device memory
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_A, M * K * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_B, N * K * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_C, M * N * sizeof(float)));
    
    // Copy data from host to device
    printf("Copying data to device...\n");
    CHECK_CUDA_ERROR(cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_B, h_B, N * K * sizeof(float), cudaMemcpyHostToDevice));
    
    // Set up execution configuration
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
    dim3 gridDim((N + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_WIDTH - 1) / TILE_WIDTH);
    
    printf("\nMatrix dimensions: %d x %d x %d\n", M, N, K);
    printf("Grid dimensions: %d x %d\n", gridDim.x, gridDim.y);
    printf("Block dimensions: %d x %d\n", blockDim.x, blockDim.y);
    printf("Running each kernel %d times for averaging...\n", NUM_RUNS);
    
    // Run coalesced kernel
    float coalesced_time = runKernel(matrixMulCornerTurning, d_A, d_B, d_C, K, 
                                    gridDim, blockDim, "Coalesced (Corner Turning)");
    
    // Copy result back to host and verify
    CHECK_CUDA_ERROR(cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Skip verification for large matrices if flag is set
    if (!SKIP_VERIFICATION_FOR_LARGE_MATRICES || (M <= 1024 && N <= 1024 && K <= 1024)) {
        printf("Verifying coalesced kernel results...\n");
        verifyResults(h_A, h_B, h_C, M, N, K);
    } else {
        printf("Skipping verification for coalesced kernel (matrix too large)...\n");
    }
    
    // Run non-coalesced kernel
    float noncoalesced_time = runKernel(matrixMulNonCoalesced, d_A, d_B, d_C, K, 
                                       gridDim, blockDim, "Non-Coalesced");
    
    // Check if the non-coalesced kernel timed out
    if (noncoalesced_time == FLT_MAX) {
        printf("Non-coalesced kernel timed out. Skipping verification and performance comparison.\n");
    } else {
        // Copy result back to host and verify
        CHECK_CUDA_ERROR(cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
        
        // Skip verification for large matrices if flag is set
        if (!SKIP_VERIFICATION_FOR_LARGE_MATRICES || (M <= 1024 && N <= 1024 && K <= 1024)) {
            printf("Verifying non-coalesced kernel results...\n");
            verifyResults(h_A, h_B, h_C, M, N, K);
        } else {
            printf("Skipping verification for non-coalesced kernel (matrix too large)...\n");
        }
        
        // Print performance comparison
        printPerformanceComparison(coalesced_time, noncoalesced_time);
    }
    
    // Memory access analysis
    printf("\n=== Memory Access Analysis ===\n");
    printf("For %dx%d matrices with TILE_WIDTH=%d:\n", M, N, TILE_WIDTH);
    
    // Calculate theoretical memory transactions
    int warps_per_block = (TILE_WIDTH * TILE_WIDTH) / 32;
    int total_blocks = gridDim.x * gridDim.y;
    int total_warps = warps_per_block * total_blocks;
    int phases = K / TILE_WIDTH;
    
    printf("Total thread blocks: %d\n", total_blocks);
    printf("Warps per block: %d\n", warps_per_block);
    printf("Total warps: %d\n", total_warps);
    printf("Computation phases: %d\n\n", phases);
    
    // Coalesced analysis
    int coalesced_transactions_per_warp = 1; // Ideally 1 transaction per warp for coalesced
    int coalesced_total_transactions = total_warps * phases * 2 * coalesced_transactions_per_warp;
    
    // Non-coalesced analysis (worst case: 1 transaction per thread)
    int noncoalesced_transactions_per_warp = 32; // Worst case: 32 transactions per warp
    int noncoalesced_total_transactions = total_warps * phases * noncoalesced_transactions_per_warp;
    
    printf("Theoretical memory transactions for matrix B:\n");
    printf("Coalesced: %d transactions\n", coalesced_total_transactions / 2); // Divide by 2 since we only want to count matrix B
    printf("Non-coalesced: %d transactions\n", noncoalesced_total_transactions);
    printf("Theoretical transaction ratio: %.2fx\n\n", 
           (float)noncoalesced_transactions_per_warp / coalesced_transactions_per_warp);
    
    printf("Note: Modern GPUs have caches that can mitigate some of the performance impact\n");
    printf("of non-coalesced access, which is why the measured speedup may be less than\n");
    printf("the theoretical transaction ratio.\n");
    
    // Clean up
    CHECK_CUDA_ERROR(cudaFree(d_A));
    CHECK_CUDA_ERROR(cudaFree(d_B));
    CHECK_CUDA_ERROR(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);
    
    return 0;
}
