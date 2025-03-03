# 5. Memory Architecture and Data Locality

> 💡 **Core Concept**: This chapter explores GPU memory hierarchy and optimization techniques that maximize performance by efficiently using the various memory types and minimizing global memory traffic.

This chapter focuses on the on-chip memory architecture of the GPU and how to organize and position data for efficient access by a massive number of threads.

## 5.1 Importance of Memory Access Efficiency

> 🔑 **Key insight**: Many GPU applications are limited by memory bandwidth rather than computational power, making memory access optimization critical for performance.

### Memory vs. Compute Bound Applications

The **compute-to-memory access ratio** (also called **arithmetic intensity**) measures how many floating-point operations (FLOPs) are performed per byte accessed from global memory:

```
Compute-to-Memory Ratio = FLOPs performed / Bytes accessed from global memory
```

This ratio determines whether an application is:
- **Memory-bound**: Performance limited by memory bandwidth (low arithmetic intensity)
- **Compute-bound**: Performance limited by computational throughput (high arithmetic intensity)

Modern GPUs can perform thousands of floating-point operations in the time it takes to access global memory once, creating a significant performance gap:

```
┌─────────────────────────────────────────────────────────┐
│                  Performance Comparison                  │
│                                                          │
│ A100 GPU:                                                │
│ ┌───────────────────┐                                    │
│ │ Compute: 19.5 TFLOPS                                   │
│ └───────────────────┘                                    │
│                                                          │
│ ┌───────────────────┐                                    │
│ │ Memory: 1.55 TB/s                                      │
│ └───────────────────┘                                    │
│                                                          │
│ Ratio: ~12.5 FLOPs possible per byte accessed           │
└─────────────────────────────────────────────────────────┘
```

### Identifying Memory Bottlenecks

Memory bottlenecks can be identified by:
1. Profiling tools showing high memory utilization
2. Performance not scaling with increased compute resources
3. Calculating theoretical arithmetic intensity and comparing with hardware capabilities

### Strategies for Improving Memory Efficiency

1. **Data reuse**: Maximize operations per memory access
2. **Coalesced access**: Ensure threads in a warp access contiguous memory
3. **Memory hierarchy utilization**: Use faster memory types (shared, constant) when possible
4. **Minimizing data transfers**: Reduce host-device communication
5. **Tiling**: Partition data to fit in faster memory levels

> ⚠️ **Common pitfall**: Many developers focus on optimizing computational aspects when memory access patterns are actually the limiting factor.

## 5.2 CUDA Memory Types

CUDA provides several memory types, each with different scope, lifetime, and performance characteristics:

```
┌───────────────────── CUDA Memory Hierarchy ─────────────────────┐
│                                                                  │
│  Fastest   ┌───────────┐                                         │
│    ▲       │ Registers │ Thread-private, on-chip                 │
│    │       └───────────┘                                         │
│    │       ┌───────────┐                                         │
│    │       │   Shared  │ Block-accessible, on-chip               │
│    │       └───────────┘                                         │
│    │       ┌───────────┐                                         │
│    │       │ Constant  │ Read-only, cached, all threads          │
│    │       └───────────┘                                         │
│    │       ┌───────────┐                                         │
│    │       │   Local   │ Thread-private, in global memory        │
│    │       └───────────┘                                         │
│    │       ┌───────────┐                                         │
│  Slowest   │   Global  │ Accessible by all threads and host      │
│    ▼       └───────────┘                                         │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### Memory Type Characteristics

| Memory Type | Access Scope | Access Speed | Size      | Lifetime      | Declaration                        |
|-------------|--------------|--------------|-----------|---------------|------------------------------------|
| Register    | Thread       | Fastest      | Limited   | Thread        | Automatic variables (non-arrays)   |
| Shared      | Block        | Very fast    | ~100KB/SM | Block         | `__shared__ int s;`                |
| Constant    | Grid (read)  | Fast (cached)| ~64KB     | Application   | `__constant__ int c;`              |
| Local       | Thread       | Slow         | Large     | Thread        | Automatic arrays, large structures |
| Global      | Host & Grid  | Slowest      | GB range  | Application   | `__device__ int g;`                |

### Global Memory

Global memory:
- Accessible by both host and device
- Largest capacity (several GB)
- Highest latency (400-800 cycles)
- Persists for the entire application lifetime
- Primary means of communication between host and device

Usage pattern:
```c
// Host allocates and transfers data
float *d_data;
cudaMalloc((void**)&d_data, size);
cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);

// Kernel accesses global memory
__global__ void kernel(float *data) {
    // Read/write global memory
    float value = data[threadIdx.x];
    data[threadIdx.x] = value * 2.0f;
}
```

### Constant Memory

Constant memory:
- Read-only from device perspective
- Cached and optimized for broadcast (all threads reading same address)
- Limited size (~64KB total)
- Declared outside any function with `__constant__` qualifier

Example:
```c
// Declare in global scope
__constant__ float constData[256];

// Host code to initialize
cudaMemcpyToSymbol(constData, h_data, size);

// Kernel using constant memory
__global__ void kernel() {
    // All threads reading same index is efficient
    float value = constData[5];
    
    // Different threads reading different indices uses cache
    float myValue = constData[threadIdx.x];
}
```

### Shared Memory

Shared memory:
- On-chip memory shared by all threads in a block
- Much faster than global memory (100x lower latency)
- Limited size (up to ~100KB per SM in modern GPUs)
- Declared with `__shared__` qualifier
- Ideal for data reuse within a block

Example:
```c
__global__ void kernel() {
    // Static allocation
    __shared__ float sharedData[256];
    
    // Or dynamic allocation (size set at kernel launch)
    extern __shared__ float dynamicShared[];
    
    // Each thread loads one element from global to shared
    sharedData[threadIdx.x] = globalData[threadIdx.x];
    
    // Synchronize to ensure all data is loaded
    __syncthreads();
    
    // Now threads can access each other's data efficiently
    float value = sharedData[255 - threadIdx.x];
}
```

> ⚠️ **Important**: Always use `__syncthreads()` after writing to shared memory before other threads read those values.

### Local Memory

Local memory:
- Thread-private storage but physically located in global memory
- Used for automatic arrays and register spilling
- Has same high latency as global memory
- Compiler-managed; not directly controlled by programmer

When local memory is used:
```c
__global__ void kernel() {
    // Large array likely placed in local memory
    float largeArray[1000];
    
    // Complex function with many variables might spill to local memory
    for (int i = 0; i < 100; i++) {
        float temp1, temp2, temp3, /*...many more local variables*/;
        // Complex calculations causing register pressure
    }
}
```

### Register Memory

Register memory:
- Fastest memory type
- Thread-private variables
- Limited quantity (e.g., 65,536 32-bit registers per SM)
- Allocated by compiler automatically for scalar variables

> 🔍 **Tip**: Use the `--ptxas-options=-v` compiler flag to see register usage for your kernels.

## 5.3 Tiling for Reduced Memory Traffic

> 💡 **Core Concept**: Tiling partitions data into chunks that fit in shared memory, allowing multiple threads to reuse the same data and dramatically reducing global memory traffic.

### The Tiling Principle

Tiling leverages the memory hierarchy by:
1. Loading a subset of data from global memory into shared memory
2. Having multiple threads reuse this data for calculations
3. Moving to the next subset until all data is processed

```
┌──────────────────── Global Memory ─────────────────────┐
│                                                         │
│  ┌─────┬─────┬─────┬─────┐                             │
│  │Tile1│Tile2│Tile3│Tile4│ ... and so on               │
│  └─────┴─────┴─────┴─────┘                             │
│     │                                                   │
│     │ Load one tile at a time                           │
│     ▼                                                   │
│  ┌─────┐                                                │
│  │Tile │ ◄── Multiple threads process using shared mem  │
│  └─────┘                                                │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Benefits of Tiling

1. **Reduced global memory traffic**: Data is loaded once into shared memory and reused many times
2. **Improved memory bandwidth utilization**: Coalesced accesses when loading tiles
3. **Higher arithmetic intensity**: More operations per global memory access
4. **Better cache utilization**: Working with subsets that fit in cache

### Memory Traffic Reduction Analysis

Consider matrix multiplication where each output element requires N multiplications:

| Approach | Global Memory Accesses | Compute Ops | Ratio (Ops/Access) |
|----------|------------------------|-------------|-------------------|
| Naive    | 2N + 1 per element    | N           | ~0.5              |
| Tiled (T×T) | 2N/T + 1 per element | N           | ~T/2              |

For a tile size of 16×16, memory traffic is reduced by a factor of 16, increasing arithmetic intensity by 16×.

### Tiling Requirements

For tiling to be effective:
1. Computation must have **data reuse** opportunities
2. The reused data must fit in **shared memory**
3. Tiles must be **independent** or have minimal dependencies
4. Thread block dimensions should match tile dimensions for efficient access

> 🔑 **Key insight**: The ideal tile size balances memory usage, thread count, and memory access patterns. Too small tiles underutilize shared memory; too large tiles reduce occupancy.

### Common Tiling Applications

Tiling works well for:
- Matrix operations (multiplication, transposition)
- Convolutions and stencil operations
- Image processing filters
- Reduction operations with intermediate results

> 🔍 **Practical tip**: Start with 16×16 or 32×8 tiles and adjust based on profiling results. The tile size should be a multiple of 32 (warp size) for optimal performance.

## 5.4 A Tiled Matrix Multiplication Kernel

> 💡 **Core Concept**: Tiling uses shared memory to reduce global memory traffic by loading small blocks of input matrices that multiple threads can reuse.

### How Tiling Improves Performance

The basic matrix multiplication kernel from Chapter 3 required each thread to:
- Read N elements from matrix M (one row)
- Read N elements from matrix N (one column)
- Perform N multiply-add operations
- Write 1 element to matrix P

This resulted in (2N+1) memory operations for N computations, giving a compute-to-memory ratio of 1:2, which is memory-bound.

With tiling, a block of threads collaboratively:
1. Loads TILE_WIDTH×TILE_WIDTH elements from M and N into shared memory
2. Each thread uses these cached elements for partial dot product calculations
3. Moves to the next tile until the full dot product is complete

This reduces global memory accesses by a factor of TILE_WIDTH, potentially increasing performance by the same factor.

### Key Implementation Aspects

- **Shared Memory Arrays**: `Mds` and `Nds` hold tiles of matrices M and N
- **Thread Indexing**: Each thread calculates its P element's location using block and thread indices
- **Phased Computation**: Outer loop processes one tile at a time (strip-mining)
- **Barrier Synchronization**: Required before and after using shared memory to ensure data is loaded/used properly
- **Memory Access Patterns**: Structured to achieve coalesced access for better performance

Tiling changes the compute-to-memory ratio from 0.25 OP/B to 4 OP/B with a 16×16 tile, potentially increasing performance by 16×.

> ⚠️ **Limitations**: This implementation assumes matrices are square and have dimensions that are multiples of the tile width.

### Tiled Matrix Multiplication Code

```c
__global__ void MatrixMulKernel(float* M, float* N, float* P, int Width) {
    // Shared memory for the tiles
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];
    
    // Thread and block indices
    // These are placed into registers and are terminated after use
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    // Row and column indices for the output element
    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;
    
    // Accumulator for the dot product
    float Pvalue = 0;
    
    // Loop over tiles
    for (int ph = 0; ph < Width/TILE_WIDTH; ++ph) {
        // Load M tile into shared memory
        Mds[ty][tx] = M[Row*Width + ph*TILE_WIDTH + tx];
        // Load N tile into shared memory
        Nds[ty][tx] = N[(ph*TILE_WIDTH+ty)*Width + Col];
        
        // Ensure all threads have loaded their elements
        __syncthreads();
        
        // Compute partial dot product using the tile
        for (int k = 0; k < TILE_WIDTH; ++k) {
            Pvalue += Mds[ty][k] * Nds[k][tx];
        }
        
        // Ensure all threads have finished using the tile
        __syncthreads();
    }
    
    // Write result to global memory
    P[Row*Width + Col] = Pvalue;
}
```

### Launching the Kernel

```c
// Calculate grid and block dimensions
dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
dim3 dimGrid(Width/TILE_WIDTH, Width/TILE_WIDTH);

// Launch kernel
MatrixMulKernel<<<dimGrid, dimBlock>>>(d_M, d_N, d_P, Width);
```

> 💡 **Performance Note**: With a 16×16 tile on an A100 GPU, this optimization allows achieving up to 6220 GFLOPS versus 389 GFLOPS for the non-tiled version, though this is still only 32% of the A100's peak 19,500 GFLOPS.

## 5.5 Boundary Checks

> ⚠️ **Problem**: When matrix dimensions aren't multiples of tile width, threads may try to access non-existent elements, causing incorrect results or crashes.

### Handling Matrix Boundaries

When working with matrices whose dimensions aren't multiples of `TILE_WIDTH`, three issues arise:

1. Threads might access elements past the end of a row (accessing incorrect data)
2. Threads might access elements past the end of a column (accessing memory outside the allocated array)
3. These boundary issues occur in all phases of execution, not just the last phase

### Boundary Check Solution

The solution requires three separate checks:

1. **For loading M tile elements**: `Row < Width && (ph*TILE_WIDTH+tx) < Width`
2. **For loading N tile elements**: `(ph*TILE_WIDTH+ty) < Width && Col < Width`
3. **For storing P results**: `Row < Width && Col < Width`

When a thread would load an invalid element, it should put 0.0 in shared memory instead, which won't affect dot product calculations.

### Tiled Matrix Multiplication With Boundary Checks

```c
__global__ void MatrixMulKernel(float* M, float* N, float* P, int Width) {
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;
    
    float Pvalue = 0;
    
    for (int ph = 0; ph < (Width+TILE_WIDTH-1)/TILE_WIDTH; ++ph) {
        // Load M tile with boundary check
        if (Row < Width && ph*TILE_WIDTH+tx < Width)
            Mds[ty][tx] = M[Row*Width + ph*TILE_WIDTH + tx];
        else
            Mds[ty][tx] = 0.0;
            
        // Load N tile with boundary check
        if (ph*TILE_WIDTH+ty < Width && Col < Width)
            Nds[ty][tx] = N[(ph*TILE_WIDTH+ty)*Width + Col];
        else
            Nds[ty][tx] = 0.0;
        
        __syncthreads();
        
        // Calculate partial dot product
        for (int k = 0; k < TILE_WIDTH; ++k) {
            Pvalue += Mds[ty][k] * Nds[k][tx];
        }
        
        __syncthreads();
    }
    
    // Store result with boundary check
    if (Row < Width && Col < Width)
        P[Row*Width + Col] = Pvalue;
}
```

> 💡 **Key insight**: Every memory access needs its own boundary check to ensure indices are within array bounds. This makes the kernel work with arbitrary matrix dimensions, not just multiples of the tile width.

This implementation is almost a general matrix multiplication kernel, with one limitation remaining: it only works for square matrices (future optimizations would handle rectangular matrices with different dimensions).

## 5.6 Impact of Memory Usage on Occupancy

> 💡 **Core Concept**: Memory resources (registers and shared memory) directly affect occupancy, which influences performance through latency hiding. Finding the optimal balance is critical for kernel optimization.

### Understanding Occupancy and Resource Limits

**Occupancy** is the ratio of active warps to the maximum possible warps on a Streaming Multiprocessor (SM). It's constrained by four main factors:

1. **Register usage**: Registers are allocated per thread
2. **Shared memory usage**: Shared memory is allocated per block
3. **Block size**: Number of threads per block
4. **Hardware limits**: Maximum warps/threads/blocks per SM

```
┌─────────────────── Resource-Occupancy Relationship ────────────────────┐
│                                                                       │
│                             ┌─────────────┐                             │
│         ┌─────────────────►│   Maximum   │◄─────────────────┐          │
│         │                   │ SM Capacity │                  │          │
│         │                   └─────────────┘                  │          │
│         │                          ▲                         │          │
│         │                          │                         │          │
│         │                          │                         │          │
│  ┌──────────────┐          ┌──────────────┐          ┌──────────────┐  │
│  │   Register   │          │    Shared    │          │    Block     │  │
│  │    Usage     │          │    Memory    │          │    Size      │  │
│  └──────────────┘          └──────────────┘          └──────────────┘  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Register Usage Impact

Every thread requires registers for its variables. Register usage affects occupancy as follows:

- More registers per thread → fewer threads can be active simultaneously
- Example: On an SM with 65,536 registers and 2,048 maximum threads:
  - Using 32 registers/thread → 100% occupancy (65,536 ÷ 32 = 2,048 threads)
  - Using 64 registers/thread → 50% occupancy (65,536 ÷ 64 = 1,024 threads)
  - Using 128 registers/thread → 25% occupancy (65,536 ÷ 128 = 512 threads)

> ⚠️ **Important**: The compiler decides register allocation, not the programmer. Use `--ptxas-options=-v` to see how many registers your kernel uses.

### Shared Memory Impact

Shared memory is allocated per block, impacting how many blocks can reside on an SM:

- More shared memory per block → fewer blocks can be active simultaneously
- Example: On an SM with 64KB shared memory:
  - Using 16KB/block → 4 blocks can reside (64KB ÷ 16KB = 4 blocks)
  - Using 32KB/block → 2 blocks can reside (64KB ÷ 32KB = 2 blocks)

If this block count becomes the limiting factor, it directly affects occupancy.

### Occupancy Calculator Example

Suppose we have a kernel with these characteristics:
- 64 registers per thread
- 8KB shared memory per block
- 256 threads per block
- Hardware: SM with 65,536 registers, 64KB shared memory, 2,048 max threads

The limiting factors would be:
- Register limit: 65,536 ÷ 64 = 1,024 threads (8 blocks of 128 threads)
- Shared memory limit: 64KB ÷ 8KB = 8 blocks (8 × 256 = 2,048 threads)
- Thread limit: 2,048 threads (8 blocks of 256 threads)

The most restrictive is the register limit, allowing only 1,024 threads (50% occupancy).

### Occupancy vs. Performance Curve

The relationship between occupancy and performance is not linear:

```
┌───────────────── Typical Occupancy-Performance Curve ────────────────┐
│                                                                       │
│ Performance                                                           │
│     ▲                                                                 │
│     │                                      ┌─────────────────────     │
│     │                                ┌─────┘                          │
│     │                         ┌──────┘                                │
│     │                    ┌────┘                                       │
│     │               ┌────┘                                            │
│     │          ┌────┘                                                 │
│     │      ┌───┘                                                      │
│     │  ┌───┘                                                          │
│     │┌─┘                                                              │
│     └─────────┬─────────┬─────────┬─────────┬─────────► Occupancy    │
│              20%        40%        60%        80%       100%         │
│                                                                       │
│                Performance often plateaus before 100% occupancy       │
└───────────────────────────────────────────────────────────────────────┘
```

> 🔑 **Key insight**: Many kernels achieve peak performance at 40-70% occupancy. Beyond this point, increasing occupancy may not improve performance and could even be counterproductive if it requires sacrificing other optimizations.

### Tools for Analyzing Occupancy

1. **NVIDIA Nsight Compute**: Provides detailed occupancy analysis
2. **CUDA Occupancy Calculator**: Spreadsheet tool to experiment with different configurations
3. **`cudaOccupancyMaxPotentialBlockSize`**: Runtime API to find optimal block size
   ```c
   int minGridSize;
   int blockSize;
   cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, 
                                     MyKernel, 0, 0);
   ```

### Strategies for Optimizing Memory Usage

1. **Register optimization**:
   - Use compiler flags like `--maxrregcount=N` to limit register usage
   - Break complex kernels into smaller ones
   - Recompute values instead of storing in registers when beneficial

2. **Shared memory optimization**:
   - Use appropriate tile sizes that balance occupancy with memory efficiency
   - Consider using multiple smaller tiles instead of one large tile
   - Dynamically allocate only what's needed with `extern __shared__`

3. **Thread block size selection**:
   - Choose sizes that are multiples of 32 (warp size)
   - Consider using rectangular blocks (e.g., 32×8) rather than square ones (16×16)
   - Experiment with different configurations, as optimal values vary by kernel

> 🔍 **Practical tip**: Profile before optimizing. Measure the actual impact of your changes, as intuition about performance can be misleading. Sometimes using more registers to avoid recomputation is better despite reducing occupancy.

### Balancing Memory Resources

Finding the optimal configuration is often a balancing act:

1. **Register vs. recomputation**: Using more registers reduces occupancy but may be faster than recomputing values
2. **Shared memory vs. global memory**: Using shared memory reduces occupancy but significantly speeds up memory access
3. **Block size vs. resources**: Larger blocks improve shared memory utilization but consume more resources per block

> 💡 **Understanding**: There's no universal "best" configuration - the optimal balance depends on the specific algorithm, device capabilities, and memory access patterns.

### Optimization Workflow

1. **Measure**: Profile kernel with NSight Compute to identify limiting resources
2. **Analyze**: Determine if occupancy is actually limiting performance
3. **Experiment**: Try different configurations of block size, shared memory, and compiler flags
4. **Benchmark**: Measure performance under each configuration
5. **Iterate**: Refine based on results to find the optimal configuration

> 🔑 **Key insight**: Don't optimize occupancy in isolation. Consider the overall goal of maximizing throughput, which might involve trading lower occupancy for better memory access patterns or instruction-level optimizations.

## 5.7 Exercises

1. Consider matrix addition. Can one use shared memory to reduce the global memory bandwidth consumption? Hint: Analyze the elements that are accessed by each thread and see whether there is any commonality between threads.
   
   You could, but it wouldn't have any performance difference because matrix addition is just an element-wise operation that contains only one operation per element. The whole point of using shared memory for matrix multiplication is because there are elements in each matrix that are "reused" across different operations. In matrix addition, each element is only used once.

2. Draw the equivalent of Fig. 5.7 for an 8x8 matrix multiplication with 2x2 tiling and 4x4 tiling. Verify that the reduction in global memory bandwidth is indeed proportional to the dimension size of the tiles.
   
   For 8x8 matrix at a tile size of 2 we have (width / tile)^2 * 2 for (P = A @ B) which gives us (8 / 2)^2 * 2 = 32 tiles to load
   For 8x8 matrix at a tile size of 4 we have (8 / 4)^2 * 2 = 8 tiles to load
   So yes, the reduction in global memory bandwidth is proportional to the dimension of the tile size.

3. What type of incorrect execution behavior can happen if one forgot to use one or both `__syncthreads()` in the kernel of fig. 5.9?
   
   It would create a race condition on either the loading of the tiles or the dot product itself.
   If it's in the tile loading phase, then some tile in P would be an incorrect computation between tiles in A or B.
   If it's in the dot product operation itself, then there could be a situation where a thread has already executed its operation and then goes back and redoes its operation because it doesn't know it needs to wait for the remaining threads.

4. Assuming that capacity is not an issue for registers or shared memory, give one important reason why it would be valuable to use shared memory instead of registers to hold values fetched from global memory?
   
   One important reason is that all threads in a block have access to the data in shared memory, while registers only have access to the data for that specific thread. This would be very important for a sorting algorithm where threads need to iteratively access data across the entire block, rather than just their data in the register.

5. For our tiled matrix multiplication kernel, if we use a 32x32 tile, what is the reduction of memory bandwidth usage for input matrices M and N?
   
   By a factor of 32.

6. Assume that a CUDA kernel is launched with 1000 thread blocks, each of which has 512 threads. If a variable is declared as a local variable in the kernel, how many versions of the variable will be created through the lifetime of the execution of the kernel?
   
   Each thread will have its own copy, so there will be 512,000 versions of the variable throughout the kernel.

7. In the previous question, if a variable is declared as a shared memory variable, how many versions of the variable will be created through the lifetime of the execution of the kernel?
   
   1000, because shared memory is block-wise so there will be one variable for each block.

8. Consider performing a matrix multiplication of two input matrices with dimensions NxN. How many times is each element in the input matrices requested from global memory when:
   a. There is no tiling?
   b. Tiles of size TxT are used?
   
   a. 2N
   b. 2N/T

9. A kernel performs 36 floating-point operations and seven 32-bit global memory accesses per thread. For each of the following device properties, indicate whether this kernel is compute-bound or memory-bound.
   a. Peak FLOPS = 200 GFLOPS, peak memory bandwidth = 100 GB/second
   b. Peak FLOPS = 300 GFLOPS, peak memory bandwidth = 250 GB/second
   
   Calculate the kernel's arithmetic intensity:
   - Arithmetic intensity = 36 FLOPs / (7 × 4 bytes) = 36 / 28 = 1.29 FLOPs/byte
   
   Device A:
   - Device ratio = 200 × 10^9 / (100 × 10^9) = 2 FLOPs/byte
   - Since 1.29 < 2, this is memory-bound
   
   Device B:
   - Device ratio = 300 × 10^9 / (250 × 10^9) = 1.2 FLOPs/byte
   - Since 1.29 > 1.2, this is compute-bound

10. To manipulate tiles, a new CUDA programmer has written a device kernel that will transpose each tile in a matrix. The tiles are of size BLOCK_WIDTH by BLOCK_WIDTH, and each of the dimensions of the matrix A is known to be a multiple of BLOCK_WIDTH. The kernel invocation and code are shown below. BLOCK_WIDTH is known at compile time and could be set anywhere from 1 to 20.
    ```c
    dim3 blockDim(BLOCK_WIDTH, BLOCK_WIDTH);
    dim3 gridDim(A_width/blockDim.x, A_height/blockDim.y);
    BlockTranspose<<<gridDim, blockDim>>>(A, A_width, A_height);

    __global__ void BlockTranspose(float* A_elements, int A_width, int A_height) {
        __shared__ float blockA[BLOCK_WIDTH][BLOCK_WIDTH];

        int baseIdx = blockIdx.x * BLOCK_WIDTH + threadIdx.x;
        baseIdx += (blockIdx.y * BLOCK_WIDTH + threadIdx.y) * A_width;

        blockA[threadIdx.y][threadIdx.x] = A_elements[baseIdx];

        A_elements[baseIdx] = blockA[threadIdx.x][threadIdx.y];
    }
    ```

    a. Out of the possible range of values for BLOCK_WIDTH, for what values of BLOCK_WIDTH will this kernel function execute correctly on the device?
       None of them will run correctly because of a race condition.
    
    b. If the code does not execute correctly for all BLOCK_WIDTH values, what is the root cause of this incorrect execution behavior? Suggest a fix to the code to make it work for all BLOCK_WIDTH values.
       You'll need to introduce a synchronization barrier (`__syncthreads()`) between reading from global memory and writing back to it. Currently, some threads might read into shared memory while others have already written their transposed values back to global memory, causing incorrect results.

11. Consider the following CUDA kernel and the corresponding host function that calls it:
    ```c
    __global__ void foo_kernel(float* a, float* b) {
        unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
        float x[4];
        __shared__ float y_s;
        __shared__ float b_s[128];
        for(unsigned int j = 0; j < 4; ++j) {
            x[j] = a[j*blockDim.x*gridDim.x + i];
        }
        if(threadIdx.x == 0) {
            y_s = 7.4f;
        }
        b_s[threadIdx.x] = b[i];
        __syncthreads();
        b[i] = 2.5f*x[0] + 3.7f*x[1] + 6.3f*x[2] + 8.5f*x[3]
                + y_s*b_s[threadIdx.x] + b_s[(threadIdx.x + 3)%128];
    }
    void foo(int* a_d, int* b_d) {
        unsigned int N = 1024;
        foo_kernel <<< (N + 128 - 1)/128, 128  >>>(a_d, b_d)
    }
    ```

    a. How many versions of the variable i are there?
       1024, one for each thread
    
    b. How many versions of the array x[] are there?
       1024, one for each thread
    
    c. How many versions of the variable y_s are there?
       8, one for each block
    
    d. How many versions of the array b_s[] are there?
       8, one for each block
    
    e. What is the amount of shared memory used per block (in bytes)?
       4 bytes per float
       1 float for y_s per block
       128 floats for b_s per block
       4 * (1 + 128) = 516 bytes per block
    
    f. What is the floating-point to global memory access ratio of the kernel in (OP/B)?
       10 FLOPs / 24B = 0.42 OP/B

12. Consider a GPU with the following hardware limits: 2048 threads/SM, 32 blocks/SM, 64k registers/SM, and 96 KB of shared memory/SM. For each of the following kernel characteristics, specify whether the kernel can achieve full occupancy. If not, specify the limiting factor.
    
    a. The kernel uses 64 threads/block, 27 registers/thread, and 4KB of shared memory/SM.
       Yes, the kernel can achieve full occupancy.
       - Register usage: 64 threads/block × 27 registers/thread × 32 blocks = 55,296 registers < 64K
       - Shared memory: 4KB × 32 blocks = 128KB > 96KB, so max 24 blocks can fit
       - Thread count: 64 threads/block × 32 blocks = 2,048 threads
       The kernel can fit the maximum number of threads with these resource constraints.
    
    b. The kernel uses 256 threads/block, 32 registers/thread, and 8KB of shared memory/SM.
       No, the kernel cannot achieve full occupancy due to register limitations.
       - Register usage: 256 threads/block × 32 registers/thread × 8 blocks = 65,536 registers, which is exactly the limit
       - Shared memory: 8KB × 8 blocks = 64KB < 96KB, so this is not limiting
       - Thread count: 256 threads/block × 8 blocks = 2,048 threads
       The kernel would be limited by register usage, but it's right at the limit. 