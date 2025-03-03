# Programming Massively Parallel Processors: A Hands-on Approach

## 2. Heterogeneous Data Parallel Computing

> 💡 **Core Concept**: This chapter introduces how to program GPUs using CUDA C by managing memory across devices and creating parallel kernels that execute the same code across thousands of threads.

### 2.1 Parallelism Paradigms

#### Task parallelism vs. Data parallelism

- **Task parallelism**: Different independent operations executed simultaneously
  - Example: Web server handling multiple client requests concurrently
  - Often uses fewer, more complex threads
  - Better for diverse workloads with different execution paths

- **Data parallelism**: Same operation applied to multiple data elements simultaneously
  - Example: Adding two vectors element by element
  - Uses many simple threads doing identical work on different data
  - Ideal for GPUs with their thousands of cores
  - CUDA primarily focuses on this model

### 2.2 CUDA C Program Structure

A typical CUDA application consists of these phases:
1. **Initialize data** on the host (CPU)
2. **Transfer data** from host to device (GPU)
3. **Execute kernel** on the device
4. **Transfer results** back from device to host

#### Threads

A thread represents a sequential execution path through a program. In CUDA:
- Each thread has its own program counter and registers
- Threads execute the same code but operate on different data
- The execution of a thread is sequential as far as a user is concerned
- CUDA applications can launch thousands or millions of threads simultaneously

```
Host Code                    Device Code (Kernel)
┌───────────────┐            ┌───────────────┐
│ Allocate mem  │            │ Thread 0      │
│ Copy to device├───────────►│ Thread 1      │
│ Launch kernel │            │ Thread 2      │
│ Copy to host  │◄───────────┤ ...           │
│ Free memory   │            │ Thread n-1    │
└───────────────┘            └───────────────┘
```

### 2.3 A Vector Addition Example

#### Vector addition in C (CPU)
```c
// compute vector sum C_h = A_h + B_h
void vecAdd(float* A_h, float* B_h, float* C_h, int N) {
    for (int i = 0; i < N; i++) {
        C_h[i] = A_h[i] + B_h[i];
    }
}
int main() {
    // Memory allocation for arrays A, B, and C
    // I/O to read A and B, N elements each
    ...
    vecAdd(A, B, C, N);
}
```

> 📝 **Note**: In CPU code, a single thread processes all elements sequentially in a for-loop.

#### Pointer Review (C Language)

- **Variables vs. Pointers**:
  - `float V;` - Declares a variable that stores a value
  - `float *P;` - Declares a pointer that stores a memory address
  
- **Key Operations**:
  - `P = &V;` - P now points to V's memory address
  - `*P` - Accesses the value at P's address (value of V)
  - `*P = 3;` - Changes the value at P's address (changes V to 3)

- **Arrays and Pointers**:
  - Array names are actually pointers to the first element
  - `A[i]` is equivalent to `*(A+i)`
  - `P = &(A[0])` makes P point to the first element of A

#### Vector addition in CUDA C (GPU approach)

CUDA vector addition involves three main steps:

```c
void vecAdd(float* A_h, float* B_h, float* C_h, int N) {
    int size = N * sizeof(float);
    float *A_d, *B_d, *C_d;
    
    // Part 1: Allocate device memory for A, B, and C
    // Copy A and B to device memory
    cudaMalloc((void**)&A_d, size);
    cudaMalloc((void**)&B_d, size);
    cudaMalloc((void**)&C_d, size);
    
    cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);

    // Part 2: Launch the kernel
    // Grid of threads performs the actual vector addition
    vecAddKernel<<<ceil(N/256.0), 256>>>(A_d, B_d, C_d, N);

    // Part 3: Copy C from device memory to host memory
    // Free device memory 
    cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);
    
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}
```

> 🔑 **Key insight**: The host function orchestrates the entire process but the actual computation happens on the GPU, where thousands of threads execute the same kernel function on different pieces of data.

### 2.4 Device Global Memory and Data Transfer

#### Memory Allocation with cudaMalloc()

```c
cudaError_t cudaMalloc(void** devPtr, size_t size);
```

- **Purpose**: Allocates object in the device global memory
- **Parameters**:
  - `devPtr`: Address of a pointer to store the allocated memory address
  - `size`: Size of allocated object in bytes
- **Returns**: cudaSuccess or an error code

> ⚠️ **Important**: `cudaMalloc` takes a pointer-to-pointer (void**) because it needs to modify the pointer value to point to the newly allocated memory. This is one of the most common sources of errors for beginners.

Example with error checking:
```c
float *A_d;
cudaError_t err = cudaMalloc((void**)&A_d, size);
if (err != cudaSuccess) {
    printf("Error allocating device memory: %s\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
}
```

#### Memory Deallocation with cudaFree()

```c
cudaError_t cudaFree(void* devPtr);
```

- **Purpose**: Frees object from device global memory
- **Parameters**:
  - `devPtr`: Pointer to previously allocated memory
- **Returns**: cudaSuccess or an error code

> 🚫 **Common mistake**: Failing to free device memory can lead to memory leaks, especially in applications that repeatedly allocate memory.

#### Data Transfer with cudaMemcpy()

```c
cudaError_t cudaMemcpy(void* dst, const void* src, 
                      size_t count, enum cudaMemcpyKind kind);
```

- **Purpose**: Transfers data between host and device memory
- **Parameters**:
  - `dst`: Destination pointer
  - `src`: Source pointer
  - `count`: Number of bytes to transfer
  - `kind`: Type/direction of transfer
- **Transfer Types**:
  - `cudaMemcpyHostToDevice`: CPU → GPU
  - `cudaMemcpyDeviceToHost`: GPU → CPU
  - `cudaMemcpyDeviceToDevice`: GPU → GPU
  - `cudaMemcpyHostToHost`: CPU → CPU (rarely used)

> 📝 **Mental model**: Think of the GPU as having its own separate memory space. Data must be explicitly moved between CPU and GPU memory spaces using cudaMemcpy.

#### Complete Vector Addition with Memory Management

```c
void vecAdd(float* A_h, float* B_h, float* C_h, int N) {
    int size = N * sizeof(float);
    float *A_d, *B_d, *C_d;

    // Allocate device memory
    cudaMalloc((void**)&A_d, size);
    cudaMalloc((void**)&B_d, size);
    cudaMalloc((void**)&C_d, size);

    // Copy from host to device
    cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);

    // Launch kernel
    vecAddKernel<<<ceil(N/256.0), 256>>>(A_d, B_d, C_d, N);

    // Copy result back to host
    cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}
```

#### Error Checking Pattern

Always check for errors in CUDA calls with this pattern:

```c
cudaError_t err = cudaFunction();
if (err != cudaSuccess) {
    printf("CUDA Error: %s\n", cudaGetErrorString(err));
    // Handle error (cleanup, exit, etc.)
}
```

> 🔍 **Pro tip**: Create a macro for error checking to avoid repetitive code:
> ```c
> #define CHECK_CUDA_ERROR(call) { \
>     cudaError_t err = call; \
>     if (err != cudaSuccess) { \
>         printf("CUDA Error: %s at %s:%d\n", \
>                cudaGetErrorString(err), __FILE__, __LINE__); \
>         exit(EXIT_FAILURE); \
>     } \
> }
> ```
> Then use: `CHECK_CUDA_ERROR(cudaMalloc((void**)&A_d, size));`

### 2.5 Kernel Functions and Threading

Kernel functions are the core of CUDA programming - they define what each thread executes.

#### CUDA Function Type Qualifiers

| Qualifier   | Callable from | Executes on | Executed by |
|-------------|---------------|------------|------------|
| `__host__`   | Host          | Host       | CPU        |
| `__global__` | Host          | Device     | GPU        |
| `__device__` | Device        | Device     | GPU        |

- **`__global__`**: Kernel functions that launch a grid of threads
- **`__device__`**: Helper functions called by kernels, executed by a single thread
- **`__host__`**: Regular CPU functions (default if no qualifier is specified)
- **`__host__ __device__`**: Functions that can be called and executed on both CPU and GPU

#### Thread Identification with Built-in Variables

Every CUDA thread has access to these built-in variables:

- **`threadIdx`**: Thread index within a block (unique within block)
- **`blockIdx`**: Block index within the grid (unique within grid)
- **`blockDim`**: Number of threads per block
- **`gridDim`**: Number of blocks in the grid

Each variable has `.x`, `.y`, and `.z` components for 3D organization.

#### Thread Indexing in 1D

To get a unique global index for each thread in a 1D grid:

```c
int i = blockIdx.x * blockDim.x + threadIdx.x;
```

Visualization of this formula:
```
Thread indices within blocks:   [0,1,2...31][0,1,2...31][0,1,2...31]
Block indices:                   Block 0     Block 1     Block 2
Global thread indices:         [0-31]      [32-63]     [64-95]...
```

#### Vector Addition Kernel Implementation

```c
__global__ void vecAddKernel(float* A, float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {  // Check boundary
        C[i] = A[i] + B[i];
    }
}
```

> ⚠️ **Critical**: The boundary check `if (i < N)` prevents out-of-bounds memory access when the number of threads exceeds the array size, which happens frequently because we round up the number of blocks.

#### From Loop to Parallelism

| CPU (Sequential)            | GPU (Parallel)                  |
|-----------------------------|---------------------------------|
| `for (int i = 0; i < N; i++)`  | Many threads, each with unique `i` |
| Explicit iteration          | Implicit iteration via threads   |
| Single thread does all work | Each thread does small amount   |

### 2.6 Calling Kernel Functions

#### Kernel Launch Syntax

```c
kernel<<<gridDim, blockDim, sharedMem, stream>>>(args...);
```

- **`gridDim`**: Number of blocks in the grid (can be int or dim3)
- **`blockDim`**: Number of threads per block (can be int or dim3)
- **`sharedMem`**: (Optional) Dynamic shared memory size in bytes
- **`stream`**: (Optional) CUDA stream for asynchronous execution

> 📝 **Note**: The `<<<...>>>` syntax is unique to CUDA C/C++ and is processed by the NVCC compiler.

#### Calculating Grid Dimensions

For a 1D vector of length N and block size of 256:

```c
int blockSize = 256;
int numBlocks = (N + blockSize - 1) / blockSize;  // Ceiling division
kernel<<<numBlocks, blockSize>>>(args...);
```

This is equivalent to:
```c
kernel<<<ceil(N/256.0), 256>>>(args...);
```

> 💭 **Intuition**: If N is 1000 and blockSize is 256, we need 4 blocks because:
> (1000 + 256 - 1) / 256 = 1255 / 256 = 4.9 → 4 blocks
> This gives us 4 × 256 = 1024 threads, which is more than the 1000 we need.
> The boundary check ensures only the first 1000 threads perform computation.

#### Complete Vector Addition Example

```c
void vecAdd(float* A_h, float* B_h, float* C_h, int N) {
    // Allocate device memory
    float *A_d, *B_d, *C_d;
    int size = N * sizeof(float);
    cudaMalloc((void**)&A_d, size);
    cudaMalloc((void**)&B_d, size);
    cudaMalloc((void**)&C_d, size);

    // Copy input vectors from host to device
    cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);

    // Launch kernel with proper grid dimensions
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    vecAddKernel<<<numBlocks, blockSize>>>(A_d, B_d, C_d, N);

    // Copy result back to host
    cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}
```

### 2.7 Common Pitfalls and Best Practices

- **Memory Management**:
  - Always free GPU memory with `cudaFree()` to prevent leaks
  - Check return values of CUDA API calls
  - Minimize data transfers between host and device

- **Kernel Execution**:
  - Always include boundary checks in kernels
  - Choose block sizes that are multiples of 32 (warp size)
  - Typical block sizes: 128, 256, or 512 threads

- **Debugging**:
  - Use `cudaGetLastError()` after kernel launches to check for errors
  - For complex kernels, print intermediate values with `printf()` (available in CUDA)
  - Consider using NVIDIA's NSight or CUDA-GDB for debugging

### 2.8 Key Takeaways

- CUDA follows the **SPMD** (Single Program, Multiple Data) programming model
- A typical CUDA program: allocate GPU memory → copy data to GPU → execute kernel → copy results back → free memory
- Data must be explicitly moved between CPU and GPU memory spaces
- CUDA kernels replace loops with parallelism across thousands of threads
- Each thread needs a way to identify which data element(s) to process using built-in variables

***

**Exercise Idea**: Try implementing a vector SAXPY operation (`Y = a*X + Y`) in CUDA, where `a` is a scalar and `X` and `Y` are vectors. 