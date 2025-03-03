# Programming Massively Parallel Processors: A Hands-on Approach

## 4. GPU Architecture and Scheduling

> 💡 **Core Concept**: This chapter explores how GPUs physically execute CUDA programs through Streaming Multiprocessors (SMs), warp-based execution, and scheduling mechanisms that enable thousands of threads to run efficiently despite hardware limitations.

### 4.1 Architecture of a Modern GPU

#### Streaming Multiprocessors (SMs)

GPUs are organized into an array of highly threaded **Streaming Multiprocessors (SMs)**:

- Each SM contains multiple processing units called **CUDA cores**
- Cores within an SM share control logic and memory resources
- Modern GPUs have many SMs (e.g., NVIDIA A100 has 108 SMs with 64 cores each, totaling 6,912 cores)
- Global memory is off-chip device memory accessible by all SMs

```
┌─────────────────────── GPU ───────────────────────┐
│                                                    │
│  ┌────────┐ ┌────────┐ ┌────────┐     ┌────────┐  │
│  │   SM   │ │   SM   │ │   SM   │     │   SM   │  │
│  │ ┌────┐ │ │ ┌────┐ │ │ ┌────┐ │     │ ┌────┐ │  │
│  │ │Core│ │ │ │Core│ │ │ │Core│ │     │ │Core│ │  │
│  │ │Core│ │ │ │Core│ │ │ │Core│ │     │ │Core│ │  │
│  │ │Core│ │ │ │Core│ │ │ │Core│ │ ... │ │Core│ │  │
│  │ │Core│ │ │ │Core│ │ │ │Core│ │     │ │Core│ │  │
│  │ │Core│ │ │ │Core│ │ │ │Core│ │     │ │Core│ │  │
│  │ └────┘ │ │ └────┘ │ │ └────┘ │     │ └────┘ │  │
│  └────────┘ └────────┘ └────────┘     └────────┘  │
│                                                    │
│ ┌──────────────── Global Memory ─────────────────┐ │
│ └───────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────┘
```

> 🔑 **Key insight**: GPUs achieve high throughput by having many simpler cores rather than a few complex ones like CPUs. This design is optimized for data-parallel workloads where the same operation is performed on many data elements.

### 4.2 Block Scheduling

When a kernel is launched, the CUDA runtime system:

1. Creates a grid of threads that will execute the kernel
2. Assigns blocks to SMs on a block-by-block basis
3. All threads in a block are assigned to the same SM (never split across SMs)
4. Multiple blocks are likely assigned to the same SM (limited by SM resources)

#### Block Assignment Process

```
Kernel Launch
    │
    ▼
┌────────────────────┐
│ CUDA Runtime       │
│ ┌────────────────┐ │
│ │ Block Queue    │ │
│ │ ┌───┐┌───┐┌───┐│ │
│ │ │B12││B13││B14││ │ ─────┐
│ │ └───┘└───┘└───┘│ │      │     ┌──────┐     ┌──────┐
│ └────────────────┘ │      │     │  SM  │     │  SM  │
└────────────────────┘      │     │┌────┐│     │┌────┐│
                            └────►││B1  ││     ││B5  ││
                                  │└────┘│     │└────┘│
                                  │┌────┐│     │┌────┐│
                                  ││B2  ││     ││B6  ││
                                  │└────┘│     │└────┘│
                                  │┌────┐│     │┌────┐│
                                  ││B3  ││     ││B7  ││
                                  │└────┘│     │└────┘│
                                  │┌────┐│     │┌────┐│
                                  ││B4  ││     ││B8  ││
                                  │└────┘│     │└────┘│
                                  └──────┘     └──────┘
```

> 🔍 **Insight**: The CUDA runtime maintains a queue of blocks waiting to be executed. As SMs complete execution of blocks, new ones are assigned from this queue. This dynamic scheduling enables transparent scalability across different GPU models.

#### Block Scheduling Implications

- Each block must be able to execute independently of other blocks
- Blocks cannot reliably communicate with each other during execution
- The order of block execution is not guaranteed and may vary between runs
- The number of blocks that can run concurrently depends on the GPU resources

> ⚠️ **Important**: Never assume any particular execution order of blocks. This is why global synchronization across blocks is not directly supported in CUDA.

### 4.3 Synchronization and Transparent Scalability

#### Thread Synchronization with __syncthreads()

```c
__syncthreads();
```

- Acts as a barrier synchronization for all threads in a block
- All threads must reach the barrier before any can proceed
- Ensures threads in a block execute in lockstep at synchronization points
- Only works within a block (no global synchronization across blocks)

#### Example with Synchronization:

```c
__global__ void syncExample(float* data) {
    __shared__ float shared_data[256];
    
    // Load data into shared memory
    shared_data[threadIdx.x] = data[threadIdx.x];
    
    // Wait for all threads to complete their loads
    __syncthreads();
    
    // Now all threads can safely read shared_data
    // loaded by other threads
    float value = shared_data[255 - threadIdx.x];
    
    // ... rest of kernel
}
```

> ⚠️ **Warning**: If `__syncthreads()` is inside a conditional statement and some threads don't execute it, this will cause a deadlock. All threads in a block must execute the same `__syncthreads()` calls.

#### Transparent Scalability

CUDA achieves transparent scalability because:
- Blocks execute independently
- The runtime can distribute blocks across available SMs
- The same code works on different GPUs with varying numbers of SMs
- More powerful GPUs just process more blocks concurrently

> 🔑 **Key insight**: A CUDA program written for an entry-level GPU with few SMs will automatically utilize all SMs in a high-end GPU with no code changes. This is why block-level independence is a fundamental principle of CUDA programming.

### 4.4 Warps and SIMD Hardware

#### Warp Organization

Once a block is assigned to an SM, it is further divided into units of 32 threads called **warps**:

- A warp is the basic scheduling unit within an SM
- All threads in a warp execute the same instruction at the same time (SIMD)
- Warp size is 32 threads in current NVIDIA GPUs
- Warps are formed by consecutive thread IDs within a block

Examples of warp partitioning:
- Threads 0-31 form the first warp
- Threads 32-63 form the second warp
- And so on...

> 📝 **Note**: If block size is not a multiple of 32, the last warp will be partially filled with inactive threads (padding).

#### Warp Formation in Multidimensional Blocks

For multidimensional thread blocks, threads are first linearized in row-major order:

```
For a 8×4 block (2D):
┌───┬───┬───┬───┬───┬───┬───┬───┐
│ 0 │ 1 │ 2 │ 3 │ 4 │ 5 │ 6 │ 7 │ ← Warp 0 (threads 0-31)
├───┼───┼───┼───┼───┼───┼───┼───┤
│ 8 │ 9 │10 │11 │12 │13 │14 │15 │
├───┼───┼───┼───┼───┼───┼───┼───┤
│16 │17 │18 │19 │20 │21 │22 │23 │
├───┼───┼───┼───┼───┼───┼───┼───┤
│24 │25 │26 │27 │28 │29 │30 │31 │
└───┴───┴───┴───┴───┴───┴───┴───┘
```

> 🔍 **Insight**: Understanding warp formation is crucial for optimizing thread organization and memory access patterns. For example, using block dimensions that are multiples of 32 in the x-dimension can lead to better performance.

### 4.5 Control Divergence

#### SIMD Execution and Divergence

SIMD (Single Instruction, Multiple Data) hardware executes the same instruction across all threads in a warp:

- Efficient when all threads take the same execution path
- Inefficient when threads take different paths (e.g., due to conditionals)

```c
__global__ void divergentCode(int* data, int* result) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // This conditional causes control divergence within warps
    if (i % 2 == 0) {
        result[i] = data[i] * 2;  // Even threads
    } else {
        result[i] = data[i] + 10; // Odd threads
    }
}
```

#### How Divergence Affects Execution

When threads in a warp diverge:

1. The hardware executes each path serially (multiple passes)
2. Threads not on the current path are inactive (masked)
3. Performance decreases proportionally to the number of different paths

```
Warp execution with divergence:
┌───────────────────────┐
│ if (i % 2 == 0) {     │ All threads evaluate condition
├───────────────────────┤
│   result[i] = data[i] │ Only even threads active (odd masked)
│   * 2;                │
├───────────────────────┤
│ } else {              │
│   result[i] = data[i] │ Only odd threads active (even masked)
│   + 10;               │
├───────────────────────┤
│ }                     │
└───────────────────────┘
```

> ⚠️ **Performance warning**: Control divergence can reduce performance by up to 32× in worst-case scenarios (when each thread in a warp takes a different path).

#### Identifying and Minimizing Divergence

Common causes of control divergence:

1. Conditionals based on thread ID: `if (threadIdx.x < 16)`
2. Data-dependent conditionals: `if (data[i] > threshold)`
3. Loops with variable iteration counts: `for (int j = 0; j < data[i]; j++)`

Strategies to minimize divergence:

1. Align conditionals with warp boundaries when possible
2. Restructure algorithms to avoid thread-dependent conditionals
3. Consider sorting data to group similar execution paths together

> 🔍 **Practical tip**: Some divergence is unavoidable, especially for boundary checks. Focus optimization efforts on the most performance-critical sections of your kernels.

### 4.6 Warp Scheduling and Latency Tolerance

#### Oversubscription of Threads

SMs are assigned more warps than they can execute simultaneously:

- Only a subset of assigned warps can execute at any given time
- This deliberate oversubscription enables latency hiding

Example: A100 SM has 64 cores but can have up to 2,048 threads (64 warps) assigned

```
                  ┌──────────────────────┐
                  │ Streaming Multiprocessor │
                  │                      │
 ┌─────────┐      │  ┌─────────────────┐ │
 │ Active  │      │  │ Execution Units │ │
 │ Warps   │──────┼─►│   (64 Cores)    │ │
 └─────────┘      │  └─────────────────┘ │
      │           │                      │
      │           │  ┌─────────────────┐ │
      └───────────┼─►│    Scheduler    │ │
                  │  └─────────────────┘ │
 ┌─────────┐      │          ▲           │
 │ Stalled │      │          │           │
 │ Warps   │──────┘          │           │
 └─────────┘                 │           │
                             │           │
 waiting for memory, etc.    └───────────┘
```

#### Latency Hiding Mechanism

When a warp encounters a long-latency operation (e.g., global memory access):

1. The warp stalls, waiting for the operation to complete
2. The SM's warp scheduler selects another ready warp for execution
3. The stalled warp resumes once its operation completes and it's selected again

> 🔑 **Key insight**: This is why GPUs don't need large caches like CPUs – they hide memory latency through massive thread parallelism rather than caching.

#### Zero-Overhead Thread Scheduling

- Warp context switches occur in hardware with no overhead
- All warp states are kept resident on the SM
- No register saving/restoring as in traditional context switches
- Warps are selected for execution based on a priority mechanism

> 💡 **Understanding**: This latency hiding is why GPUs can dedicate more chip area to arithmetic units instead of caches and branch prediction – parallelism is used to hide latency rather than trying to eliminate it.

### 4.7 Resource Partitioning and Occupancy

#### SM Resource Limitations

Each SM has limited resources that must be shared among resident blocks:

1. **Registers**: Fast on-chip memory for thread-private variables
2. **Shared Memory**: On-chip memory shared within a block
3. **Thread slots**: Maximum number of threads per SM
4. **Block slots**: Maximum number of blocks per SM

Example A100 constraints:
- 65,536 registers per SM
- 164 KB shared memory per SM
- 2,048 max threads per SM
- 32 max blocks per SM

#### Occupancy Definition

**Occupancy** is the ratio of active warps to the maximum possible warps on an SM:

```
Occupancy = Active Warps / Maximum Warps per SM
```

Higher occupancy typically provides better latency hiding, but is not always correlated with peak performance.

> 📝 **Note**: 100% occupancy is not always necessary for optimal performance. Many kernels achieve peak performance at 50-75% occupancy.

#### Resource-Limited Occupancy Examples

1. **Register-limited example**:
   - SM supports 2,048 threads (64 warps)
   - Kernel uses 32 registers per thread
   - Total registers needed: 2,048 threads × 32 registers = 65,536 registers
   - If SM has only 65,536 registers, occupancy is 100%
   - If kernel used 40 registers, only 1,638 threads could be active (80% occupancy)

2. **Block-limited example**:
   - SM supports 16 blocks maximum
   - Kernel uses 64 threads per block
   - Maximum warps = 16 blocks × 64 threads ÷ 32 threads/warp = 32 warps
   - If SM supports 64 warps, occupancy would be 50%

> ⚠️ **Performance cliff warning**: When a resource limit is reached, adding just one more register per thread or a bit more shared memory can dramatically reduce occupancy, causing a "performance cliff."

#### Balancing Resource Usage

Strategies for optimizing occupancy:
1. Use fewer registers (compiler flags like `--maxrregcount`)
2. Use smaller thread blocks to increase the number of blocks per SM
3. Reduce shared memory usage
4. Consider kernel splitting to reduce per-thread resource needs

> 🔍 **Practical tip**: Use the CUDA Occupancy Calculator or `cudaOccupancyMaxActiveBlocksPerMultiprocessor()` to determine the limiting factor for your kernel.

### 4.8 Querying Device Properties

CUDA provides API functions to query device capabilities at runtime:

```c
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, 0);  // Get properties for device 0

// Display key properties
printf("Device name: %s\n", prop.name);
printf("Compute capability: %d.%d\n", prop.major, prop.minor);
printf("SMs: %d\n", prop.multiProcessorCount);
printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
printf("Max threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
printf("Max blocks per SM: %d\n", prop.maxBlocksPerMultiProcessor);
printf("Shared memory per SM: %lu KB\n", prop.sharedMemPerMultiprocessor / 1024);
printf("Registers per SM: %d\n", prop.regsPerMultiprocessor);
```

> 💡 **Practical application**: Use device properties to dynamically adjust kernel launch parameters based on the specific GPU your application is running on.

### 4.9 Key Takeaways

- **GPU Architecture**: GPUs consist of multiple SMs, each containing many CUDA cores that share control logic and memory resources.

- **Block Scheduling**: Blocks are assigned to SMs for independent execution, enabling transparent scalability across different GPU models.

- **Thread Synchronization**: `__syncthreads()` provides barrier synchronization within a block, but not across blocks.

- **Warp Execution**: Threads are executed in warps of 32 threads following the SIMD model, with all threads in a warp executing the same instruction simultaneously.

- **Control Divergence**: When threads in a warp take different paths, execution serializes, reducing performance. Minimize divergence when possible.

- **Latency Hiding**: GPUs tolerate long-latency operations by maintaining many more threads than can execute simultaneously, switching between them with zero overhead.

- **Occupancy**: The ratio of active warps to maximum possible warps affects performance. It's limited by register, shared memory, thread, and block constraints.

- **Resource Balance**: Optimizing for peak performance requires balancing register usage, shared memory, and thread organization to achieve sufficient (but not necessarily maximum) occupancy.

***

**Exercise Ideas**:
1. Calculate the occupancy for a kernel with different resource requirements on your specific GPU
2. Analyze a kernel for potential control divergence and suggest optimizations
3. Experiment with different block sizes to find the optimal configuration for a specific algorithm 