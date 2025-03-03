# 6. Performance Considerations

## 6.1 Memory Coalescing

> **Core Concept**: Memory coalescing combines multiple memory accesses from threads in a warp into a single, efficient DRAM request, significantly improving global memory bandwidth utilization.

Global memory bandwidth is often a performance bottleneck in CUDA applications. Memory coalescing is a technique to efficiently utilize this bandwidth by organizing thread memory access patterns to match DRAM's burst-oriented architecture. This technique is often used alongside the tiling approach from Chapter 5 to maximize memory efficiency.

### Why Coalescing Matters

The global memory of CUDA devices is implemented with DRAM, which has relatively high access latency compared to the GPU's computational speed. Modern DRAM designs use parallelism to deliver data in "bursts" - accessing a range of consecutive locations at once rather than individual addresses. Memory coalescing takes advantage of this architecture.

When threads in a warp execute a load instruction, the hardware detects whether they access consecutive global memory locations. If so, these accesses are combined into a single request for consecutive locations. For example, if thread 0 accesses location X, thread 1 accesses X+1, thread 2 accesses X+2, and so on, all these accesses will be coalesced into a single memory transaction.

### Coalesced vs. Uncoalesced Access Patterns

**Row-Major Matrix (Coalesced):**
- In a matrix multiplication example with row-major storage, the array index is typically `k*Width+col`
- Since consecutive threads have consecutive values of `col`, they access consecutive memory addresses
- This creates coalesced access where threads in a warp access adjacent memory locations

**Column-Major Matrix (Uncoalesced):**
- With column-major storage, the array index becomes `col*Width+k`
- Consecutive threads now access memory locations that are Width elements apart
- This pattern cannot be coalesced, significantly reducing memory bandwidth utilization

### Optimization Strategies

When memory access patterns aren't naturally coalesced, several strategies can help:

1. **Rearrange thread mapping**: Change how threads are assigned to data elements to create coalesced access patterns

2. **Modify data layout**: Store data in formats (usually row-major for CUDA) that enable coalesced access

3. **Corner turning**: Use shared memory as an intermediate buffer:
   - Load data from global memory in a coalesced pattern
   - Reorganize the data in shared memory
   - Access the reorganized data for computation

**Example: Corner Turning for Matrix Multiplication**
When multiplying matrices where one is in row-major and one in column-major format:
- For the row-major matrix, threads load elements as usual with coalesced access
- For the column-major matrix, assign consecutive threads to load consecutive elements in the same column rather than row
- After loading into shared memory, threads can access the data in any pattern without penalty

These techniques allow CUDA applications to effectively utilize the available global memory bandwidth, which is crucial for achieving high performance in memory-bound applications.

## 6.2 Hiding Memory Latency

> **Core Concept**: Modern DRAM systems use multiple parallel structures (bursts, banks, and channels) to hide memory access latency and maximize bandwidth utilization.

As we explained in Section 6.1, DRAM bursting is a form of parallel organization: Multiple locations are accessed in the DRAM core array in parallel. However, bursting alone is not sufficient to realize the level of DRAM access bandwidth required by modern processors. DRAM systems typically employ two more forms of parallel organization: **banks** and **channels**.

### DRAM System Organization

At the highest level, a processor contains one or more channels. Each channel is a memory controller with a bus that connects a set of DRAM banks to the processor. Modern systems typically have one to eight channels, with multiple banks connected to each channel.

The data transfer bandwidth of a bus is defined by its width and clock frequency. Modern double data rate (DDR) busses perform two data transfers per clock cycle: one at the rising edge and one at the falling edge. For example:

- A 64-bit DDR bus with a 1 GHz clock has a bandwidth of 8B × 2 × 1 GHz = 16 GB/s
- A modern CPU might require at least 32 GB/s (2 channels)
- A modern GPU might require 256 GB/s (16 channels)

### The Role of Banks in Memory Performance

For each channel, the number of banks that connect to it is determined by the need to fully utilize the data transfer bandwidth of the bus. Each bank contains:
- An array of DRAM cells
- Sensing amplifiers for accessing cells
- Interface for delivering bursts of data to the bus

When a single bank is connected to a channel, memory access is highly inefficient:

1. The long latency for cell access (decoder enabling cells and charge sharing with amplifiers) must complete before data transfer
2. The data transfer time is typically much shorter than the access latency
3. If the ratio of access latency to data transfer time is 20:1, channel utilization would be only 4.8%

### Banking for Improved Bandwidth Utilization

When multiple banks are connected to a channel, accesses can be initiated in parallel:

- While one bank is transferring data, other banks can be accessing their cell arrays
- With two banks, channel utilization can potentially double
- If the ratio of cell array access latency to data transfer time is R, at least R+1 banks are needed to fully utilize channel bandwidth

More banks are beneficial for two key reasons:
1. Reducing **bank conflicts** (multiple accesses targeting the same bank)
2. Providing sufficient memory capacity while maintaining reasonable access latencies

### Thread Parallelism and Memory Organization

There is a critical connection between thread execution and DRAM organization:

- To achieve high memory bandwidth, sufficient threads must make simultaneous memory accesses
- Maximizing occupancy ensures enough threads are resident on SMs to:
  - Hide core pipeline latency (utilizing instruction throughput)
  - Hide DRAM access latency (utilizing memory bandwidth)
- Optimal performance requires memory accesses to be distributed across channels and banks, with coalesced access to each bank

### Interleaved Data Distribution

Modern memory systems distribute array elements across channels and banks in an interleaved pattern:

- Elements are spread sequentially across channels first, then banks
- This ensures even small arrays utilize multiple channels and banks
- Example: With 4 channels and 2 banks per channel:
  - Elements [0-1] → Channel 0, Bank 0
  - Elements [2-3] → Channel 1, Bank 0
  - Elements [4-5] → Channel 2, Bank 0
  - Elements [6-7] → Channel 3, Bank 0
  - Elements [8-9] → Channel 0, Bank 1
  - And so on...

### Practical Example: Tiled Matrix Multiplication

In tiled matrix multiplication:

- During each phase, thread blocks access different tiles of input matrices
- These accesses are distributed across channels and banks based on memory layout
- Multiple blocks executing in parallel create simultaneous memory requests to different memory subsystems
- GPU caches can combine duplicate accesses from different thread blocks
- As matrix size increases, memory accesses utilize more channels and banks

This demonstrates the symbiotic relationship between thread parallelism and memory organization:
- Thread parallelism creates simultaneous memory requests needed to utilize parallel DRAM structures
- Effective utilization of DRAM channels and banks is essential for achieving high execution throughput

> **Performance Insight**: For optimal memory performance, applications should maximize thread occupancy while ensuring memory accesses are well-distributed across channels and banks. This requires attention to both thread organization and data layout.

## 6.3 Thread Coarsening

Thread coarsening involves assigning multiple units of work to each thread instead of the finest parallelization granularity:

- In fine-grained parallelism, each thread handles the smallest possible work unit:
  - Vector addition: one element per thread
  - Image processing: one pixel per thread
  - Matrix multiplication: one output element per thread

- Advantages of fine-grained parallelism:
  - Maximizes transparent scalability
  - Fully exposes parallelism to hardware
  - Hardware can parallelize or serialize as needed

- Disadvantages emerge when parallelization has associated costs:
  - Redundant data loading across thread blocks
  - Duplicate computation
  - Synchronization overhead

- Thread coarsening reduces these costs by partially serializing work:
  - Example: In tiled matrix multiplication, adjacent output tiles reuse input tiles
  - Without coarsening: Each thread block loads its own copy of input tiles
  - With coarsening: One thread block processes multiple output tiles, loading input data once

### Implementation Considerations

- Thread coarsening implementation requires:
  - Adding a coarsening factor constant 
  - Replacing single column indices with starting column indices
  - Creating multiple result variables per thread
  - Using coarsening loops to process multiple elements
  - Loading input tiles once and reusing them for multiple outputs

### Potential Pitfalls

- Applying coarsening when unnecessary (when no parallelization cost exists)
- Over-coarsening leading to hardware underutilization
- Increased per-thread resource usage reducing occupancy

> **Performance Insight**: Thread coarsening is most effective when parallelization costs are high and hardware would otherwise serialize execution. The optimal coarsening factor is typically device-specific and dataset-specific, requiring tuning for different target platforms.

## 6.4 A Checklist of Optimizations

Table 6.1 summarizes key CUDA optimization strategies to consider:

### 1. Maximizing Occupancy
- **Benefits**: Hides pipeline and DRAM latency with more parallel work
- **Strategies**: Tune SM resource usage (threads per block, shared memory, registers)

### 2. Enabling Coalesced Global Memory Access
- **Benefits**: Reduces pipeline stalls, decreases memory traffic, better utilizes cache lines
- **Strategies**:
  - Transfer between global and shared memory in coalesced patterns
  - Perform uncoalesced accesses in shared memory (e.g., corner turning)
  - Rearrange thread-to-data mapping
  - Rearrange data layout

### 3. Minimizing Control Divergence
- **Benefits**: Improves SIMD efficiency, reduces idle cores
- **Strategies**:
  - Rearrange thread-to-work/data mapping
  - Rearrange data layout

### 4. Tiling Reused Data
- **Benefits**: Reduces pipeline stalls, decreases global memory traffic
- **Strategy**: Place reused data in shared memory/registers to load from global memory only once

### 5. Privatization
- **Benefits**: Reduces atomic update stalls and memory contention
- **Strategy**: Apply partial updates to private copies before updating universal copy

### 6. Thread Coarsening
- **Benefits**: Reduces redundant work, divergence, synchronization, and memory traffic
- **Strategy**: Assign multiple units of parallelism to each thread

> **Note**: These optimizations appear throughout different computation patterns covered in later chapters, often with context-specific implementations. 



### Exercises

```c
__global__ void foo_kernel(float* a, float* b, float* c, float* d, float* e) {
  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
  __shared__ float a_s[256];
  __shared__ float bc_s[4*256];
  a_s[threadIdx.x] = a[i];
  for(unsigned int j = 0; j < 4, ++j) {
    bc_s[j*256 + threadIdx.x] = b[j*blockDim.x*gridDim.x + i] + c[i*4 + j];
  }
  __syncthreads();
  d[i + 8] = a_s[threadIdx.x];
  e[i*8] = bc_s[threadIdx.x*4];
}
```

For each of the following memory accesses, specify whether they are coalesced or uncoalesced or coalescing is not applicable:
a. array a of line 5
coalesced
b. array a_s of like 5
coalesced or N/A
c. array b of line 7
uncoalesced
d. array c of line 7
uncoalesced
e. array bc_s of line 7
uncoalesced or N/A
f. array a_s of line 10
coalesced or N/A
g. array d of line 10
coalesced
h. array bc_s of line 11
uncoalesced or N/A
i. array e of line 11
uncoalesced


What is the floating point to global memory access ratio (in OP/B) of each of the following matrix-matrix multiplication kernels?
a. 