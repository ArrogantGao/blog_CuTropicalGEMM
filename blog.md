# CuTropicalGEMM.jl: A fast package for Matrix Multiplication of Tropical Numbers on GPU

In this blog, I would like to share some experience of developing a julia package called [CuTropicalGemm.jl](github.com/TensorBFS/CuTropicalGEMM.jl), which is used to calculate Generic Matrix Multiplication (GEMM) of Tropical Numbers on Nvidia GPUs.
In the following sections, I will introduce the background of this package, the technique we used and show some examples.

## What are tropical numbers and why we need them?

Tropical numbers, also known as the tropical semiring, is defined on $\mathbb{R} \cup \{\infty\}$ with addition and multiplication defined as $x \oplus y = \max\{(x,~y)\},~x \otimes y = x + y$, this algebra is also called the TropicalMaxPlus.
The following properties holds the tropical numbers

1. Addition $\oplus$ is commutative monoid and $0$ is its neutral element of addition.
2. Multiplication $\otimes$ is monoid and $0$ is its neutral element of multiplication.
3. Multiplication left and right distributes over addition.

In recent years, the tropical numbers have been widely used in various areas, including optimization, physics, and computer science, due to its computational simplicity.

For example, it was shown that solving the groud state energy of spin glass problem can be mapped as contraction of a tropical tensor network, which is actually calculating tons of matrix multiplication.
Although such reformulation did not actucally reduce the complexity of the problem, but in this way we can fully use the power of parallel computing technology developed in recent years and greatly speed up the computation.

For that purpose, a fast implementation of the TropicalGEMM on GPU is on demand.

## How we developed the package?

Of course the package should be fast and fully used the power of GPU, and we also want it easy to be used.
So basically there are two aspects:

1. An user friendly higer level interface,
2. A high performance lower level kernel.

In our package, the former is achieved via the type system in Julia, and the later is achieved via wrapped C-Cuda kernels.

### User friendly tropical interface

Type system of Julia allows us to create our own type, and then overload the operations of them. 
That is what we do in package [TropicalNumbers.jl](github.com/TensorBFS/TropicalNumbers.jl), simply by defining
```julia
abstract type AbstractSemiring <: Number end

struct Tropical{T} <: AbstractSemiring
    n::T

    Tropical{T}(x) where T = new{T}(T(x))
    function Tropical(x::T) where T
        new{T}(x)
    end
end
```
and then we overloaded the operations
```julia
Base.:*(a::Tropical, b::Tropical) = Tropical(a.n + b.n)
Base.:+(a::Tropical, b::Tropical) = Tropical(max(a.n, b.n))
```
Then users can simply use the tropical algebra, for example
```julia
julia> using TropicalNumbers

julia> a = Tropical(1.0)
1.0ₜ

julia> b = Tropical(2.0)
2.0ₜ

julia> a + b
2.0ₜ

julia> a * b
3.0ₜ
```
and opeartions of vectors and matrices also work
```julia
julia> A = Tropical.(rand(2, 2))
2×2 Matrix{Tropical{Float64}}:
 0.2238665251106623ₜ  0.18355043791779635ₜ
 0.3673107532619566ₜ   0.1573950170887196ₜ

julia> B = Tropical.(rand(2))
2-element Vector{Tropical{Float64}}:
 0.16479545470285972ₜ
  0.3666513822212566ₜ

julia> A * B
2-element Vector{Tropical{Float64}}:
 0.5502018201390529ₜ
 0.5321062079648163ₜ
```
Since we define the tropical number as a subtype of `Number`,
```julia
julia> isbitstype(Tropical{Float64})
true
```
which means the storage of tropical number in the memory is continium.

All these things work naturely, users will be able to use the tropical algebra just like real numbers.

### High performance CUDA kernel

For the lower level kernel, due to the requirement of performance, we choose to use `C-Cuda` directly and wrap the code with a Julia interface.
In this part, we simply used some skills common in GEMM kernels to speed up the code, and we learned a lot from the repo [CUDA_gemm](https://github.com/Cjkkkk/CUDA_gemm) by [Cjkkk](https://cjkkkk.github.io).
Here we will briefly introduce these skills, and for more detailed introduction, I recommand this [blog](https://zhuanlan.zhihu.com/p/441146275) in Chinese.

As we all know, GPU is fast because it have a large amount of cores, and the cores are grouped into blocks.
In the GPU kernels, we can allocate tasks to different blocks by decomposing them into a series of unrelated sub-tasks. Furthermore, we can further decompose the sub-tasks and assign them to threads for execution.
However, this large number of parallel tasks also leads to another issue, which is the significant data payload. 
Although the shared memory accessible by blocks is similar in speed to the L2 cache in CPUs, its capacity is limited. 
Therefore, it is not possible to load all data into shared memory at once, and only the data required for the current block calculation can be loaded each time.
Unfortunately, GEMM is a very memory-intensive operation, for example, when calculating the GEMM between a $M \times K$ matrix and a $K \times N$ matrix, if we use the naive way, i.e. evaluate the element in the result matrix one by one, we will have to load $M \times K \times N$ elements from the slow global memeory directly to registers in the whole process, and this generally far exceeds the data bandwidth of the GPU, resulting in severe performance issues.

To avoid the heavy data loading, we first split the target matrix into blocks with size $BM \times BN$, and each GPU block will be used to calculate one of the tiled matrix, as shown in the fig below: 

![Fig.1](figs/block.png)

When calculating each block, we will only need to load matrices with size $BM \times BK$ and $BK \times BN$ for~$K / BK$ times from global memory to shared memory.
In that case, the total data loading will be reduce to 
$$
    M \times N \times K \times \left( \frac{1}{BM} + \frac{1}{BN} \right)
$$
which is much smaller than the naive way.

Then in each block, we further tile the matrix and use the registers to store the data, as shown by

![Fig.1](figs/thread.png)

The tiled target matrix will be further divided as small matrices with size $TM \times TN$, and each thread will be used to calculate one of the tiled matrix.
During this process, the outer product way is used and data will be loaded from shared memory to registers, with the amount of $(TM + TN) \times BK$ in total.

Then we replaced the operations directly.
As we mentioned above, the `Tropical` type is `bitstype`, and the data storaged in memeories are simply the floating point numbers, which can be directly used by CUDA kernels.
Although the tropical algebra can not use the fused multiple add core (FMA), we found that the operation add/mul and max/min can be done on FMA and ALU parallelly, which means that we can use the FMA to do the add/mul and ALU to do the max/min at the same time.
Then after all the calculation is done, the target in the registers will be stored back to global memory directly.

For the boundary elements, a padding strategy is used, we simply set the element which are not acctually in the matrix as the zero element of the corresponding algebra, so that they will not effect the result of the calculation.
In our package, we set the parameters as
$$
    BM = 64,~BK = 32,~BN = 64,~TM = TN = 4.
$$

In the end, we wrapped the code in Julia, and overloaded the function `LinearAlgebra.mutmul!`, so that a simple $*$ for tropical matrices will call our function and use GPU in calculations.
Here is an example:
```julia
julia> using CUDA, LinearAlgebra, TropicalNumbers, CuTropicalGEMM

julia> A = CuArray(Tropical.(rand(2,2)))
2×2 CuArray{Tropical{Float64}, 2, CUDA.Mem.DeviceBuffer}:
  0.5054551076120295ₜ  0.2566654342554737ₜ
 0.40277483290611305ₜ  0.8717314798683612ₜ

julia> B = CuArray(Tropical.(rand(2,2)))
2×2 CuArray{Tropical{Float64}, 2, CUDA.Mem.DeviceBuffer}:
 0.7488281325136905ₜ  0.03728702805013795ₜ
 0.8437060742174199ₜ   0.9777629175478465ₜ

julia> A * B
2×2 CuArray{Tropical{Float64}, 2, CUDA.Mem.DeviceBuffer}:
   1.25428324012572ₜ  1.2344283518033201ₜ
 1.7154375540857811ₜ  1.8494943974162077ₜ
```
It is much faster than the previous CPU implementation `TropicalGEMM.jl`
```julia
julia> A = Tropical.(rand(1024,1024));

julia> B = Tropical.(rand(1024,1024));

julia> @time A * B;
  4.215667 seconds (6 allocations: 8.030 MiB)

julia> using TropicalGEMM

julia> @time A * B;
  0.095628 seconds (2 allocations: 8.000 MiB)

julia> CuA = CuArray(A);

julia> CuB = CuArray(B);

julia> @time CuA * CuB;
  0.001809 seconds (7 allocations: 256 bytes)
```

### Benchmark

A simple benchmark is conducted to test the performance of our package, as shown below.
We compared the performance of `CuTropicalGEMM.jl`, [GemmKernels.jl](https://github.com/JuliaGPU/GemmKernels.jl) and direct CUDA.jl map reduce on Tropical GEMM with single precision.
The test is done on using NVIDIA A800 80GB PCIe, and the performance of Cublas on normal GEMM is used as a reference of the maximum computing power.

![Benchmark](figs/benchmark.png)

Further benchmarking is still in development, and will be uoloaded to this repo [CuTropicalGEMM_benchmark](https://github.com/ArrogantGao/CuTropicalGEMM_benchmark).

## Further Optimization

The second aspect is to further optimize the code, especially the performance on narrow matrices.
As mentioned above, our package now is using the padding strategy to handle the boundary elements, and the minimum matrix size we process in each block on the GPU are $64 \times 32 \times 64$, which is optimal for large square matrices.
However, the performance of the code on narrow matrices is not good enough, and the reason is that the padding strategy will enlarge the matrix size a lot.
For example, when handling a matrix multiplication with size $4 \times 4 \times 10^6$, what is actually calculated is $64 \times 32 \times 10^6$, which means that the code will waste a lot of time on the padding elements.
Unfortunately, such narrow matrices are very common in the tensor network contraction process, and the performance of the code on narrow matrices is very important.

Now we are considering further optimize the code for narrow matrices, related code are stored in the branch [narrow matrices](https://github.com/TensorBFS/CuTropicalGEMM.jl/tree/narrow_matrices).

## Acknowledgement

I am very grateful for the guidance and assistance provided by Professor Liu during the project implementation process.
I would like to thank Tim Besard for his invaluable guidance and support during the development of the package, his expertise in GPU utilization have been immensely helpful. 
I also want to thank Tyler Thomas for his assistance in understanding the usage of BinaryBuilder.jl.
