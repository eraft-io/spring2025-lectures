import time
from typing import Callable
import torch
import torch.nn as nn
from torch.profiler import ProfilerActivity
from torch.utils.cpp_extension import load_inline
import triton
import triton.language as tl
from execute_util import text, link, image
from file_util import ensure_directory_exists
from lecture_util import article_link
from torch_util import get_device
from lecture_06_utils import check_equal, check_equal2, get_local_url, round1, mean
import os

def main():
    announcements()

    text("Last lecture: high-level overview of GPUs and performance <br/> 上节课：GPU 和性能的高层概述")
    text("This lecture: benchmarking/profiling + write kernels <br/> 本节课：基准测试/性能分析 + 编写内核")

    if not torch.cuda.is_available():
        text("You should run this lecture on a GPU to get the full experience. <br/> 你应该在 GPU 上运行本节课以获得完整体验。")

    review_of_gpus()
    benchmarking_and_profiling()  # Important for understanding!

    kernel_fusion_motivation()
    cuda_kernels()  # Write kernels in CUDA/C++
    triton_kernels()  # Write kernels in Python
    pytorch_compilation()  # Don't write kernels at all?

    # More advanced computations
    triton_softmax_main()

    text("## Summary <br/> ## 总结")

    text("Gap between the programming model (PyTorch, Triton, PTX) and hardware => performance mysteries <br/> 编程模型 (PyTorch, Triton, PTX) 与硬件之间的差距 => 性能奥秘")

    text("Benchmarking for understanding scaling <br/> 基准测试用于理解扩展性")
    text("Profiling for understanding internals of PyTorch functions (bottoms out with kernels) <br/> 性能分析用于理解 PyTorch 函数的内部机制 (从内核底层)")
    text("Looking at PTX assembly to understand internals of CUDA kernels <br/> 查看 PTX 汇编以理解 CUDA 内核的内部机制")

    text("5 ways to write a function: manual, PyTorch, compiled, CUDA, Triton <br/> 编写函数的5种方式：手动、PyTorch、编译、CUDA、Triton")
    text("GeLU (element-wise), softmax (row-wise), matmul (complex aggregation) <br/> GeLU (逐元素)、softmax (逐行)、matmul (复杂聚合)")

    text("Key principle: organize computation to minimize reads/writes <br/> 关键原则：组织计算以最小化读写")
    text("Key ideas: kernel fusion (warehouse/factory analogy), tiling (shared memory) <br/> 关键思路：内核融合 (仓库/工厂类比)、分块 (共享内存)")
    text("Automatic compilers (Triton, torch.compile) will get better over time <br/> 自动编译器 (Triton, torch.compile) 会随着时间推移变得更好")

    further_reading()


def announcements():
    text("Assignment 1 leaderboard <br/> 作业 1 排行榜 "), link(title="[Leaderboard]", url="https://github.com/stanford-cs336/spring2025-assignment1-basics-leaderboard")
    text("Assignment 2 is out <br/> 作业 2 已发布 "), link(title="[A2]", url="https://github.com/stanford-cs336/spring2025-assignment2-systems")


def review_of_gpus():
    text("## Hardware <br/> ## 硬件")
    image("https://miro.medium.com/v2/resize:fit:2000/format:webp/1*6xoBKi5kL2dZpivFe1-zgw.jpeg", width=800)
    text("Compute: streaming multiprocessors (SMs) [A100: 108] <br/> 计算：流多处理器 (SMs) [A100: 108]")
    text("Memory: <br/> 内存：")
    text("- DRAM [A100: 80GB] - big, slow <br/> - DRAM [A100: 80GB] - 大、慢")
    text("- L2 cache [A100: 40MB] <br/> - L2 缓存 [A100: 40MB]")
    text("- L1 cache [A100: 192KB per SM] - small, fast <br/> - L1 缓存 [A100: 每 SM 192KB] - 小、快")

    text("You can look at the specs on your actual GPU. <br/> 你可以查看你实际 GPU 的规格。")
    print_gpu_specs()

    text("Basic structure: run f(i) for all i = 0, ..., N-1 <br/> 基本结构：对所有 i = 0, ..., N-1 运行 f(i)")

    text("## Execution model <br/> ## 执行模型")
    image("https://docs.nvidia.com/cuda/parallel-thread-execution/_images/grid-with-CTAs.png", width=600)
    text("- *Thread*: process individual index (i.e., f(i)) <br/> - *线程*：处理单个索引 (即 f(i))")
    text("- *Thread block* (a.k.a. concurrent thread arrays): scheduled on a single SM <br/> - *线程块* (又名并发线程数组)：在单个 SM 上调度")
    text("- *Grid*: collection of thread blocks <br/> - *网格*：线程块的集合")

    text("Why thread blocks? Shared memory. <br/> 为什么要有线程块？共享内存。")
    text("- Intuition: group f(i)'s that read similar data together <br/> - 直觉：将读取相似数据的 f(i) 组合在一起")
    text("- Threads within a thread block have shared memory (as fast as L1 cache) [A100: 164KB] <br/> - 线程块内的线程拥有共享内存 (与 L1 缓存一样快) [A100: 164KB]")
    text("- Can synchronize threads (for reading/writing) within a block (but not across blocks) <br/> - 可以在块内同步线程 (用于读/写)，但不能跨块同步")

    text("### Hardware and execution interact. <br/> ### 硬件和执行相互影响。")
    image("https://developer-blogs.nvidia.com/wp-content/uploads/2019/06/pasted-image-0.png", width=400)
    text("Thread blocks scheduled onto SMs in waves. <br/> 线程块以波次方式调度到 SM 上。")
    text("Problem: last wave has fewer thread blocks, leaving some SMs idle (low occupancy). <br/> 问题：最后一道波次的线程块较少，导致一些 SM 空闲 (低占用率)。")
    text("Wave quantization: make number of thread blocks divide # SMs. <br/> 波次量化：使线程块数量能被 SM 数量整除。")
    text("Rule of thumb: number of thread blocks should be >= 4x # SMs <br/> 经验法则：线程块数量应该 >= 4 倍的 SM 数量")
    text("Challenge: some aspects of hardware are hidden from the execution model (e.g., scheduling, # SMs). <br/> 挑战：硬件的某些方面对执行模型是隐藏的 (如调度、SM 数量)。")

    text("### Arithmetic intensity: # FLOPs / # bytes <br/> ### 算术强度：# FLOPs / # 字节")
    text("- If high, operation is compute-bound (good) <br/> - 如果高，操作是计算绑定的 (好)")
    text("- If low, operation is memory-bound (bad) <br/> - 如果低，操作是内存绑定的 (坏)")
    text("General rule: matrix multiplication is compute-bound, everything else is memory-bound <br/> 一般规则：矩阵乘法是计算绑定的，其他都是内存绑定的")


def benchmarking_and_profiling():
    text("IMPORTANT: benchmark/profile your code! <br/> 重要：请对你的代码进行基准测试/性能分析！")

    text("You can read spec sheets (marketing material) and papers <br/> 你可以阅读规格表 (营销材料) 和论文")
    text("...but performance depends on your library version, your hardware, your workload <br/> ...但性能取决于你的库版本、硬件和工作负载")
    text("...so there is no substitute for benchmarking/profiling your code. <br/> ...所以没有什么可以替代对你的代码进行基准测试/性能分析。")

    text("Example computation: running forward/backward passes on an MLP. <br/> 示例计算：运行 MLP 的前向/反向传播。")
    run_mlp(dim=128, num_layers=16, batch_size=128, num_steps=5)

    benchmarking()       # How long does it take?
    profiling()          # Where time is being spent?

    text("Every time you make a change, benchmark/profile! <br/> 每次你做修改时，都要进行基准测试/性能分析！")


class MLP(nn.Module):
    """Simple MLP: linear -> GeLU -> linear -> GeLU -> ... -> linear -> GeLU"""
    def __init__(self, dim: int, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(dim, dim) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
            x = torch.nn.functional.gelu(x)
        return x


def run_mlp(dim: int, num_layers: int, batch_size: int, num_steps: int) -> Callable:
    # Define a model (with random weights)
    model = MLP(dim, num_layers).to(get_device())

    # Define an input (random)
    x = torch.randn(batch_size, dim, device=get_device())

    def run():
        # Run the model `num_steps` times (note: no optimizer updates)
        for step in range(num_steps):
            # Forward
            y = model(x).mean()

            # Backward
            y.backward()

    return run


def run_operation1(dim: int, operation: Callable) -> Callable:
    # Setup: create one random dim x dim matrices
    x = torch.randn(dim, dim, device=get_device())
    # Return a function to perform the operation
    return lambda : operation(x)


def run_operation2(dim: int, operation: Callable) -> Callable:
    # Setup: create two random dim x dim matrices
    x = torch.randn(dim, dim, device=get_device())
    y = torch.randn(dim, dim, device=get_device())
    # Return a function to perform the operation
    return lambda : operation(x, y)


def benchmarking():
    text("Benchmarking measures the wall-clock time of performing some operation. <br/> 基准测试测量执行某个操作的挂钟时间。")

    text("It only gives you end-to-end time, not where time is spent (profiling). <br/> 它只给你端到端的时间，而不是时间花在哪里 (性能分析)。")

    text("It is still useful for: <br/> 它仍然对以下方面有用：")
    text("- comparing different implementations (which is faster?), and <br/> - 比较不同实现 (哪个更快？)，以及")
    text("- understanding how performance scales (e.g., with dimension). <br/> - 理解性能如何扩展 (例如随维度变化)。")

    text("Let's define a convenient function for benchmarking an arbitrary function. <br/> 让我们定义一个方便的函数来对任意函数进行基准测试。")
    benchmark("sleep", lambda : time.sleep(50 / 1000))

    text("### Benchmarking matrix multiplication <br/> ### 基准测试矩阵乘法")
    text("First, let us benchmark matrix multiplication of square matrices. <br/> 首先，让我们对方阵的矩阵乘法进行基准测试。")
    if torch.cuda.is_available():
        dims = (1024, 2048, 4096, 8192, 16384)  # @inspect dims
    else:
        dims = (1024, 2048)  # @inspect dims
    
    matmul_results = [] 
    for dim in dims:
        # @ inspect dim
        result = benchmark(f"matmul(dim={dim})", run_operation2(dim=dim, operation=lambda a, b: a @ b))
        matmul_results.append((dim, result))  # @inspect matmul_results

    text("Let us benchmark our MLP! <br/> 让我们对我们的 MLP 进行基准测试！")
    dim = 256  # @inspect dim
    num_layers = 4  # @inspect num_layers 
    batch_size = 256  # @inspect batch_size
    num_steps = 2  # @inspect num_steps

    mlp_base = benchmark("run_mlp", run_mlp(dim=dim, num_layers=num_layers, batch_size=batch_size, num_steps=num_steps)) # @inspect mlp_base


    text("Scale the number of steps. <br/> 扩展步数。")
    step_results = []
    for scale in (2, 3, 4, 5):
        result = benchmark(f"run_mlp({scale}x num_steps)", 
                         run_mlp(dim=dim, num_layers=num_layers, 
                                batch_size=batch_size, num_steps=scale * num_steps)) # @inspect result, @inspect scale, @inspect num_steps
        step_results.append((scale, result))  # @inspect step_results

    text("Scale the number of layers. <br/> 扩展层数。")
    layer_results = []
    for scale in (2, 3, 4, 5):
        result = benchmark(f"run_mlp({scale}x num_layers)", 
                         run_mlp(dim=dim, num_layers=scale * num_layers, 
                                batch_size=batch_size, num_steps=num_steps)) # @inspect result, @inspect scale, @inspect num_layers, @inspect num_steps
        layer_results.append((scale, result))  # @inspect layer_results

    text("Scale the batch size. <br/> 扩展批量大小。")
    batch_results = []
    for scale in (2, 3, 4, 5):
        result = benchmark(f"run_mlp({scale}x batch_size)", 
                         run_mlp(dim=dim, num_layers=num_layers, 
                                batch_size=scale * batch_size, num_steps=num_steps)) # @inspect result, @inspect scale, @inspect num_layers, @inspect num_steps
        batch_results.append((scale, result))  # @inspect batch_results

    text("Scale the dimension. <br/> 扩展维度。")
    dim_results = []
    for scale in (2, 3, 4, 5):
        result = benchmark(f"run_mlp({scale}x dim)", 
                         run_mlp(dim=scale * dim, num_layers=num_layers, 
                                batch_size=batch_size, num_steps=num_steps)) # @inspect result, @inspect scale, @inspect num_layers, @inspect num_steps
        dim_results.append((scale, result))  # @inspect dim_results

    text("The timings are not always predictable due to the non-homogenous nature of CUDA kernels, hardware, etc. <br/> 由于 CUDA 内核、硬件等的非均匀特性，时间并不总是可预测的。")

    text("You can also use `torch.utils.benchmark`, which provides more amenities. <br/> 你也可以使用 `torch.utils.benchmark`，它提供了更多便利。"), 
    link("https://pytorch.org/tutorials/recipes/recipes/benchmark.html")
    text("We did not use this to make benchmarking more transparent. <br/> 我们没有使用这个来使基准测试更透明。")


def benchmark(description: str, run: Callable, num_warmups: int = 1, num_trials: int = 3):
    """Benchmark `func` by running it `num_trials`, and return all the times."""
    # Warmup: first times might be slower due to compilation, things not cached.
    # Since we will run the kernel multiple times, the timing that matters is steady state.
    for _ in range(num_warmups):
        run()
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # Wait for CUDA threads to finish (important!)

    # Time it for real now!
    times: list[float] = [] # @inspect times, @inspect description
    for trial in range(num_trials):  # Do it multiple times to capture variance
        start_time = time.time()

        run()  # Actually perform computation
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # Wait for CUDA threads to finish (important!)

        end_time = time.time()
        times.append((end_time - start_time) * 1000) # @inspect times

    mean_time = mean(times) # @inspect mean_time
    return mean_time


def profiling():
    text("While benchmarking looks at end-to-end time, profiling looks at where time is spent. <br/> 基准测试查看端到端时间，性能分析查看时间花在哪里。")
    text("Obvious: profiling helps you understand where time is being spent. <br/> 显而易见：性能分析帮助你理解时间花在哪里。")
    text("Deeper: profiling helps you understand (what is being called). <br/> 更深入：性能分析帮助你理解 (什么被调用)。")

    text("PyTorch has a nice built-in profiler <br/> PyTorch 有一个很好的内置性能分析器 "), link("https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html")

    text("Let's profile some code to see what is going on under the hood. <br/> 让我们对一些代码进行性能分析，看看底层发生了什么。")
    sleep_function = lambda : time.sleep(50 / 1000)
    sleep_profile = profile("sleep", sleep_function) 
    text(f"## sleep")
    text(sleep_profile, verbatim=True)
    

    text("Let's start with some basic operations. <br/> 让我们从一些基本操作开始。")
    add_function = lambda a, b: a + b
    add_profile = profile("add", run_operation2(dim=2048, operation=add_function))
    text(f"## add")
    text(add_profile, verbatim=True)

    matmul_function = lambda a, b: a @ b
    matmul_profile = profile("matmul", run_operation2(dim=2048, operation=matmul_function))
    text(f"## matmul")
    text(matmul_profile, verbatim=True)

    matmul_function_128 = lambda a, b: a @ b
    matmul_profile_128 = profile("matmul(dim=128)", run_operation2(dim=128, operation=matmul_function_128))
    text(f"## matmul(dim=128)")
    text(matmul_profile_128, verbatim=True)

    text("Observations <br/> 观察")
    text("- You can see what CUDA kernels are actually being called. <br/> - 你可以看到实际上调用了哪些 CUDA 内核。")
    text("- Different CUDA kernels are invoked depending on the tensor dimensions. <br/> - 根据张量维度会调用不同的 CUDA 内核。")

    text("Name of CUDA kernel tells us something about the implementation. <br/> CUDA 内核的名称告诉我们一些关于实现的信息。")
    text("Example: cutlass_80_simt_sgemm_256x128_8x4_nn_align1 <br/> 示例：cutlass_80_simt_sgemm_256x128_8x4_nn_align1")
    text("- cutlass: NVIDIA's CUDA library for linear algebra <br/> - cutlass：NVIDIA 的线性代数 CUDA 库")
    text("- 256x128: tile size <br/> - 256x128：分块大小")

    text("Let's now look at some composite operations. <br/> 现在让我们看一些复合操作。")
    cdist_function = lambda a, b: torch.cdist(a, b)
    cdist_profile = profile("cdist", run_operation2(dim=2048, operation=cdist_function))
    text(f"## cdist")
    text(cdist_profile, verbatim=True)

    gelu_function = lambda a, b: torch.nn.functional.gelu(a + b)
    gelu_profile = profile("gelu", run_operation2(dim=2048, operation=gelu_function))
    text(f"## gelu")
    text(gelu_profile, verbatim=True)

    softmax_function = lambda a, b: torch.nn.functional.softmax(a + b, dim=-1)
    softmax_profile = profile("softmax", run_operation2(dim=2048, operation=softmax_function))
    text(f"## softmax")
    text(softmax_profile, verbatim=True)

    text("Now let's profile our MLP. <br/> 现在让我们对我们的 MLP 进行性能分析。")
    text("We will also visualize our stack trace using a flame graph, which reveals where time is being spent. <br/> 我们还将使用火焰图可视化堆栈跟踪，以显示时间花在哪里。")
    if torch.cuda.is_available():
        mlp_profile = profile("mlp", run_mlp(dim=2048, num_layers=64, batch_size=1024, num_steps=2), with_stack=True)
    else:
        mlp_profile = profile("mlp", run_mlp(dim=128, num_layers=16, batch_size=128, num_steps=2), with_stack=True)
    text(f"## mlp")
    text(mlp_profile, verbatim=True)


def profile(description: str, run: Callable, num_warmups: int = 1, with_stack: bool = False):
    # Warmup
    for _ in range(num_warmups):
        run()
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # Wait for CUDA threads to finish (important!)

    # Run the code with the profiler
    with torch.profiler.profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            # Output stack trace for visualization
            with_stack=with_stack,
            # Needed to export stack trace for visualization
            experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True)) as prof:
        run()
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # Wait for CUDA threads to finish (important!)

    # Print out table
    table = prof.key_averages().table(sort_by="cuda_time_total",
                                      max_name_column_width=80,
                                      row_limit=10)
    #text(f"## {description}")
    #text(table, verbatim=True)

    # Write stack trace visualization
    if with_stack:
        text_path = f"var/stacks_{description}.txt"
        svg_path = f"var/stacks_{description}.svg"
        prof.export_stacks(text_path, "self_cuda_time_total")

    return table

def kernel_fusion_motivation():
    text("Horace He's blog post <br/> Horace He 的博客文章 "), link(title="[Article]", url="https://horace.io/brrr_intro.html")

    text("Analogy: warehouse : DRAM :: factory : SRAM <br/> 类比：仓库 : DRAM :: 工厂 : SRAM")
    image("https://horace.io/img/perf_intro/factory_bandwidth.png", width=800)

    text("Each operation needs to read/compute/write: <br/> 每个操作需要读取/计算/写入：")
    image("https://horace.io/img/perf_intro/multi_operators.png", width=800)

    text("If we *fuse* the operations, only need to read/write once: <br/> 如果我们*融合*这些操作，只需要读写一次：")
    image("https://horace.io/img/perf_intro/operator_fusion.png", width=800)

    text("To see the effect of fusion, let's consider the GeLU activation function. <br/> 为了看到融合的效果，让我们考虑 GeLU 激活函数。 "), 
    link("https://pytorch.org/docs/stable/generated/torch.nn.GELU.html")

    text("Let's consider two ways to compute GeLU: <br/> 让我们考虑两种计算 GeLU 的方式：")
    x = torch.tensor([1.])  # @inspect x

    text("1. The default PyTorch implementation (fused): <br/> 1. 默认的 PyTorch 实现 (融合的)：")
    y1 = pytorch_gelu(x)  # @inspect y1

    text("2. We can also write our own by hand (not fused): <br/> 2. 我们也可以自己手写 (非融合的)：")
    y2 = manual_gelu(x)  # @inspect y2

    # Check that the implementations match
    assert torch.allclose(y1, y2)

    # Check more systematically
    check_equal(pytorch_gelu, manual_gelu)

    text("Let's benchmark. <br/> 让我们做基准测试。")
    manual_time = benchmark("manual_gelu", run_operation1(dim=16384, operation=manual_gelu)) # @inspect manual_time
    pytorch_time = benchmark("pytorch_gelu", run_operation1(dim=16384, operation=pytorch_gelu)) # @inspect pytorch_time
    if manual_time is not None and pytorch_time is not None:
        text(f"The fused version is significantly faster: {manual_time:.2f} ms, {pytorch_time:.2f} ms <br/> 融合版本明显更快：{manual_time:.2f} 毫秒，{pytorch_time:.2f} 毫秒")
    else:
        text("Could not compare times - benchmark results were None <br/> 无法比较时间 - 基准测试结果为 None")

    text("Let's look under the hood. <br/> 让我们看看底层。")
    manual_gelu_profile = profile("manual_gelu", run_operation1(dim=16384, operation=manual_gelu))
    text(f"## manual_gelu")
    text(manual_gelu_profile, verbatim=True)
    pytorch_gelu_profile = profile("pytorch_gelu", run_operation1(dim=16384, operation=pytorch_gelu))
    text(f"## pytorch_gelu")
    text(pytorch_gelu_profile, verbatim=True)
    text("The PyTorch just calls one kernel whereas the others are atomic (remember the warehouse/factory) <br/> PyTorch 只调用一个内核，而其他的是原子的 (记住仓库/工厂) ")

    text(f"## Look at Nsight profiler for MLP   <br/> ## 查看 MLP 的 Nsight 性能分析器   ")


def cuda_kernels():
    text("Now let's open the box to understand what's going on inside a CUDA kernel by writing our own. <br/> 现在让我们打开黑盒子，通过编写自己的 CUDA 内核来理解内部发生了什么。")

    text("Let's write the GeLU function in CUDA. <br/> 让我们用 CUDA 编写 GeLU 函数。")
    cuda_gelu = create_cuda_gelu() # @inspect cuda_gelu
    x = manual_gelu # @inspect x

    text("Check correctness of our implementation. <br/> 检查我们实现的正确性。")
    if cuda_gelu is not None:
        check_equal(cuda_gelu, manual_gelu)

    text("Benchmark our CUDA version. <br/> 对我们的 CUDA 版本进行基准测试。")
    pytorch_time = benchmark("pytorch_gelu", run_operation1(dim=16384, operation=pytorch_gelu)) # @inspect pytorch_time
    manual_time = benchmark("manual_gelu", run_operation1(dim=16384, operation=manual_gelu)) # @inspect manual_time
    if cuda_gelu is not None:
        cuda_time = benchmark("cuda_gelu", run_operation1(dim=16384, operation=cuda_gelu)) # @inspect cuda_time 
        cuda_gelu_profile = profile("cuda_gelu", run_operation1(dim=16384, operation=cuda_gelu))
        text(f"## cuda_gelu")
        text(cuda_gelu_profile, verbatim=True)
    text("Our CUDA implementation is faster than manual, but not as good as PyTorch. <br/> 我们的 CUDA 实现比手写的快，但不如 PyTorch。")

    text("Elementwise operations are easy in CUDA (though you can still be smarter). <br/> 逐元素操作在 CUDA 中很容易 (虽然你仍然可以更聪明)。")
    text("But most interesting operations (e.g., matmul, softmax, RMSNorm) require reading multiple values. <br/> 但大多数有趣的操作 (如 matmul、softmax、RMSNorm) 需要读取多个值。")
    text("For that, you have to think about managing shared memory, etc. <br/> 为此，你必须考虑管理共享内存等。")


def create_cuda_gelu():
    text("CUDA is an extension of C/C++ with APIs for managing GPUs. <br/> CUDA 是 C/C++ 的扩展，带有管理 GPU 的 API。")

    text("Simplified picture: write f(i), CUDA kernel computes f(i) for all i. <br/> 简化图景：写 f(i)，CUDA 内核为所有 i 计算 f(i)。")

    image("https://docs.nvidia.com/cuda/parallel-thread-execution/_images/grid-with-CTAs.png", width=0.5)
    text("Grid: collection of thread blocks: numBlocks = (2, 4), blockDim = (1, 8) <br/> 网格：线程块的集合：numBlocks = (2, 4), blockDim = (1, 8)")
    text("Thread block: collection of threads: blockIdx = (0, 1) <br/> 线程块：线程的集合：blockIdx = (0, 1)")
    text("Thread: single unit of operation: threadIdx = (0, 3). <br/> 线程：操作的基本单元：threadIdx = (0, 3)。")

    text("You write code that a thread execute, using (blockIdx, blockDim, threadIdx) to determine what to do. <br/> 你编写线程执行的代码，使用 (blockIdx, blockDim, threadIdx) 来确定要做什么。")

    text("Set CUDA_LAUNCH_BLOCKING so that if there are errors, CUDA will tell you what went wrong. <br/> 设置 CUDA_LAUNCH_BLOCKING，这样如果有错误，CUDA 会告诉你哪里出了问题。")
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    text("The `load_inline` function makes it convenient to write CUDA code and bind it to a Python module for immediate use. <br/> `load_inline` 函数使得编写 CUDA 代码并将其绑定到 Python 模块以便立即使用变得方便。")

    # CUDA code: has the full logic
    cuda_gelu_src = open("gelu.cu").read()
    text(cuda_gelu_src, verbatim=True)

    # C++ code: defines the gelu function
    cpp_gelu_src = "torch::Tensor gelu(torch::Tensor x);"

    text("Compile the CUDA code and bind it to a Python module. <br/> 编译 CUDA 代码并将其绑定到 Python 模块。")
    ensure_directory_exists("var/cuda_gelu")
    if not torch.cuda.is_available():
        return None
    module = load_inline(
        cuda_sources=[cuda_gelu_src],
        cpp_sources=[cpp_gelu_src],
        functions=["gelu"],
        extra_cflags=["-O2"],
        verbose=True,
        name="inline_gelu",
        build_directory="var/cuda_gelu",
    )

    cuda_gelu = getattr(module, "gelu")
    return cuda_gelu


def triton_kernels():
    triton_introduction()
    triton_gelu_main()


def triton_introduction():
    text("Developed by OpenAI in 2021 <br/> 由 OpenAI 于 2021 年开发 "), 
    link("https://openai.com/research/triton")

    text("Make GPU programming more accessible <br/> 使 GPU 编程更容易上手")
    text("- Write in Python <br/> - 用 Python 编写")
    text("- Think about thread blocks rather than threads <br/> - 考虑线程块而不是线程")

    text("What does Triton offer?", verbatim=True)
    text("                                             CUDA      Triton", verbatim=True)
    text("- Memory coalescing (transfer from DRAM)     manual    automatic", verbatim=True)
    text("- Shared memory management                   manual    automatic", verbatim=True)
    text("- Scheduling within SMs                      manual    automatic", verbatim=True)
    text("- Scheduling across SMs                      manual    manual", verbatim=True)

    text("Compiler does more work, can actually outperform PyTorch implementations! <br/> 编译器做了更多工作，实际上可以超越 PyTorch 的实现！")


def triton_gelu_main():
    if not torch.cuda.is_available():
        return

    text("One big advantage of Triton is that you can step through the Python code. <br/> Triton 的一个主要优势是你可以逐步执行 Python 代码。")

    text("Let's step through a Triton kernel. <br/> 让我们逐步执行一个 Triton 内核。")
    x = torch.randn(8192, device=get_device())
    y1 = triton_gelu(x)

    print_ptx_main()  # Look at the generated instructions

    text("Check that it's correct. <br/> 检查它是否正确。")
    check_equal(triton_gelu, manual_gelu)

    text("Let's now benchmark it compared to the PyTorch and CUDA implementations. <br/> 现在让我们与 PyTorch 和 CUDA 实现进行基准测试对比。")
    text("Remember to set TRITON_INTERPRET=0 for good performance. <br/> 记得设置 TRITON_INTERPRET=0 以获得良好性能。")
    manual_time = benchmark("manual_gelu", run_operation1(dim=16384, operation=manual_gelu)) # @inspect manual_time
    pytorch_time = benchmark("pytorch_gelu", run_operation1(dim=16384, operation=pytorch_gelu)) # @inspect pytorch_time
    cuda_time = benchmark("cuda_gelu", run_operation1(dim=16384, operation=create_cuda_gelu())) # @inspect cuda_time
    triton_time = benchmark("triton_gelu", run_operation1(dim=16384, operation=triton_gelu)) # @inspect triton_time

    triton_gelu_profile = profile("triton_gelu", run_operation1(dim=16384, operation=triton_gelu))
    text(f"## triton_gelu")
    text(triton_gelu_profile, verbatim=True)

    text("Our Triton implementation (triton_gelu): <br/> 我们的 Triton 实现 (triton_gelu)：")
    text("- is almost as good as the PyTorch implementation (pytorch_gelu). <br/> - 与 PyTorch 实现 (pytorch_gelu) 几乎一样好。")
    text("- is actually slower than our naive CUDA implementation (cuda_gelu). <br/> - 实际上比我们简单的 CUDA 实现 (cuda_gelu) 慢。")

    text("Triton operates on blocks, CUDA operates on threads. <br/> Triton 操作块，CUDA 操作线程。")
    text("Blocks allows Triton compiler to do other optimizations (e.g., thread coarsening). <br/> 块允许 Triton 编译器进行其他优化 (如线程粗化)。")

    text("Everything is way faster than the manual implementation (manual_gelu). <br/>一切都比手动实现 (manual_gelu) 快得多。")


def triton_gelu(x: torch.Tensor):
    assert x.is_cuda
    assert x.is_contiguous()

    # Allocate output tensor
    y = torch.empty_like(x)

    # Determine grid (elements divided into blocks)
    num_elements = x.numel()
    block_size = 1024  # Number of threads
    num_blocks = triton.cdiv(num_elements, block_size)

    triton_gelu_kernel[(num_blocks,)](x, y, num_elements, BLOCK_SIZE=block_size)

    return y


@triton.jit
def triton_gelu_kernel(x_ptr, y_ptr, num_elements, BLOCK_SIZE: tl.constexpr):
    # Input is at `x_ptr` and output is at `y_ptr`
    #     |        Block 0            |          Block 1          |      ...      |
    #                            BLOCK_SIZE                                 num_elements

    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE

    # Indices where this thread block should operate
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Handle boundary
    mask = offsets < num_elements

    # Read
    x = tl.load(x_ptr + offsets, mask=mask)

    # Approx gelu is 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    # Compute (tl.tanh doesn't exist, use tanh(a) = (exp(2a) - 1) / (exp(2a) + 1)
    a = 0.79788456 * (x + 0.044715 * x * x * x)
    exp = tl.exp(2 * a)
    tanh = (exp - 1) / (exp + 1)
    y = 0.5 * x * (1 + tanh)

    # Store
    tl.store(y_ptr + offsets, y, mask=mask)


def print_ptx_main():
    text("PTX (parallel thread execution) is like an assembly language for GPUs. <br/> PTX (并行线程执行) 就像是 GPU 的汇编语言。")

    text("We can see the PTX code generated by Triton. <br/> 我们可以看到 Triton 生成的 PTX 代码。")
    link("https://docs.nvidia.com/cuda/parallel-thread-execution/index.html")

    ptx = print_ptx("triton_gelu", triton_gelu_kernel)
    text(ptx, verbatim=True)

    text("Observations: <br/> 观察：")
    text("- ld.global.* and st.global.* reads and writes from global memory <br/> - ld.global.* 和 st.global.* 从全局内存读写")
    text("- %ctaid.x is block index, %tid.x is thread index <br/> - %ctaid.x 是块索引，%tid.x 是线程索引")
    text("- %f* are floating point registers, %r* are integer registers <br/> - %f* 是浮点寄存器，%r* 是整数寄存器")
    text("- One thread processes 8 elements at the same time (thread coarsening) <br/> - 一个线程同时处理 8 个元素 (线程粗化)")

    
def print_ptx(name: str, kernel):
    if os.environ.get("TRITON_INTERPRET") == "1":
        text("PTX is not generated when in interpret mode. <br/> 在解释模式下不会生成 PTX。")
        return

    """Print out the PTX code generated by Triton for the given `kernel`."""
    ptx_path = f"var/{name}-ptx.txt"
    text("Let's go poke around at the PTX code. <br/> 让我们来看看 PTX 代码。")
    link(get_local_url(ptx_path))

    with open(ptx_path, "w") as f:
        return list(kernel.cache[0].values())[0].asm["ptx"]

    


def pytorch_compilation():
    text("So far, we have seen three ways to write GeLU: <br/> 到目前为止，我们已经看到了三种编写 GeLU 的方式：")
    text("- Use the default PyTorch function <br/> - 使用默认的 PyTorch 函数")
    text("- Write it in Python <br/> - 用 Python 编写 "), link(manual_gelu)
    text("- Write it in CUDA <br/> - 用 CUDA 编写 "), link(create_cuda_gelu)
    text("- Write it in Triton <br/> - 用 Triton 编写 "), link(triton_gelu)

    text("- Write it in Python and compile it into Triton <br/> - 用 Python 编写并编译成 Triton")
    compiled_gelu = torch.compile(manual_gelu)

    text("Check correctness of our implementation. <br/> 检查我们实现的正确性。")
    check_equal(compiled_gelu, manual_gelu)

    if not torch.cuda.is_available():
        return

    text("Let's benchmark and profile it! <br/> 让我们对它进行基准测试和性能分析！")
    manual_time = benchmark("manual_gelu", run_operation1(dim=16384, operation=manual_gelu)) # @inspect manual_time
    pytorch_time = benchmark("pytorch_gelu", run_operation1(dim=16384, operation=pytorch_gelu)) # @inspect pytorch_time
    cuda_time = benchmark("cuda_gelu", run_operation1(dim=16384, operation=create_cuda_gelu())) # @inspect cuda_time
    triton_time = benchmark("triton_gelu", run_operation1(dim=16384, operation=triton_gelu)) # @inspect triton_time
    compiled_time = benchmark("compiled_gelu", run_operation1(dim=16384, operation=compiled_gelu)) # @inspect compiled_time

    text("Let's look under the hood <br/> 让我们看看底层")
    compiled_gelu_profile = profile("compiled_gelu", run_operation1(dim=16384, operation=compiled_gelu))
    text(f"## compiled_gelu")
    text(compiled_gelu_profile, verbatim=True)


def triton_softmax_main():
    text("So far, we've looked at elementwise operations in Triton (e.g., GeLU). <br/> 到目前为止，我们已经在 Triton 中看了逐元素操作 (如 GeLU)。")
    text("Now let us look at operations that aggregate over multiple values. <br/> 现在让我们看看聚合多个值的操作。")

    text("We will roughly follow the Triton fused softmax tutorial: <br/> 我们将大致遵循 Triton 融合 softmax 教程："), link("https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html")

    text("Recall the softmax operation is used in attention and generating probabilities. <br/> 回想一下，softmax 操作用于注意力和生成概率。")
    text("Normalize each row of a matrix: <br/> 归一化矩阵的每一行：")
    text("[A1 A2 A3]   =>   [A1/A A2/A A3/A]", verbatim=True)
    text("[B1 B2 B3]   =>   [B1/B B2/B B3/B]", verbatim=True)

    text("Let's first start with the naive implementation and keep track of reads/writes. <br/> 让我们先从简单实现开始，并跟踪读写次数。")
    x = torch.tensor([
        [5., 5, 5],
        [0, 0, 100],
    ], device=get_device())
    y1 = manual_softmax(x) # @inspect y1

    if not torch.cuda.is_available():
        return

    text("Now let us write the Triton kernel. <br/> 现在让我们编写 Triton 内核。")
    y2 = triton_softmax(x)
    assert torch.allclose(y1, y2)

    text("Check our implementations are correct. <br/> 检查我们的实现是否正确。")
    check_equal2(pytorch_softmax, manual_softmax)
    check_equal2(pytorch_softmax, triton_softmax)

    compiled_softmax = torch.compile(manual_softmax)

    text("Now let's benchmark everything. <br/> 现在让我们对所有进行基准测试。")
    manual_time = benchmark("manual_softmax", run_operation1(dim=16384, operation=manual_softmax)) # @inspect manual_time
    compiled_time = benchmark("compiled_softmax", run_operation1(dim=16384, operation=compiled_softmax)) # @inspect compiled_time
    pytorch_time = benchmark("pytorch_softmax", run_operation1(dim=16384, operation=pytorch_softmax)) # @inspect pytorch_time
    triton_time = benchmark("triton_softmax", run_operation1(dim=16384, operation=triton_softmax)) # @inspect triton_time

    text("Look under the hood using the profiler. <br/> 使用性能分析器看看底层。")
    manual_softmax_profile = profile("manual_softmax", run_operation1(dim=16384, operation=manual_softmax))
    text(f"## manual_softmax")
    text(manual_softmax_profile, verbatim=True)
    compiled_softmax_profile = profile("compiled_softmax", run_operation1(dim=16384, operation=compiled_softmax))
    text(f"## compiled_softmax")
    text(compiled_softmax_profile, verbatim=True)
    pytorch_softmax_profile = profile("pytorch_softmax", run_operation1(dim=16384, operation=pytorch_softmax))
    text(f"## pytorch_softmax")
    text(pytorch_softmax_profile, verbatim=True)
    triton_softmax_profile = profile("triton_softmax", run_operation1(dim=16384, operation=triton_softmax))
    text(f"## triton_softmax")
    text(triton_softmax_profile, verbatim=True)

    text("Let's end by looking at the PTX code. <br/> 让我们以查看 PTX 代码结束。")
    ptx = print_ptx("triton_softmax", triton_softmax_kernel)
    text(ptx, verbatim=True)


def manual_softmax(x: torch.Tensor):
    # M: number of rows, N: number of columns
    M, N = x.shape

    # Compute the max of each row (MN reads, M writes)
    x_max = x.max(dim=1)[0]

    # Subtract off the max (MN + M reads, MN writes)
    x = x - x_max[:, None]

    # Exponentiate (MN reads, MN writes)
    numerator = torch.exp(x)

    # Compute normalization constant (MN reads, M writes)
    denominator = numerator.sum(dim=1)

    # Normalize (MN reads, MN writes)
    y = numerator / denominator[:, None]

    # Total: 5MN + M reads, 3MN + 2M writes
    # In principle, should have MN reads, MN writes (speedup of 4x!)
    return y


def triton_softmax(x: torch.Tensor):
    # Allocate output tensor
    y = torch.empty_like(x)

    # Determine grid
    M, N = x.shape                          # Number of rows x number of columns
    block_size = triton.next_power_of_2(N)  # Each block contains all the columns
    num_blocks = M                          # Each block is a row

    # Launch kernel
    triton_softmax_kernel[(M,)](
        x_ptr=x, y_ptr=y,
        x_row_stride=x.stride(0), y_row_stride=y.stride(0),
        num_cols=N, BLOCK_SIZE=block_size
    )

    return y


@triton.jit
def triton_softmax_kernel(x_ptr, y_ptr, x_row_stride, y_row_stride, num_cols, BLOCK_SIZE: tl.constexpr):
    assert num_cols <= BLOCK_SIZE

    # Process each row independently
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)

    # Read from global memory
    x_start_ptr = x_ptr + row_idx * x_row_stride
    x_ptrs = x_start_ptr + col_offsets
    x_row = tl.load(x_ptrs, mask=col_offsets < num_cols, other=float("-inf"))

    # Compute
    x_row = x_row - tl.max(x_row, axis=0)
    numerator = tl.exp(x_row)
    denominator = tl.sum(numerator, axis=0)
    y_row = numerator / denominator

    # Write back to global memory
    y_start_ptr = y_ptr + row_idx * y_row_stride
    y_ptrs = y_start_ptr + col_offsets
    tl.store(y_ptrs, y_row, mask=col_offsets < num_cols)


def triton_matmul_main():
    text("Matrix multipliction is perhaps the most optimized algorithm ever. <br/> 矩阵乘法可能是最经过优化的算法。")

    text("If you write matrix multiplication in CUDA, there's all sorts of crazy things you have to do. <br/> 如果你用 CUDA 写矩阵乘法，有各种疯狂的事情要做。")
    link("https://github.com/openai/blocksparse/blob/master/src/matmul_op_gpu.cu")

    text("It's much easier in Triton. <br/> 在 Triton 中要容易得多。")
    link("https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html")

    text("       k                  j                     ", verbatim=True)
    text("  [ A1 A2 A3 ]       [ B1 B2 B3 ]   [ C1 C2 C3 ]", verbatim=True)
    text("i [ A4 A5 A6 ]  *  k [ B4 B5 B6 ] = [ C4 C5 C6 ]", verbatim=True)
    text("  [ A7 A8 A9 ]       [ B7 B8 B9 ]   [ C7 C8 C9 ]", verbatim=True)

    text("Naively: need MKN reads, MN writes <br/> 朴素地：需要 MKN 次读取，MN 次写入")

    text("Computing C4 and C5 both need A4, A5, A6. <br/> 计算 C4 和 C5 都需要 A4, A5, A6。")
    text("Can we read A4, A5, A6 from DRAM once to compute both? <br/> 我们能从 DRAM 读取一次 A4, A5, A6 来计算两者吗？")
    text("Answer: yes, using shared memory! <br/> 答案：是的，使用共享内存！")

    text("## Tiling (leveraging shared memory) <br/> ## 分块 (利用共享内存)")

    text("Recall that shared memory is: <br/> 回想一下共享内存是：")
    text("- fast (10x faster) and small(~100KB) <br/> - 快 (10倍快) 且小 (~100KB)")
    text("- shared between all the threads in a block. <br/> - 块内所有线程共享。")
    image("https://miro.medium.com/v2/resize:fit:2000/format:webp/1*6xoBKi5kL2dZpivFe1-zgw.jpeg")

    text("Trivial: for small matrices, load all of A and B into shared memory, then could compute C. <br/> 对于小矩阵，简单地将 A 和 B 全部加载到共享内存，然后可以计算 C。")
    text("Now we get MK + KN reads, MN writes <br/> 现在我们得到 MK + KN 次读取，MN 次写入")

    text("But what if we have big matrices... <br/> 但如果是大矩阵呢...")

    image("https://www.researchgate.net/profile/Axel-Huebl/publication/320499173/figure/fig1/AS:614298980196359@1523471698396/Performance-critical-A-B-part-of-the-GEMM-using-a-tiling-strategy-A-thread-iterates.png", width=0.5)
    text("Key idea: divide the matrix into blocks. <br/> 关键思路：将矩阵分成块。")
    text("For each block of A and block of B: <br/> 对于 A 的每个块和 B 的每个块：")
    text("- load into shared memory, <br/> - 加载到共享内存，")
    text("- do mini-matrix multiplication, <br/> - 做小矩阵乘法，")
    text("- write the partial sum. <br/> - 写入部分和。")

    text("Animation of tiled matrix multiplication <br/> 分块矩阵乘法动画 "), link("https://youtu.be/aMvCEEBIBto")

    text("## Leveraging L2 cache <br/> ## 利用 L2 缓存")

    text("Two ways of computing 9 elements of a matrix: <br/> 计算矩阵 9 个元素的两种方式：")
    image("https://triton-lang.org/main/_images/grouped_vs_row_major_ordering.png", width=0.5)
    text("1. Loads 9 + 81 = 90 blocks <br/> 1. 加载 9 + 81 = 90 个块")
    text("1. Loads 27 + 27 = 54 blocks <br/> 1. 加载 27 + 27 = 54 个块")

    text("Process the blocks in an order that minimizes the reads. <br/> 以最小化读取次数的顺序处理块。")

    text("Why write your own kernel for matrix multiplication (e.g., A @ B)? <br/> 为什么要为矩阵乘法 (如 A @ B) 编写自己的内核？")
    text("Answer: fusion with another operation (e.g., gelu(A @ B)) <br/> 答案：与另一个操作融合 (如 gelu(A @ B))")

    if not torch.cuda.is_available():
        return
    text("Let's try it! <br/> 让我们试试！")
    benchmark("pytorch_matmul", run_operation2(dim=16384, operation=torch.matmul))
    benchmark("triton_matmul", run_operation2(dim=16384, operation=triton_matmul))

    # Not working for some reason
    #print_ptx("triton_matmul", triton_matmul_kernel)


def further_reading():
    text("Horace He's blog post <br/> Horace He 的博客文章 "), link(title="[Article]", url="https://horace.io/brrr_intro.html")

    text("CUDA MODE Lecture 1: how to profile CUDA kernels in PyTorch <br/> CUDA MODE 讲座 1：如何在 PyTorch 中分析 CUDA 内核 "), link(title="[Video]", url="https://www.youtube.com/watch?v=LuhJEEJQgUM")
    text("CUDA MODE Lecture 2: Chapters 1-3 of PPMP book <br/> CUDA MODE 讲座 2：PPMP 书籍第 1-3 章 "), link(title="[Video]", url="https://www.youtube.com/watch?v=NQ-0D5Ti2dc")
    text("CUDA MODE Lecture 3: Getting started with CUDA for Python Programmers <br/> CUDA MODE 讲座 3：Python 程序员的 CUDA 入门 "), link(title="[Video]", url="https://www.youtube.com/watch?v=4sgKnKbR-WE")
    text("CUDA MODE Lecture 4: Compute and memory basics <br/> CUDA MODE 讲座 4：计算和内存基础 "), link(title="[Video]", url="https://www.youtube.com/watch?v=lTmYrKwjSOU")
    text("CUDA MODE Lecture 8: CUDA performance checklist <br/> CUDA MODE 讲座 8：CUDA 性能检查清单 "), link(title="[Video]", url="https://www.youtube.com/watch?v=SGhfUhlowB4")

    text("HetSys Course: Lecture 1: Programming heterogenous computing systems with GPUs <br/> HetSys 课程：讲座 1：使用 GPU 编程异构计算系统 "), link(title="[Video]", url="https://www.youtube.com/watch?v=8JGo2zylE80")
    text("HetSys Course: Lecture 2: SIMD processing and GPUs <br/> HetSys 课程：讲座 2：SIMD 处理和 GPU "), link(title="[Video]", url="https://www.youtube.com/watch?v=x1MA4MtO4Tc")
    text("HetSys Course: Lecture 3: GPU Software Hierarchy <br/> HetSys 课程：讲座 3：GPU 软件层次结构 "), link(title="[Video]", url="https://www.youtube.com/watch?v=KGZ00J5MJz0")
    text("HetSys Course: Lecture 4: GPU Memory Hierarchy <br/> HetSys 课程：讲座 4：GPU 内存层次结构 "), link(title="[Video]", url="https://www.youtube.com/watch?v=ZQKMZIP3Fzg")
    text("HetSys Course: Lecture 5: GPU performance considerations <br/> HetSys 课程：讲座 5：GPU 性能考虑 "), link(title="[Video]", url="https://www.youtube.com/watch?v=ODeprwr3Jho")

    link(title="[A100 GPU with NVIDIA Ampere Architecture]", url="https://jonathan-hui.medium.com/ai-chips-a100-gpu-with-nvidia-ampere-architecture-3034ed685e6e")
    link(title="[NVIDIA Deep Learning Performance Guide]", url="https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html")
    link(title="[GPU Puzzles]", url="https://github.com/srush/gpu-puzzles")
    link(title="[Triton Paper]", url="https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf")
    link(title="[PyTorch 2.0 Acceleration]", url="https://towardsdatascience.com/how-pytorch-2-0-accelerates-deep-learning-with-operator-fusion-and-cpu-gpu-code-generation-35132a85bd26")

############################################################

def print_gpu_specs():
    num_devices = torch.cuda.device_count()  # @inspect num_devices
    text(f"{num_devices} devices <br/> {num_devices} 个设备")
    for i in range(num_devices):
        properties = torch.cuda.get_device_properties(i)  # @inspect properties
        text(f"{i}: {properties}")


def pytorch_softmax(x: torch.Tensor):
    return torch.nn.functional.softmax(x, dim=-1)


def pytorch_gelu(x: torch.Tensor):
    # Use the tanh approximation to match our implementation
    return torch.nn.functional.gelu(x, approximate="tanh")


def manual_gelu(x: torch.Tensor):
    return 0.5 * x * (1 + torch.tanh(0.79788456 * (x + 0.044715 * x * x * x)))




if __name__ == "__main__":
    main()
