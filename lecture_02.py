from execute_util import text, link, image
from facts import a100_flop_per_sec, h100_flop_per_sec
import torch.nn.functional as F
import timeit
import torch
from typing import Iterable
from torch import nn
import numpy as np
from lecture_util import article_link
from jaxtyping import Float
from einops import rearrange, einsum, reduce
from references import zero_2019


def main():
    text("Last lecture: overview, tokenization <br/> 上节课：概述、分词")

    text("Overview of this lecture: <br/> 本节课概述：")
    text("- We will discuss all the **primitives** needed to train a model. <br/> - 我们将讨论训练模型所需的所有**基本构件**。")
    text("- We will go bottom-up from tensors to models to optimizers to the training loop. <br/> - 我们将从张量到模型再到优化器再到训练循环自底向上地学习。")
    text("- We will pay close attention to efficiency (use of **resources**). <br/> - 我们将密切关注效率（**资源**的使用）。")

    text("In particular, we will account for two types of resources: <br/> 特别是，我们将核算两类资源：")
    text("- Memory (GB) <br/> - 内存（GB）")
    text("- Compute (FLOPs) <br/> - 计算（FLOPs）")

    motivating_questions()

    text("We will not go over the Transformer. <br/> 我们不会详细讲解 Transformer。")
    text("There are excellent expositions: <br/> 有很好的参考资料：")
    link(title="Assignment 1 handout", url="https://github.com/stanford-cs336/assignment1-basics/blob/main/cs336_spring2025_assignment1_basics.pdf")
    link(title="Mathematical description", url="https://johnthickstun.com/docs/transformers.pdf")
    link(title="Illustrated Transformer", url="http://jalammar.github.io/illustrated-transformer/")
    link(title="Illustrated GPT-2", url="https://jalammar.github.io/illustrated-gpt2/")
    text("Instead, we'll work with simpler models. <br/> 相反，我们将使用更简单的模型。")

    text("What knowledge to take away: <br/> 要带走的知识：")
    text("- Mechanics: straightforward (just PyTorch) <br/> - 机制：直接（只是 PyTorch）")
    text("- Mindset: resource accounting (remember to do it) <br/> - 思维方式：资源核算（记得要做）")
    text("- Intuitions: broad strokes (no large models) <br/> - 直觉：大致了解（没有大模型）")

    text("## Memory accounting <br/> 内存核算")
    tensors_basics()
    tensors_memory()

    text("## Compute accounting <br/> 计算核算")
    tensors_on_gpus()
    tensor_operations()
    tensor_einops()
    tensor_operations_flops()
    gradients_basics()
    gradients_flops()

    text("## Models <br/> 模型")
    module_parameters()
    custom_model()

    text("Training loop and best practices <br/> 训练循环和最佳实践")
    note_about_randomness()
    data_loading()

    optimizer()
    train_loop()
    checkpointing()
    mixed_precision_training()


def motivating_questions():
    text("Let's do some napkin math. <br/> 让我们做一些粗略的计算。")

    text("**Question**: How long would it take to train a 70B parameter model on 15T tokens on 1024 H100s? <br/> **问题**：在 1024 个 H100 上用 15T token 训练一个 70B 参数的模型需要多长时间？")
    total_flops = 6 * 70e9 * 15e12  # @inspect total_flops
    assert h100_flop_per_sec == 1979e12 / 2
    mfu = 0.5
    flops_per_day = h100_flop_per_sec * mfu * 1024 * 60 * 60 * 24  # @inspect flops_per_day
    days = total_flops / flops_per_day  # @inspect days

    text("**Question**: What's the largest model that can you can train on 8 H100s using AdamW (naively)? <br/> **问题**：在 8 个 H100 上使用 AdamW（天真地）能训练的最大模型是多少？")
    h100_bytes = 80e9  # @inspect h100_bytes
    bytes_per_parameter = 4 + 4 + (4 + 4)  # parameters, gradients, optimizer state  @inspect bytes_per_parameter
    num_parameters = (h100_bytes * 8) / bytes_per_parameter  # @inspect num_parameters
    text("Caveat 1: we are naively using float32 for parameters and gradients.  We could also use bf16 for parameters and gradients (2 + 2) and keep an extra float32 copy of the parameters (4). This doesn't save memory, but is faster. <br/> 注意 1：我们天真地对参数和梯度使用 float32。我们也可以对参数和梯度使用 bf16（2 + 2）并保留一个额外的 float32 参数副本（4）。这不会节省内存，但更快。"), link(zero_2019)
    text("Caveat 2: activations are not accounted for (depends on batch size and sequence length). <br/> 注意 2：激活未计入（取决于批次大小和序列长度）。")

    text("This is a rough back-of-the-envelope calculation. <br/> 这是一个粗略的估算。")


def tensors_basics():
    text("Tensors are the basic building block for storing everything: parameters, gradients, optimizer state, data, activations. <br/> 张量是存储一切的基本构建块：参数、梯度、优化器状态、数据、激活。")
    link(title="[PyTorch docs on tensors]", url="https://pytorch.org/docs/stable/tensors.html")

    text("You can create tensors in multiple ways: <br/> 你可以用多种方式创建张量：")
    x = torch.tensor([[1., 2, 3], [4, 5, 6]])  # @inspect x
    x = torch.zeros(4, 8)  # 4x8 matrix of all zeros @inspect x
    x = torch.ones(4, 8)  # 4x8 matrix of all ones @inspect x
    x = torch.randn(4, 8)  # 4x8 matrix of iid Normal(0, 1) samples @inspect x

    text("Allocate but don't initialize the values: <br/> 分配但不初始化值：")
    x = torch.empty(4, 8)  # 4x8 matrix of uninitialized values @inspect x
    text("...because you want to use some custom logic to set the values later <br/> ...因为你想稍后使用一些自定义逻辑来设置值")
    nn.init.trunc_normal_(x, mean=0, std=1, a=-2, b=2)  # @inspect x


def tensors_memory():
    text("Almost everything (parameters, gradients, activations, optimizer states) are stored as floating point numbers. <br/> 几乎所有东西（参数、梯度、激活、优化器状态）都存储为浮点数。")

    text("## float32 <br/> float32")
    link(title="[Wikipedia]", url="https://en.wikipedia.org/wiki/Single-precision_floating-point_format")
    image("images/fp32.png", width=600)
    text("The float32 data type (also known as fp32 or single precision) is the default. <br/> float32 数据类型（也称为 fp32 或单精度）是默认值。")
    text("Traditionally, in scientific computing, float32 is the baseline; you could use double precision (float64) in some cases. <br/> 传统上，在科学计算中，float32 是基准；在某些情况下可以使用双精度（float64）。")
    text("In deep learning, you can be a lot sloppier. <br/> 在深度学习中，你可以更加随意。")

    text("Let's examine memory usage of these tensors. <br/> 让我们检查这些张量的内存使用情况。")
    text("Memory is determined by the (i) number of values and (ii) data type of each value. <br/> 内存由（i）值的数量和（ii）每个值的数据类型决定。")
    x = torch.zeros(4, 8)  # @inspect x
    assert x.dtype == torch.float32  # Default type
    assert x.numel() == 4 * 8
    assert x.element_size() == 4  # Float is 4 bytes
    assert get_memory_usage(x) == 4 * 8 * 4  # 128 bytes

    text("One matrix in the feedforward layer of GPT-3: <br/> GPT-3 前馈层中的一个矩阵：")
    assert get_memory_usage(torch.empty(12288 * 4, 12288)) == 2304 * 1024 * 1024  # 2.3 GB
    text("...which is a lot! <br/> ...这很多！")

    text("## float16 <br/> float16")
    link(title="[Wikipedia]", url="https://en.wikipedia.org/wiki/Half-precision_floating-point_format")
    image("images/fp16.png", width=400)
    text("The float16 data type (also known as fp16 or half precision) cuts down the memory. <br/> float16 数据类型（也称为 fp16 或半精度）可以减少内存。")
    x = torch.zeros(4, 8, dtype=torch.float16)  # @inspect x
    assert x.element_size() == 2
    text("However, the dynamic range (especially for small numbers) isn't great. <br/> 然而，动态范围（特别是对于小数）不太好。")
    x = torch.tensor([1e-8], dtype=torch.float16)  # @inspect x
    assert x == 0  # Underflow!
    text("If this happens when you train, you can get instability. <br/> 如果训练时发生这种情况，可能会导致不稳定。")

    text("## bfloat16 <br/> bfloat16")
    link(title="[Wikipedia]", url="https://en.wikipedia.org/wiki/Bfloat16_floating-point_format")
    image("images/bf16.png", width=400)
    text("Google Brain developed bfloat (brain floating point) in 2018 to address this issue. <br/> Google Brain 于 2018 年开发了 bfloat（脑浮点）来解决这个问题。")
    text("bfloat16 uses the same memory as float16 but has the same dynamic range as float32! <br/> bfloat16 使用与 float16 相同的内存，但具有与 float32 相同的动态范围！")
    text("The only catch is that the resolution is worse, but this matters less for deep learning. <br/> 唯一的问题是分辨率更差，但这对深度学习来说不太重要。")
    x = torch.tensor([1e-8], dtype=torch.bfloat16)  # @inspect x
    assert x != 0  # No underflow!

    text("Let's compare the dynamic ranges and memory usage of the different data types: <br/> 让我们比较不同数据类型的动态范围和内存使用情况：")
    float32_info = torch.finfo(torch.float32)  # @inspect float32_info
    float16_info = torch.finfo(torch.float16)  # @inspect float16_info
    bfloat16_info = torch.finfo(torch.bfloat16)  # @inspect bfloat16_info

    text("## fp8 <br/> fp8")
    text("In 2022, FP8 was standardized, motivated by machine learning workloads. <br/> 2022 年，FP8 被标准化，由机器学习工作负载推动。")
    link("https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html")
    image("https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/_images/fp8_formats.png", width=400)
    text("H100s support two variants of FP8: E4M3 (range [-448, 448]) and E5M2 ([-57344, 57344]). <br/> H100 支持两种 FP8 变体：E4M3（范围 [-448, 448]）和 E5M2（[-57344, 57344]）。")
    text("Reference: <br/> 参考："), link("https://arxiv.org/pdf/2209.05433.pdf")

    text("Implications on training: <br/> 对训练的影响：")
    text("- Training with float32 works, but requires lots of memory. <br/> - 使用 float32 训练可行，但需要大量内存。")
    text("- Training with fp8, float16 and even bfloat16 is risky, and you can get instability. <br/> - 使用 fp8、float16 甚至 bfloat16 训练有风险，可能会导致不稳定。")
    text("- Solution (later): use mixed precision training, see <br/> - 解决方案（稍后）：使用混合精度训练，见 "), link(mixed_precision_training)


def tensors_on_gpus():
    text("By default, tensors are stored in CPU memory. <br/> 默认情况下，张量存储在 CPU 内存中。")
    x = torch.zeros(32, 32)
    assert x.device == torch.device("cpu")

    text("However, in order to take advantage of the massive parallelism of GPUs, we need to move them to GPU memory. <br/> 然而，为了利用 GPU 的大规模并行性，我们需要将它们移动到 GPU 内存中。")
    image("images/cpu-gpu.png", width=400)

    text("Let's first see if we have any GPUs. <br/> 让我们先看看是否有 GPU。")
    if not torch.cuda.is_available():
        return

    num_gpus = torch.cuda.device_count()  # @inspect num_gpus
    for i in range(num_gpus):
        properties = torch.cuda.get_device_properties(i)  # @inspect properties

    memory_allocated = torch.cuda.memory_allocated()  # @inspect memory_allocated

    text("Move the tensor to GPU memory (device 0). <br/> 将张量移动到 GPU 内存（设备 0）。")
    y = x.to("cuda:0")
    assert y.device == torch.device("cuda", 0)

    text("Or create a tensor directly on the GPU: <br/> 或者直接在 GPU 上创建张量：")
    z = torch.zeros(32, 32, device="cuda:0")

    new_memory_allocated = torch.cuda.memory_allocated()  # @inspect new_memory_allocated
    memory_used = new_memory_allocated - memory_allocated  # @inspect memory_used
    assert memory_used == 2 * (32 * 32 * 4)  # 2 32x32 matrices of 4-byte floats



def tensor_operations():
    text("Most tensors are created from performing operations on other tensors. <br/> 大多数张量是通过对其他张量执行操作来创建的。")
    text("Each operation has some memory and compute consequence. <br/> 每个操作都有一些内存和计算的影响。")

    tensor_storage()
    tensor_slicing()
    tensor_elementwise()
    tensor_matmul()


def tensor_storage():
    text("What are tensors in PyTorch? <br/> PyTorch 中的张量是什么？")
    text("PyTorch tensors are pointers into allocated memory <br/> PyTorch 张量是指向已分配内存的指针")
    text("...with metadata describing how to get to any element of the tensor. <br/> ...带有描述如何访问张量中任何元素的元数据。")
    image("https://martinlwx.github.io/img/2D_tensor_strides.png", width=400)
    link(title="[PyTorch docs]", url="https://pytorch.org/docs/stable/generated/torch.Tensor.stride.html")
    x = torch.tensor([
        [0., 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11],
        [12, 13, 14, 15],
    ])

    text("To go to the next row (dim 0), skip 4 elements in storage. <br/> 要转到下一行（dim 0），在存储中跳过 4 个元素。")
    assert x.stride(0) == 4

    text("To go to the next column (dim 1), skip 1 element in storage. <br/> 要转到下一列（dim 1），在存储中跳过 1 个元素。")
    assert x.stride(1) == 1

    text("To find an element: <br/> 要找到一个元素：")
    r, c = 1, 2
    index = r * x.stride(0) + c * x.stride(1)  # @inspect index
    assert index == 6


def tensor_slicing():
    x = torch.tensor([[1., 2, 3], [4, 5, 6]])  # @inspect x

    text("Many operations simply provide a different **view** of the tensor. <br/> 许多操作只是提供张量的不同**视图**。")
    text("This does not make a copy, and therefore mutations in one tensor affects the other. <br/> 这不会创建副本，因此对一个张量的修改会影响另一个。")

    text("Get row 0: <br/> 获取第 0 行：")
    y = x[0]  # @inspect y
    assert torch.equal(y, torch.tensor([1., 2, 3]))
    assert same_storage(x, y)

    text("Get column 1: <br/> 获取第 1 列：")
    y = x[:, 1]  # @inspect y
    assert torch.equal(y, torch.tensor([2, 5]))
    assert same_storage(x, y)

    text("View 2x3 matrix as 3x2 matrix: <br/> 将 2x3 矩阵视为 3x2 矩阵：")
    y = x.view(3, 2)  # @inspect y
    assert torch.equal(y, torch.tensor([[1, 2], [3, 4], [5, 6]]))
    assert same_storage(x, y)

    text("Transpose the matrix: <br/> 转置矩阵：")
    y = x.transpose(1, 0)  # @inspect y
    assert torch.equal(y, torch.tensor([[1, 4], [2, 5], [3, 6]]))
    assert same_storage(x, y)

    text("Check that mutating x also mutates y. <br/> 检查修改 x 也会修改 y。")
    x[0][0] = 100  # @inspect x, @inspect y
    assert y[0][0] == 100

    text("Note that some views are non-contiguous entries, which means that further views aren't possible. <br/> 注意，一些视图是非连续条目，这意味着无法进一步创建视图。")
    x = torch.tensor([[1., 2, 3], [4, 5, 6]])  # @inspect x
    y = x.transpose(1, 0)  # @inspect y
    assert not y.is_contiguous()
    try:
        y.view(2, 3)
        assert False
    except RuntimeError as e:
        assert "view size is not compatible with input tensor's size and stride" in str(e)

    text("One can enforce a tensor to be contiguous first: <br/> 可以先强制张量连续：")
    y = x.transpose(1, 0).contiguous().view(2, 3)  # @inspect y
    assert not same_storage(x, y)
    text("Views are free, copying take both (additional) memory and compute. <br/> 视图是免费的，复制需要（额外的）内存和计算。")


def tensor_elementwise():
    text("These operations apply some operation to each element of the tensor <br/> 这些操作对张量的每个元素应用某些操作")
    text("...and return a (new) tensor of the same shape. <br/> ...并返回相同形状的（新）张量。")

    x = torch.tensor([1, 4, 9])
    assert torch.equal(x.pow(2), torch.tensor([1, 16, 81]))
    assert torch.equal(x.sqrt(), torch.tensor([1, 2, 3]))
    assert torch.equal(x.rsqrt(), torch.tensor([1, 1 / 2, 1 / 3]))  # i -> 1/sqrt(x_i)

    assert torch.equal(x + x, torch.tensor([2, 8, 18]))
    assert torch.equal(x * 2, torch.tensor([2, 8, 18]))
    assert torch.equal(x / 0.5, torch.tensor([2, 8, 18]))

    text("`triu` takes the upper triangular part of a matrix. <br/> `triu` 取矩阵的上三角部分。")
    x = torch.ones(3, 3).triu()  # @inspect x
    assert torch.equal(x, torch.tensor([
        [1, 1, 1],
        [0, 1, 1],
        [0, 0, 1]],
    ))
    text("This is useful for computing an causal attention mask, where M[i, j] is the contribution of i to j. <br/> 这对于计算因果注意力掩码很有用，其中 M[i, j] 是 i 对 j 的贡献。")


def tensor_matmul():
    text("Finally, the bread and butter of deep learning: matrix multiplication. <br/> 最后，深度学习的基本功：矩阵乘法。")
    x = torch.ones(16, 32)
    w = torch.ones(32, 2)
    y = x @ w
    assert y.size() == torch.Size([16, 2])

    text("In general, we perform operations for every example in a batch and token in a sequence. <br/> 一般来说，我们对批次中的每个样本和序列中的每个 token 执行操作。")
    image("images/batch-sequence.png", width=400)
    x = torch.ones(4, 8, 16, 32)
    w = torch.ones(32, 2)
    y = x @ w
    assert y.size() == torch.Size([4, 8, 16, 2])
    text("In this case, we iterate over values of the first 2 dimensions of `x` and multiply by `w`. <br/> 在这种情况下，我们遍历 `x` 的前 2 个维度的值并乘以 `w`。")


def tensor_einops():
    einops_motivation()

    text("Einops is a library for manipulating tensors where dimensions are named. <br/> Einops 是一个用于操作张量的库，其中维度是命名的。")
    text("It is inspired by Einstein summation notation (Einstein, 1916). <br/> 它受到爱因斯坦求和记号（Einstein, 1916）的启发。")
    link(title="[Einops tutorial]", url="https://einops.rocks/1-einops-basics/")

    jaxtyping_basics()
    einops_einsum()
    einops_reduce()
    einops_rearrange()
    

def einops_motivation():
    text("Traditional PyTorch code: <br/> 传统的 PyTorch 代码：")
    x = torch.ones(2, 2, 3)  # batch, sequence, hidden  @inspect x
    y = torch.ones(2, 2, 3)  # batch, sequence, hidden  @inspect y
    z = x @ y.transpose(-2, -1)  # batch, sequence, sequence  @inspect z
    text("Easy to mess up the dimensions (what is -2, -1?)... <br/> 很容易搞混维度（-2, -1 是什么？）...")


def jaxtyping_basics():
    text("How do you keep track of tensor dimensions? <br/> 如何跟踪张量维度?")

    text("Old way: <br/> 旧方法:")
    x = torch.ones(2, 2, 1, 3)  # batch seq heads hidden  @inspect x

    text("New (jaxtyping) way: <br/> 新(jaxtyping)方法:")
    x: Float[torch.Tensor, "batch seq heads hidden"] = torch.ones(2, 2, 1, 3)  # @inspect x
    text("Note: this is just documentation (no enforcement). <br/> 注意: 这只是文档(没有强制执行)。")


def einops_einsum():
    text("Einsum is generalized matrix multiplication with good bookkeeping. <br/> Einsum 是具有良好记录的广义矩阵乘法。")

    text("Define two tensors: <br/> 定义两个张量：")
    x: Float[torch.Tensor, "batch seq1 hidden"] = torch.ones(2, 3, 4)  # @inspect x
    y: Float[torch.Tensor, "batch seq2 hidden"] = torch.ones(2, 3, 4)  # @inspect y

    text("Old way: <br/> 旧方法：")
    z = x @ y.transpose(-2, -1)  # batch, sequence, sequence  @inspect z

    text("New (einops) way: <br/> 新（einops）方法：")
    z = einsum(x, y, "batch seq1 hidden, batch seq2 hidden -> batch seq1 seq2")  # @inspect z
    text("Dimensions that are not named in the output are summed over. <br/> 输出中未命名的维度会被求和。")

    text("Or can use `...` to represent broadcasting over any number of dimensions: <br/> 或者可以使用 `...` 表示任意数量维度的广播：")
    z = einsum(x, y, "... seq1 hidden, ... seq2 hidden -> ... seq1 seq2")  # @inspect z


def einops_reduce():
    text("You can reduce a single tensor via some operation (e.g., sum, mean, max, min). <br/> 你可以通过某些操作（如 sum、mean、max、min）对单个张量进行归约。")
    x: Float[torch.Tensor, "batch seq hidden"] = torch.ones(2, 3, 4)  # @inspect x

    text("Old way: <br/> 旧方法：")
    y = x.sum(dim=-1)  # @inspect y

    text("New (einops) way: <br/> 新（einops）方法：")
    y = reduce(x, "... hidden -> ...", "sum")  # @inspect y


def einops_rearrange():
    text("Sometimes, a dimension represents two dimensions <br/> 有时，一个维度表示两个维度")
    text("...and you want to operate on one of them. <br/> ...而你想对其中一个进行操作。")

    x: Float[torch.Tensor, "batch seq total_hidden"] = torch.ones(2, 3, 8)  # @inspect x
    text("...where `total_hidden` is a flattened representation of `heads * hidden1` <br/> ...其中 `total_hidden` 是 `heads * hidden1` 的扁平化表示")
    w: Float[torch.Tensor, "hidden1 hidden2"] = torch.ones(4, 4)

    text("Break up `total_hidden` into two dimensions (`heads` and `hidden1`): <br/> 将 `total_hidden` 分解为两个维度（`heads` 和 `hidden1`）：")
    x = rearrange(x, "... (heads hidden1) -> ... heads hidden1", heads=2)  # @inspect x

    text("Perform the transformation by `w`: <br/> 通过 `w` 执行变换：")
    x = einsum(x, w, "... hidden1, hidden1 hidden2 -> ... hidden2")  # @inspect x

    text("Combine `heads` and `hidden2` back together: <br/> 将 `heads` 和 `hidden2` 合并回去：")
    x = rearrange(x, "... heads hidden2 -> ... (heads hidden2)")  # @inspect x


def tensor_operations_flops():
    text("Having gone through all the operations, let us examine their computational cost. <br/> 学完所有操作后，让我们来检查它们的计算成本。")

    text("A floating-point operation (FLOP) is a basic operation like addition (x + y) or multiplication (x y). <br/> 浮点运算 (FLOP) 是加法 (x + y) 或乘法 (x y) 等基本操作。")

    text("Two terribly confusing acronyms (pronounced the same!): <br/> 两个容易混淆的缩写词 (发音相同!):")
    text("- FLOPs: floating-point operations (measure of computation done) <br/> - FLOPs: 浮点运算次数 (已完成计算的度量)")
    text("- FLOP/s: floating-point operations per second (also written as FLOPS), which is used to measure the speed of hardware. <br/> - FLOP/s: 每秒浮点运算次数 (也写作 FLOPS)，用于衡量硬件速度。")

    text("## Intuitions <br/> 直觉")
    text("Training GPT-3 (2020) took 3.14e23 FLOPs. <br/> 训练 GPT-3 (2020) 花费了 3.14e23 FLOPs。"), article_link("https://lambdalabs.com/blog/demystifying-gpt-3")
    text("Training GPT-4 (2023) is speculated to take 2e25 FLOPs <br/> 推测训练 GPT-4 (2023) 花费 2e25 FLOPs "), article_link("https://patmcguinness.substack.com/p/gpt-4-details-revealed")
    text("US executive order: any foundation model trained with >= 1e26 FLOPs must be reported to the government (revoked in 2025) <br/> 美国行政命令: 任何使用 >= 1e26 FLOPs 训练的基础模型必须向政府报告 (2025 年撤销)")

    text("A100 has a peak performance of 312 teraFLOP/s <br/> A100 的峰值性能为 312 teraFLOP/s "), link(title="[spec]", url="https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf")
    assert a100_flop_per_sec == 312e12

    text("H100 has a peak performance of 1979 teraFLOP/s with sparsity, 50% without <br/> H100 的峰值性能为 1979 teraFLOP/s (稀疏)，无稀疏时为 50% "), link(title="[spec]", url="https://resources.nvidia.com/en-us-tensor-core/nvidia-tensor-core-gpu-datasheet")
    assert h100_flop_per_sec == 1979e12 / 2

    text("8 H100s for 2 weeks: <br/> 8 个 H100 运行 2 周:")
    total_flops = 8 * (60 * 60 * 24 * 7) * h100_flop_per_sec  # @inspect total_flops

    text("## Linear model <br/> 线性模型")
    text("As motivation, suppose you have a linear model. <br/> 作为动机，假设你有一个线性模型。")
    text("- We have n points <br/> - 我们有 n 个点")
    text("- Each point is d-dimsional <br/> - 每个点是 d 维的")
    text("- The linear model maps each d-dimensional vector to a k outputs <br/> - 线性模型将每个 d 维向量映射到 k 个输出")

    if torch.cuda.is_available():
        B = 16384  # Number of points
        D = 32768  # Dimension
        K = 8192   # Number of outputs
    else:
        B = 1024
        D = 256
        K = 64

    device = get_device()
    x = torch.ones(B, D, device=device)
    w = torch.randn(D, K, device=device)
    y = x @ w
    text("We have one multiplication (x[i][j] * w[j][k]) and one addition per (i, j, k) triple. <br/> 我们对每个 (i, j, k) 三元组有一次乘法 (x[i][j] * w[j][k]) 和一次加法。")
    actual_num_flops = 2 * B * D * K  # @inspect actual_num_flops

    text("## FLOPs of other operations <br/> 其他操作的 FLOPs")
    text("- Elementwise operation on a m x n matrix requires O(m n) FLOPs. <br/> - m x n 矩阵的逐元素操作需要 O(m n) FLOPs。")
    text("- Addition of two m x n matrices requires m n FLOPs. <br/> - 两个 m x n 矩阵的加法需要 m n FLOPs。")
    text("In general, no other operation that you'd encounter in deep learning is as expensive as matrix multiplication for large enough matrices. <br/> 一般来说，对于足够大的矩阵，你在深度学习中遇到的其他操作都没有矩阵乘法昂贵。")

    text("Interpretation: <br/> 解释:")
    text("- B is the number of data points <br/> - B 是数据点的数量")
    text("- (D K) is the number of parameters <br/> - (D K) 是参数的数量")
    text("- FLOPs for forward pass is 2 (# tokens) (# parameters) <br/> - 前向传播的 FLOPs 是 2 (# tokens) (# parameters)")
    text("It turns out this generalizes to Transformers (to a first-order approximation). <br/> 事实证明这可以推广到 Transformer (一阶近似)。")

    text("How do our FLOPs calculations translate to wall-clock time (seconds)? <br/> 我们的 FLOPs 计算如何转换为实际时间 (秒)?")
    text("Let us time it! <br/> 让我们计时!")
    actual_time = time_matmul(x, w)  # @inspect actual_time
    actual_flop_per_sec = actual_num_flops / actual_time  # @inspect actual_flop_per_sec

    text("Each GPU has a specification sheet that reports the peak performance. <br/> 每个 GPU 都有一个规格表报告峰值性能。")
    text("- A100 <br/> - A100 "), link(title="[spec]", url="https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf")
    text("- H100 <br/> - H100 "), link(title="[spec]", url="https://resources.nvidia.com/en-us-tensor-core/nvidia-tensor-core-gpu-datasheet")
    text("Note that the FLOP/s depends heavily on the data type! <br/> 注意 FLOP/s 在很大程度上取决于数据类型!")
    promised_flop_per_sec = get_promised_flop_per_sec(device, x.dtype)  # @inspect promised_flop_per_sec

    text("## Model FLOPs utilization (MFU) <br/> 模型 FLOPs 利用率 (MFU)")

    text("Definition: (actual FLOP/s) / (promised FLOP/s) [ignore communication/overhead] <br/> 定义: (实际 FLOP/s) / (承诺 FLOP/s) [忽略通信/开销]")
    mfu = actual_flop_per_sec / promised_flop_per_sec  # @inspect mfu
    text("Usually, MFU of >= 0.5 is quite good (and will be higher if matmuls dominate) <br/> 通常，MFU >= 0.5 相当不错 (如果矩阵乘法占主导地位会更高)")

    text("Let's do it with bfloat16: <br/> 让我们用 bfloat16 试试:")
    x = x.to(torch.bfloat16)
    w = w.to(torch.bfloat16)
    bf16_actual_time = time_matmul(x, w)  # @inspect bf16_actual_time
    bf16_actual_flop_per_sec = actual_num_flops / bf16_actual_time  # @inspect bf16_actual_flop_per_sec
    bf16_promised_flop_per_sec = get_promised_flop_per_sec(device, x.dtype)  # @inspect bf16_promised_flop_per_sec
    bf16_mfu = bf16_actual_flop_per_sec / bf16_promised_flop_per_sec  # @inspect bf16_mfu
    text("Note: comparing bfloat16 to float32, the actual FLOP/s is higher. <br/> 注意: 与 float32 相比，bfloat16 的实际 FLOP/s 更高。")
    text("The MFU here is rather low, probably because the promised FLOPs is a bit optimistic. <br/> 这里的 MFU 相当低，可能是因为承诺的 FLOPs 有点乐观。")

    text("## Summary <br/> 总结")
    text("- Matrix multiplications dominate: (2 m n p) FLOPs <br/> - 矩阵乘法占主导: (2 m n p) FLOPs")
    text("- FLOP/s depends on hardware (H100 >> A100) and data type (bfloat16 >> float32) <br/> - FLOP/s 取决于硬件 (H100 >> A100) 和数据类型 (bfloat16 >> float32)")
    text("- Model FLOPs utilization (MFU): (actual FLOP/s) / (promised FLOP/s) <br/> - 模型 FLOPs 利用率 (MFU): (实际 FLOP/s) / (承诺 FLOP/s)")


def gradients_basics():
    text("So far, we've constructed tensors (which correspond to either parameters or data) and passed them through operations (forward). <br/> 到目前为止，我们构造了张量(对应参数或数据)并通过操作传递它们(前向)。")
    text("Now, we're going to compute the gradient (backward). <br/> 现在，我们要计算梯度(反向)。")

    text("As a simple example, let's consider the simple linear model: <br/> 作为一个简单的例子，让我们考虑简单的线性模型:")
    text("y = 0.5 (x * w - 5)^2 <br/> y = 0.5 (x * w - 5)^2")

    text("Forward pass: compute loss <br/> 前向传播: 计算损失")
    x = torch.tensor([1., 2, 3])
    w = torch.tensor([1., 1, 1], requires_grad=True)  # Want gradient
    pred_y = x @ w
    loss = 0.5 * (pred_y - 5).pow(2)

    text("Backward pass: compute gradients <br/> 反向传播: 计算梯度")
    loss.backward()
    assert loss.grad is None
    assert pred_y.grad is None
    assert x.grad is None
    assert torch.equal(w.grad, torch.tensor([1, 2, 3]))


def gradients_flops():
    text("Let us do count the FLOPs for computing gradients. <br/> 让我们来统计计算梯度的 FLOPs。")

    text("Revisit our linear model <br/> 重新审视我们的线性模型")
    if torch.cuda.is_available():
        B = 16384  # Number of points
        D = 32768  # Dimension
        K = 8192   # Number of outputs
    else:
        B = 1024
        D = 256
        K = 64

    device = get_device()
    x = torch.ones(B, D, device=device)
    w1 = torch.randn(D, D, device=device, requires_grad=True)
    w2 = torch.randn(D, K, device=device, requires_grad=True)

    text("Model: x --w1--> h1 --w2--> h2 -> loss <br/> 模型: x --w1--> h1 --w2--> h2 -> loss")
    h1 = x @ w1
    h2 = h1 @ w2
    loss = h2.pow(2).mean()

    text("Recall the number of forward FLOPs: <br/> 回忆前向 FLOPs 的数量: "), link(tensor_operations_flops)
    text("- Multiply x[i][j] * w1[j][k] <br/> - 乘法 x[i][j] * w1[j][k]")
    text("- Add to h1[i][k] <br/> - 加到 h1[i][k]")
    text("- Multiply h1[i][j] * w2[j][k] <br/> - 乘法 h1[i][j] * w2[j][k]")
    text("- Add to h2[i][k] <br/> - 加到 h2[i][k]")
    num_forward_flops = (2 * B * D * D) + (2 * B * D * K)  # @inspect num_forward_flops

    text("How many FLOPs is running the backward pass? <br/> 运行反向传播需要多少 FLOPs?")
    h1.retain_grad()  # For debugging
    h2.retain_grad()  # For debugging
    loss.backward()

    text("Recall model: x --w1--> h1 --w2--> h2 -> loss <br/> 回忆模型: x --w1--> h1 --w2--> h2 -> loss")

    text("- h1.grad = d loss / d h1 <br/> - h1.grad = d loss / d h1")
    text("- h2.grad = d loss / d h2 <br/> - h2.grad = d loss / d h2")
    text("- w1.grad = d loss / d w1 <br/> - w1.grad = d loss / d w1")
    text("- w2.grad = d loss / d w2 <br/> - w2.grad = d loss / d w2")

    text("Focus on the parameter w2. <br/> 关注参数 w2。")
    text("Invoke the chain rule. <br/> 调用链式法则。")

    num_backward_flops = 0  # @inspect num_backward_flops

    text("w2.grad[j,k] = sum_i h1[i,j] * h2.grad[i,k] <br/> w2.grad[j,k] = sum_i h1[i,j] * h2.grad[i,k]")
    assert w2.grad.size() == torch.Size([D, K])
    assert h1.size() == torch.Size([B, D])
    assert h2.grad.size() == torch.Size([B, K])
    text("For each (i, j, k), multiply and add. <br/> 对于每个 (i, j, k)，乘法和加法。")
    num_backward_flops += 2 * B * D * K  # @inspect num_backward_flops

    text("h1.grad[i,j] = sum_k w2[j,k] * h2.grad[i,k] <br/> h1.grad[i,j] = sum_k w2[j,k] * h2.grad[i,k]")
    assert h1.grad.size() == torch.Size([B, D])
    assert w2.size() == torch.Size([D, K])
    assert h2.grad.size() == torch.Size([B, K])
    text("For each (i, j, k), multiply and add. <br/> 对于每个 (i, j, k)，乘法和加法。")
    num_backward_flops += 2 * B * D * K  # @inspect num_backward_flops

    text("This was for just w2 (D*K parameters). <br/> 这只是 w2 (D*K 参数)。")
    text("Can do it for w1 (D*D parameters) as well (though don't need x.grad). <br/> 也可以对 w1 (D*D 参数) 做同样的事情 (尽管不需要 x.grad)。")
    num_backward_flops += (2 + 2) * B * D * D  # @inspect num_backward_flops

    text("A nice graphical visualization: <br/> 一个很好的图形化可视化: "), article_link("https://medium.com/@dzmitrybahdanau/the-flops-calculus-of-language-model-training-3b19c1f025e4")
    image("https://miro.medium.com/v2/resize:fit:1400/format:webp/1*VC9y_dHhCKFPXj90Qshj3w.gif", width=500)

    text("Putting it togther: <br/> 综合起来:")
    text("- Forward pass: 2 (# data points) (# parameters) FLOPs <br/> - 前向传播: 2 (# 数据点) (# 参数) FLOPs")
    text("- Backward pass: 4 (# data points) (# parameters) FLOPs <br/> - 反向传播: 4 (# 数据点) (# 参数) FLOPs")
    text("- Total: 6 (# data points) (# parameters) FLOPs <br/> - 总计: 6 (# 数据点) (# 参数) FLOPs")


def module_parameters():
    input_dim = 16384
    output_dim = 32

    text("Model parameters are stored in PyTorch as `nn.Parameter` objects. <br/> 模型参数在 PyTorch 中存储为 `nn.Parameter` 对象。")
    w = nn.Parameter(torch.randn(input_dim, output_dim))
    assert isinstance(w, torch.Tensor)  # Behaves like a tensor
    assert type(w.data) == torch.Tensor  # Access the underlying tensor

    text("## Parameter initialization <br/> 参数初始化")

    text("Let's see what happens. <br/> 让我们看看会发生什么。")
    x = nn.Parameter(torch.randn(input_dim))
    output = x @ w  # @inspect output
    assert output.size() == torch.Size([output_dim])
    text(f"Note that each element of `output` scales as sqrt(input_dim): {output[0]}. <br/> 注意 `output` 的每个元素按 sqrt(input_dim) 缩放: {output[0]}。")
    text("Large values can cause gradients to blow up and cause training to be unstable. <br/> 较大的值可能导致梯度爆炸，使训练不稳定。")

    text("We want an initialization that is invariant to `input_dim`. <br/> 我们想要一个对 `input_dim` 不变的初始化。")
    text("To do that, we simply rescale by 1/sqrt(input_dim) <br/> 为此，我们简单地按 1/sqrt(input_dim) 重新缩放")
    w = nn.Parameter(torch.randn(input_dim, output_dim) / np.sqrt(input_dim))
    output = x @ w  # @inspect output
    text(f"Now each element of `output` is constant: {output[0]}. <br/> 现在 `output` 的每个元素是常数: {output[0]}。")

    text("Up to a constant, this is Xavier initialization. <br/> 除了常数因子，这就是 Xavier 初始化。"), link(title="[paper]", url="https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf"), link(title="[stackexchange]", url="https://ai.stackexchange.com/questions/30491/is-there-a-proper-initialization-technique-for-the-weight-matrices-in-multi-head")

    text("To be extra safe, we truncate the normal distribution to [-3, 3] to avoid any chance of outliers. <br/> 为了更安全，我们将正态分布截断到 [-3, 3] 以避免任何异常值的可能。")
    w = nn.Parameter(nn.init.trunc_normal_(torch.empty(input_dim, output_dim), std=1 / np.sqrt(input_dim), a=-3, b=3))


def custom_model():
    text("Let's build up a simple deep linear model using `nn.Parameter`. <br/> 让我们使用 `nn.Parameter` 构建一个简单的深度线性模型。")

    D = 64  # Dimension
    num_layers = 2
    model = Cruncher(dim=D, num_layers=num_layers)

    param_sizes = [
        (name, param.numel())
        for name, param in model.state_dict().items()
    ]
    assert param_sizes == [
        ("layers.0.weight", D * D),
        ("layers.1.weight", D * D),
        ("final.weight", D),
    ]
    num_parameters = get_num_parameters(model)
    assert num_parameters == (D * D) + (D * D) + D

    text("Remember to move the model to the GPU. <br/> 记得将模型移动到 GPU。")
    device = get_device()
    model = model.to(device)

    text("Run the model on some data. <br/> 在一些数据上运行模型。")
    B = 8  # Batch size
    x = torch.randn(B, D, device=device)
    y = model(x)
    assert y.size() == torch.Size([B])


class Linear(nn.Module):
    """Simple linear layer."""
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(input_dim, output_dim) / np.sqrt(input_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight


class Cruncher(nn.Module):
    def __init__(self, dim: int, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([
            Linear(dim, dim)
            for i in range(num_layers)
        ])
        self.final = Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply linear layers
        B, D = x.size()
        for layer in self.layers:
            x = layer(x)

        # Apply final head
        x = self.final(x)
        assert x.size() == torch.Size([B, 1])

        # Remove the last dimension
        x = x.squeeze(-1)
        assert x.size() == torch.Size([B])

        return x


def get_batch(data: np.array, batch_size: int, sequence_length: int, device: str) -> torch.Tensor:
    text("Sample `batch_size` random positions into `data`. <br/> 从 `data` 中采样 `batch_size` 个随机位置。")
    start_indices = torch.randint(len(data) - sequence_length, (batch_size,))
    assert start_indices.size() == torch.Size([batch_size])

    text("Index into the data. <br/> 索引数据。")
    x = torch.tensor([data[start:start + sequence_length] for start in start_indices])
    assert x.size() == torch.Size([batch_size, sequence_length])

    text("## Pinned memory <br/> 锁页内存")

    text("By default, CPU tensors are in paged memory. We can explicitly pin. <br/> 默认情况下，CPU 张量在分页内存中。我们可以显式锁页。")
    if torch.cuda.is_available():
        x = x.pin_memory()

    text("This allows us to copy `x` from CPU into GPU asynchronously. <br/> 这允许我们异步地将 `x` 从 CPU 复制到 GPU。")
    x = x.to(device, non_blocking=True)

    text("This allows us to do two things in parallel (not done here): <br/> 这允许我们并行做两件事 (这里没有做):")
    text("- Fetch the next batch of data into CPU <br/> - 将下一批数据获取到 CPU")
    text("- Process `x` on the GPU. <br/> - 在 GPU 上处理 `x`。")

    article_link("https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/")
    article_link("https://gist.github.com/ZijiaLewisLu/eabdca955110833c0ce984d34eb7ff39?permalink_comment_id=3417135")

    return x


def note_about_randomness():
    text("Randomness shows up in many places: parameter initialization, dropout, data ordering, etc. <br/> 随机性出现在很多地方: 参数初始化、dropout、数据排序等。")
    text("For reproducibility, we recommend you always pass in a different random seed for each use of randomness. <br/> 为了可重复性，我们建议你每次使用随机性时都传入不同的随机种子。")
    text("Determinism is particularly useful when debugging, so you can hunt down the bug. <br/> 确定性在调试时特别有用，这样你可以追踪 bug。")

    text("There are three places to set the random seed which you should do all at once just to be safe. <br/> 有三个地方需要设置随机种子，为了安全起见你应该同时设置它们。")

    # Torch
    seed = 0
    torch.manual_seed(seed)

    # NumPy
    import numpy as np
    np.random.seed(seed)

    # Python
    import random
    random.seed(seed)


def data_loading():
    text("In language modeling, data is a sequence of integers (output by the tokenizer). <br/> 在语言建模中，数据是一个整数序列 (由分词器输出)。")

    text("It is convenient to serialize them as numpy arrays (done by the tokenizer). <br/> 将它们序列化为 numpy 数组很方便 (由分词器完成)。")
    orig_data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.int32)
    orig_data.tofile("data.npy")

    text("You can load them back as numpy arrays. <br/> 你可以将它们作为 numpy 数组加载回来。")
    text("Don't want to load the entire data into memory at once (LLaMA data is 2.8TB). <br/> 不想一次将所有数据加载到内存中 (LLaMA 数据是 2.8TB)。")
    text("Use memmap to lazily load only the accessed parts into memory. <br/> 使用 memmap 惰性地只将访问的部分加载到内存中。")
    data = np.memmap("data.npy", dtype=np.int32)
    assert np.array_equal(data, orig_data)

    text("A *data loader* generates a batch of sequences for training. <br/> *数据加载器* 生成一批序列用于训练。")
    B = 2  # Batch size
    L = 4  # Length of sequence
    x = get_batch(data, batch_size=B, sequence_length=L, device=get_device())
    assert x.size() == torch.Size([B, L])


class SGD(torch.optim.Optimizer):
    def __init__(self, params: Iterable[nn.Parameter], lr: float = 0.01):
        super(SGD, self).__init__(params, dict(lr=lr))

    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                grad = p.grad.data
                p.data -= lr * grad


class AdaGrad(torch.optim.Optimizer):
    def __init__(self, params: Iterable[nn.Parameter], lr: float = 0.01):
        super(AdaGrad, self).__init__(params, dict(lr=lr))

    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                # Optimizer state
                state = self.state[p]
                grad = p.grad.data

                # Get squared gradients g2 = sum_{i<t} g_i^2
                g2 = state.get("g2", torch.zeros_like(grad))

                # Update optimizer state
                g2 += torch.square(grad)
                state["g2"] = g2

                # Update parameters
                p.data -= lr * grad / torch.sqrt(g2 + 1e-5)


def optimizer():
    text("Recall our deep linear model. <br/> 回顾我们的深度线性模型。")
    B = 2
    D = 4
    num_layers = 2
    model = Cruncher(dim=D, num_layers=num_layers).to(get_device())

    text("Let's define the AdaGrad optimizer <br/> 让我们定义 AdaGrad 优化器")
    text("- momentum = SGD + exponential averaging of grad <br/> - 动量 = SGD + 梯度的指数平均")
    text("- AdaGrad = SGD + averaging by grad^2 <br/> - AdaGrad = SGD + 按 grad^2 平均")
    text("- RMSProp = AdaGrad + exponentially averaging of grad^2 <br/> - RMSProp = AdaGrad + grad^2 的指数平均")
    text("- Adam = RMSProp + momentum <br/> - Adam = RMSProp + 动量")

    text("AdaGrad: <br/> AdaGrad: "), link("https://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf")
    optimizer = AdaGrad(model.parameters(), lr=0.01)
    state = model.state_dict()  # @inspect state

    text("Compute gradients <br/> 计算梯度")
    x = torch.randn(B, D, device=get_device())
    y = torch.tensor([4., 5.], device=get_device())
    pred_y = model(x)
    loss = F.mse_loss(input=pred_y, target=y)
    loss.backward()

    text("Take a step <br/> 执行一步更新")
    optimizer.step()
    state = model.state_dict()  # @inspect state

    text("Free up the memory (optional) <br/> 释放内存 (可选)")
    optimizer.zero_grad(set_to_none=True)

    text("## Memory <br/> ## 内存")

    # Parameters
    num_parameters = (D * D * num_layers) + D  # @inspect num_parameters
    assert num_parameters == get_num_parameters(model)

    # Activations
    num_activations = B * D * num_layers  # @inspect num_activations

    # Gradients
    num_gradients = num_parameters  # @inspect num_gradients

    # Optimizer states
    num_optimizer_states = num_parameters  # @inspect num_optimizer_states

    # Putting it all together, assuming float32
    total_memory = 4 * (num_parameters + num_activations + num_gradients + num_optimizer_states)  # @inspect total_memory

    text("## Compute (for one step) <br/> ## 计算 (单步)")
    flops = 6 * B * num_parameters  # @inspect flops

    text("## Transformers <br/> ## Transformer")

    text("The accounting for a Transformer is more complicated, but the same idea. <br/> Transformer 的核算更复杂，但思路相同。")
    text("Assignment 1 will ask you to do that. <br/> 作业 1 会要求你完成这个。")

    text("Blog post describing memory usage for Transformer training <br/> 描述 Transformer 训练内存使用的博客文章 "), article_link("https://erees.dev/transformer-memory/")
    text("Blog post descibing FLOPs for a Transformer: <br/> 描述 Transformer FLOPs 的博客文章: "), article_link("https://www.adamcasson.com/posts/transformer-flops")


def train_loop():
    text("Generate data from linear function with weights (0, 1, 2, ..., D-1). <br/> 从权重为 (0, 1, 2, ..., D-1) 的线性函数生成数据。")
    D = 16
    true_w = torch.arange(D, dtype=torch.float32, device=get_device())
    def get_batch(B: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.randn(B, D).to(get_device())
        true_y = x @ true_w
        return (x, true_y)

    text("Let's do a basic run <br/> 让我们做一次基本运行")
    train("simple", get_batch, D=D, num_layers=0, B=4, num_train_steps=10, lr=0.01)

    text("Do some hyperparameter tuning <br/> 做一些超参数调优")
    train("simple", get_batch, D=D, num_layers=0, B=4, num_train_steps=10, lr=0.1)


def train(name: str, get_batch,
          D: int, num_layers: int,
          B: int, num_train_steps: int, lr: float):
    model = Cruncher(dim=D, num_layers=0).to(get_device())
    optimizer = SGD(model.parameters(), lr=0.01)

    for t in range(num_train_steps):
        # Get data
        x, y = get_batch(B=B)

        # Forward (compute loss)
        pred_y = model(x)
        loss = F.mse_loss(pred_y, y)

        # Backward (compute gradients)
        loss.backward()

        # Update parameters
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)


def checkpointing():
    text("Training language models take a long time and certainly will certainly crash. <br/> 训练语言模型需要很长时间，而且肯定会崩溃。")
    text("You don't want to lose all your progress. <br/> 你不想失去所有的进度。")

    text("During training, it is useful to periodically save your model and optimizer state to disk. <br/> 在训练期间，定期将模型和优化器状态保存到磁盘是很有用的。")

    model = Cruncher(dim=64, num_layers=3).to(get_device())
    optimizer = AdaGrad(model.parameters(), lr=0.01)

    text("Save the checkpoint: <br/> 保存检查点:")
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, "model_checkpoint.pt")

    text("Load the checkpoint: <br/> 加载检查点:")
    loaded_checkpoint = torch.load("model_checkpoint.pt")


def mixed_precision_training():
    text("Choice of data type (float32, bfloat16, fp8) have tradeoffs. <br/> 数据类型的选择 (float32, bfloat16, fp8) 有权衡。")
    text("- Higher precision: more accurate/stable, more memory, more compute <br/> - 更高精度: 更准确/稳定，更多内存，更多计算")
    text("- Lower precision: less accurate/stable, less memory, less compute <br/> - 更低精度: 较不准确/稳定，更少内存，更少计算")

    text("How can we get the best of both worlds? <br/> 我们如何兼得两者的优点?")

    text("Solution: use float32 by default, but use {bfloat16, fp8} when possible. <br/> 解决方案: 默认使用 float32，但尽可能使用 {bfloat16, fp8}。")

    text("A concrete plan: <br/> 一个具体的计划:")
    text("- Use {bfloat16, fp8} for the forward pass (activations). <br/> - 前向传播 (激活值) 使用 {bfloat16, fp8}。")
    text("- Use float32 for the rest (parameters, gradients). <br/> - 其余部分 (参数、梯度) 使用 float32。")

    text("- Mixed precision training <br/> - 混合精度训练 "), link("https://arxiv.org/pdf/1710.03740.pdf")

    text("Pytorch has an automatic mixed precision (AMP) library. <br/> Pytorch 有一个自动混合精度 (AMP) 库。")
    link("https://pytorch.org/docs/stable/amp.html")
    link("https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/")

    text("NVIDIA's Transformer Engine supports FP8 for linear layers <br/> NVIDIA 的 Transformer Engine 支持线性层使用 FP8")
    text("Use FP8 pervasively throughout training <br/> 在训练过程中普遍使用 FP8 "), link("https://arxiv.org/pdf/2310.18313.pdf")


############################################################

def get_memory_usage(x: torch.Tensor):
    return x.numel() * x.element_size()


def get_promised_flop_per_sec(device: str, dtype: torch.dtype) -> float:
    """Return the peak FLOP/s for `device` operating on `dtype`."""
    if not torch.cuda.is_available():
        text("No CUDA device available, so can't get FLOP/s. <br/> 没有可用的 CUDA 设备，所以无法获取 FLOP/s。")
        return 1
    properties = torch.cuda.get_device_properties(device)

    if "A100" in properties.name:
        # https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf")
        if dtype == torch.float32:
            return 19.5e12
        if dtype in (torch.bfloat16, torch.float16):
            return 312e12
        raise ValueError(f"Unknown dtype: {dtype}")

    if "H100" in properties.name:
        # https://resources.nvidia.com/en-us-tensor-core/nvidia-tensor-core-gpu-datasheet")
        if dtype == torch.float32:
            return 67.5e12
        if dtype in (torch.bfloat16, torch.float16):
            return 1979e12 / 2  # 1979 is for sparse, dense is half of that
        raise ValueError(f"Unknown dtype: {dtype}")

    raise ValueError(f"Unknown device: {device}")


def same_storage(x: torch.Tensor, y: torch.Tensor):
    return x.untyped_storage().data_ptr() == y.untyped_storage().data_ptr()


def time_matmul(a: torch.Tensor, b: torch.Tensor) -> float:
    """Return the number of seconds required to perform `a @ b`."""

    # Wait until previous CUDA threads are done
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    def run():
        # Perform the operation
        a @ b

        # Wait until CUDA threads are done
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    # Time the operation `num_trials` times
    num_trials = 5
    total_time = timeit.timeit(run, number=num_trials)

    return total_time / num_trials


def get_num_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())

def get_device(index: int = 0) -> torch.device:
    """Try to use the GPU if possible, otherwise, use CPU."""
    if torch.cuda.is_available():
        return torch.device(f"cuda:{index}")
    else:
        return torch.device("cpu")

if __name__ == "__main__":
    main()
