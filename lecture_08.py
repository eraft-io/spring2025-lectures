import torch
import time
import os
from typing import List, Callable
import torch.nn.functional as F
import torch.distributed as dist
import torch.distributed.fsdp
from execute_util import text, image, link, system_text
from torch_util import get_device
from lecture_util import article_link
from lecture_08_utils import spawn, int_divide, summarize_tensor, get_init_params, render_duration

def main():
    text("Last week: parallelism within a single GPU <br/> 上周：单 GPU 内的并行")
    text("This week: parallelism across multiple GPUs <br/> 本周：多 GPU 间的并行")
    image("images/gpu-node-overview.png", width=500)

    text("In both cases, **compute** (arithmetic logic units) is far from inputs/outputs (**data**). <br/> 在这两种情况下，**计算** (算术逻辑单元) 远离输入/输出 (**数据**)。")
    text("Unifying theme: orchestrate computation to avoid data transfer bottlenecks <br/> 统一的主题：编排计算以避免数据传输瓶颈")

    text("Last week: reduce memory accesses via fusion/tiling <br/> 上周：通过融合/分块减少内存访问")
    text("This week: reduce communication across GPUs/nodes via replication/sharding <br/> 本周：通过复制/分片减少 GPU/节点间的通信")

    text("Generalized hierarchy (from small/fast to big/slow): <br/> 通用层次结构 (从小/快到大/慢)：")
    text("- Single node, single GPU: L1 cache / shared memory <br/> - 单节点，单 GPU：L1 缓存/共享内存")
    text("- Single node, single GPU: HBM <br/> - 单节点，单 GPU：HBM")
    text("- Single node, multi-GPU: NVLink <br/> - 单节点，多 GPU：NVLink")
    text("- Multi-node, multi-GPU: NVSwitch <br/> - 多节点，多 GPU：NVSwitch")

    text("This lecture: concretize the concepts from last lecture in code <br/> 本节课：在代码中具体化上节课的概念")

    link(title="[stdout for this lecture]", url="var/traces/lecture_08_stdout.txt")

    text("### Part 1: building blocks of distributed communication/computation <br/> ### 第 1 部分：分布式通信/计算的构建模块")
    collective_operations()    # Conceptual programming interface
    torch_distributed()        # How this is implemented in NCCL/PyTorch
    benchmarking()             # Measure actual NCCL bandwidth

    text("### Part 2: distributed training <br/> ### 第 2 部分：分布式训练")
    text("Walk through bare-bones implementations of each strategy on deep MLPs. <br/> 逐步实现深度 MLP 每种策略的基本版本。")
    text("Recall that MLPs are the compute bottleneck in Transformers, so this is representative. <br/> 记住 MLP 是 Transformer 中的计算瓶颈，所以这是代表性的。")
    data_parallelism()         # Cut up along the batch dimension
    tensor_parallelism()       # Cut up along the width dimension
    pipeline_parallelism()     # Cut up along the depth dimension

    text("What's missing? <br/> 缺少什么？")
    text("- More general models (with attention, etc.) <br/> - 更通用的模型 (如带注意力机制等)")
    text("- More communication/computation overlap <br/> - 更多的通信/计算重叠")
    text("- This require more complex code with more bookkeeping <br/> - 这需要更复杂的代码和更多的记录工作")
    text("- Jax/TPUs: just define the model, the sharding strategy, and the Jax compiler handles the rest <br/> - Jax/TPUs：只需定义模型和分片策略，Jax 编译器处理其余工作 "), link(title="[levanter]", url="https://crfm.stanford.edu/2023/06/16/levanter-1_0-release.html")
    text("- But we're doing PyTorch so you can see how one builds up from the primitives <br/> - 但我们使用 PyTorch 是为了让你们看到如何从基本原语构建")

    text("### Summary <br/> ### 总结")
    text("- Many ways to parallelize: data (batch), tensor/expert (width), pipeline (depth), sequence (length) <br/> - 多种并行化方式：数据 (批次)、张量/专家 (宽度)、流水线 (深度)、序列 (长度)")
    text("- Can **re-compute** or store in **memory** or store in another GPUs memory and **communicate** <br/> - 可以**重新计算**或存储在**内存**或存储在另一个 GPU 内存中并进行**通信**")
    text("- Hardware is getting faster, but will always want bigger models, so will have this hierarchical structure <br/> - 硬件越来越快，但永远会想要更大的模型，所以会保持这种层次结构")


def collective_operations():
    text("**Collective operations** are the conceptual primitives used for distributed programming <br/> **集合操作** 是用于分布式编程的概念原语 "), article_link("https://en.wikipedia.org/wiki/Collective_operation")
    text("- Collective means that you specify communication pattern across many (e.g., 256) nodes. <br/> - 集合意味着你指定跨多个 (如 256) 节点的通信模式。")
    text("- These are classic in the parallel programming literature from the 1980s. <br/> - 这些是 1980 年代并行编程文献中的经典内容。")
    text("- Better/faster abstraction than managing point-to-point communication yourself. <br/> - 比自己管理点对点通信更好的/更快的抽象。")

    text("Terminology: <br/> 术语：")
    text("- **World size**: number of devices (e.g., 4) <br/> - **世界大小**：设备数量 (如 4)")
    text("- **Rank**: a device (e.g., 0, 1, 2, 3) <br/> - **进程编号**：一个设备 (如 0, 1, 2, 3)")

    text("### Broadcast <br/> ### 广播"), image("https://pytorch.org/tutorials/_images/broadcast.png", width=400)

    text("### Scatter <br/> ### 散射"), image("https://pytorch.org/tutorials/_images/scatter.png", width=400)

    text("### Gather <br/> ### 收集"), image("https://pytorch.org/tutorials/_images/gather.png", width=400)

    text("### Reduce <br/> ### 归约"), image("https://pytorch.org/tutorials/_images/reduce.png", width=400)

    text("### All-gather <br/> ### 全收集"), image("https://pytorch.org/tutorials/_images/all_gather.png", width=400)

    text("### Reduce-scatter <br/> ### 散射归约"), image("https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/_images/reducescatter.png", width=400)

    text("### All-reduce = reduce-scatter + all-gather <br/> ### 全归约 = 散射归约 + 全收集"), image("https://pytorch.org/tutorials/_images/all_reduce.png", width=400)

    text("Way to remember the terminology: <br/> 记住术语的方式：")
    text("- Reduce: performs some associative/commutative operation (sum, min, max) <br/> - 归约：执行某些结合/交换运算 (求和、最小值、最大值)")
    text("- Broadcast/scatter is inverse of gather <br/> - 广播/散射是收集的逆操作")
    text("- All: means destination is all devices <br/> - All：意味着目标是所有设备")


def torch_distributed():
    text("### Hardware <br/> ### 硬件")
    text("Classic (in the home): <br/> 经典 (在家用)：")
    image("https://media.springernature.com/lw685/springer-static/image/art%3A10.1186%2Fs42774-021-00098-3/MediaObjects/42774_2021_98_Fig1_HTML.png?as=webp", width=400)
    text("- GPUs on same node communicate via a PCI(e) bus (v7.0, 16 lanes => 242 GB/s) <br/> - 同一节点上的 GPU 通过 PCI(e) 总线通信 (v7.0，16 通道 => 242 GB/s) "), article_link("https://en.wikipedia.org/wiki/PCI_Express")
    text("- GPUs on different nodes communicate via Ethernet (~200 MB/s) <br/> - 不同节点上的 GPU 通过以太网通信 (~200 MB/s)")

    text("Modern (in the data center): <br/> 现代 (在数据中心)：")
    image("https://www.nextplatform.com/wp-content/uploads/2018/04/nvidia-nvswitch-topology-two.jpg", width=400)
    text("- Within a node: NVLink connects GPUs directly, bypass CPU <br/> - 节点内：NVLink 直接连接 GPU，绕过 CPU")
    text("- Across nodes: NVSwitch connects GPUs directly, bypass Ethernet <br/> - 跨节点：NVSwitch 直接连接 GPU，绕过以太网")

    text("Each H100 has 18 NVLink 4.0 links, for a total of 900GB/s <br/> 每个 H100 有 18 个 NVLink 4.0 链接，总计 900GB/s "), article_link("https://www.nvidia.com/en-us/data-center/nvlink/")
    text("In comparison, memory bandwidth for HBM is 3.9 TB/s <br/> 相比之下，HBM 内存带宽为 3.9 TB/s "), article_link("https://resources.nvidia.com/en-us-tensor-core/nvidia-tensor-core-gpu-datasheet")

    text("Let's check what our hardware setup is. <br/> 让我们检查我们的硬件设置。 "), article_link("https://guide.ncloud-docs.com/docs/en/server-baremetal-a100-check-vpc")
    if torch.cuda.is_available():
        os.system("nvidia-smi topo -m")
        text("Note GPUs are connected via NV18, also connected to NICs (for PCIe) <br/> 注意：GPU 通过 NV18 连接，也连接到网卡 (用于 PCIe)")

    text("### NVIDIA Collective Communication Library (NCCL) <br/> ### NVIDIA 集合通信库 (NCCL)")
    text("NCCL translates collective operations into low-level packets that are sent between GPUs. <br/> NCCL 将集合操作转换为在 GPU 之间发送的低级数据包。 "), link(title="[talk]", url="https://www.nvidia.com/en-us/on-demand/session/gtcspring21-s31880/")
    text("- Detects topology of hardware (e.g., number of nodes, switches, NVLink/PCIe) <br/> - 检测硬件拓扑 (如节点数、交换机数、NVLink/PCIe)")
    text("- Optimizes the path between GPUs <br/> - 优化 GPU 之间的路径")
    text("- Launches CUDA kernels to send/receive data <br/> - 启动 CUDA 内核发送/接收数据")

    text("### PyTorch distributed library (`torch.distributed`) <br/> ### PyTorch 分布式库 (`torch.distributed`)")
    link(title="[Documentation]", url="https://pytorch.org/docs/stable/distributed.html")

    text("- Provides clean interface for collective operations (e.g., `all_gather_into_tensor`) <br/> - 为集合操作提供清晰的接口 (如 `all_gather_into_tensor`)")
    text("- Supports multiple backends for different hardware: gloo (CPU), nccl (GPU) <br/> - 支持不同硬件的多个后端：gloo (CPU)、nccl (GPU)")
    text("- Also supports higher-level algorithms (e.g., `FullyShardedDataParallel`) [not used in this course] <br/> - 还支持更高级的算法 (如 `FullyShardedDataParallel`) [本课程不使用]")

    text("Let's walk through some examples. <br/> 让我们通过一些例子来了解。")
    spawn(collective_operations_main, world_size=4)


def collective_operations_main(rank: int, world_size: int):
    """This function is running asynchronously for each process (rank = 0, ..., world_size - 1)."""
    setup(rank, world_size)

    # All-reduce
    dist.barrier()  # Waits for all processes to get to this point (in this case, for print statements)

    tensor = torch.tensor([0., 1, 2, 3], device=get_device(rank)) + rank  # Both input and output

    print(f"Rank {rank} [before all-reduce]: {tensor}", flush=True)
    dist.all_reduce(tensor=tensor, op=dist.ReduceOp.SUM, async_op=False)  # Modifies tensor in place
    print(f"Rank {rank} [after all-reduce]: {tensor}", flush=True)

    # Reduce-scatter
    dist.barrier()

    input = torch.arange(world_size, dtype=torch.float32, device=get_device(rank)) + rank  # Input
    output = torch.empty(1, device=get_device(rank))  # Allocate output

    print(f"Rank {rank} [before reduce-scatter]: input = {input}, output = {output}", flush=True)
    dist.reduce_scatter_tensor(output=output, input=input, op=dist.ReduceOp.SUM, async_op=False)
    print(f"Rank {rank} [after reduce-scatter]: input = {input}, output = {output}", flush=True)

    # All-gather
    dist.barrier()

    input = output  # Input is the output of reduce-scatter
    output = torch.empty(world_size, device=get_device(rank))  # Allocate output

    print(f"Rank {rank} [before all-gather]: input = {input}, output = {output}", flush=True)
    dist.all_gather_into_tensor(output_tensor=output, input_tensor=input, async_op=False)
    print(f"Rank {rank} [after all-gather]: input = {input}, output = {output}", flush=True)

    text("Indeed, all-reduce = reduce-scatter + all-gather! <br/> 确实，all-reduce = reduce-scatter + all-gather！")

    cleanup()


def benchmarking():
    text("Let's see how fast communication happens (restrict to one node). <br/> 让我们看看通信有多快 (限制在单个节点上)。")

    # All-reduce
    spawn(all_reduce, world_size=4, num_elements=100 * 1024**2)

    # Reduce-scatter
    spawn(reduce_scatter, world_size=4, num_elements=100 * 1024**2)

    # References
    link(title="How to reason about operations", url="https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md#allreduce")
    link(title="Sample code", url="https://github.com/stas00/ml-engineering/blob/master/network/benchmarks/all_reduce_bench.py")


def all_reduce(rank: int, world_size: int, num_elements: int):
    setup(rank, world_size)

    # Create tensor
    tensor = torch.randn(num_elements, device=get_device(rank))

    # Warmup
    dist.all_reduce(tensor=tensor, op=dist.ReduceOp.SUM, async_op=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # Wait for CUDA kernels to finish
        dist.barrier()            # Wait for all the processes to get here

    # Perform all-reduce
    start_time = time.time()
    dist.all_reduce(tensor=tensor, op=dist.ReduceOp.SUM, async_op=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # Wait for CUDA kernels to finish
        dist.barrier()            # Wait for all the processes to get here
    end_time = time.time()

    duration = end_time - start_time
    print(f"[all_reduce] Rank {rank}: all_reduce(world_size={world_size}, num_elements={num_elements}) took {render_duration(duration)}", flush=True)

    # Measure the effective bandwidth
    dist.barrier()
    size_bytes = tensor.element_size() * tensor.numel()
    sent_bytes = size_bytes * 2 * (world_size - 1)  # 2x because send input and receive output
    total_duration = world_size * duration
    bandwidth = sent_bytes / total_duration
    print(f"[all_reduce] Rank {rank}: all_reduce measured bandwidth = {round(bandwidth / 1024**3)} GB/s", flush=True)

    cleanup()


def reduce_scatter(rank: int, world_size: int, num_elements: int):
    setup(rank, world_size)

    # Create input and outputs
    input = torch.randn(world_size, num_elements, device=get_device(rank))  # Each rank has a matrix
    output = torch.empty(num_elements, device=get_device(rank))

    # Warmup
    dist.reduce_scatter_tensor(output=output, input=input, op=dist.ReduceOp.SUM, async_op=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # Wait for CUDA kerels to finish
        dist.barrier()            # Wait for all the processes to get here

    # Perform reduce-scatter
    start_time = time.time()
    dist.reduce_scatter_tensor(output=output, input=input, op=dist.ReduceOp.SUM, async_op=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # Wait for CUDA kerels to finish
        dist.barrier()            # Wait for all the processes to get here
    end_time = time.time()

    duration = end_time - start_time
    print(f"[reduce_scatter] Rank {rank}: reduce_scatter(world_size={world_size}, num_elements={num_elements}) took {render_duration(duration)}", flush=True)

    # Measure the effective bandwidth
    dist.barrier()
    data_bytes = input.element_size() * input.numel()  # How much data in the input
    sent_bytes = data_bytes * (world_size - 1)  # How much needs to be sent (no 2x here)
    total_duration = world_size * duration  # Total time for transmission
    bandwidth = sent_bytes / total_duration
    print(f"[reduce_scatter] Rank {rank}: reduce_scatter measured bandwidth = {round(bandwidth / 1024**3)} GB/s", flush=True)

    cleanup()


def data_parallelism():
    image("images/data-parallelism.png", width=300)
    text("Sharding strategy: each rank gets a slice of the data <br/> 分片策略：每个进程获得数据的切片")

    data = generate_sample_data()
    spawn(data_parallelism_main, world_size=4, data=data, num_layers=4, num_steps=1)

    text("Notes: <br/> 注意：")
    text("- Losses are different across ranks (computed on local data) <br/> - 不同进程上的损失不同 (在本地数据上计算)")
    text("- Gradients are all-reduced to be the same across ranks <br/> - 梯度经过 all-reduce 在各进程上保持一致")
    text("- Therefore, parameters remain the same across ranks <br/> - 因此，参数在各进程上保持一致")


def generate_sample_data():
    batch_size = 128
    num_dim = 1024
    data = torch.randn(batch_size, num_dim)
    return data


def data_parallelism_main(rank: int, world_size: int, data: torch.Tensor, num_layers: int, num_steps: int):
    setup(rank, world_size)

    # Get the slice of data for this rank (in practice, each rank should load only its own data)
    batch_size = data.size(0)  # @inspect batch_size
    num_dim = data.size(1)  # @inspect num_dim
    local_batch_size = int_divide(batch_size, world_size)  # @inspect local_batch_size
    start_index = rank * local_batch_size  # @inspect start_index
    end_index = start_index + local_batch_size  # @inspect end_index
    data = data[start_index:end_index].to(get_device(rank))

    # Create MLP parameters params[0], ..., params[num_layers - 1] (each rank has all parameters)
    params = [get_init_params(num_dim, num_dim, rank) for i in range(num_layers)]
    optimizer = torch.optim.AdamW(params, lr=1e-3)  # Each rank has own optimizer state

    for step in range(num_steps):
        # Forward pass
        x = data
        for param in params:
            x = x @ param
            x = F.gelu(x)
        loss = x.square().mean()  # Loss function is average squared magnitude

        # Backward pass
        loss.backward()

        # Sync gradients across workers (only difference between standard training and DDP)
        for param in params:
            dist.all_reduce(tensor=param.grad, op=dist.ReduceOp.AVG, async_op=False)

        # Update parameters
        optimizer.step()

        print(f"[data_parallelism] Rank {rank}: step = {step}, loss = {loss.item()}, params = {[summarize_tensor(params[i]) for i in range(num_layers)]}", flush=True)

    cleanup()


def tensor_parallelism():
    image("images/tensor-parallelism.png", width=300)
    text("Sharding strategy: each rank gets part of each layer, transfer all data/activations <br/> 分片策略：每个进程获得每一层的一部分，传输所有数据/激活值")

    data = generate_sample_data()
    spawn(tensor_parallelism_main, world_size=4, data=data, num_layers=4)


def tensor_parallelism_main(rank: int, world_size: int, data: torch.Tensor, num_layers: int):
    setup(rank, world_size)

    data = data.to(get_device(rank))
    batch_size = data.size(0)  # @inspect batch_size
    num_dim = data.size(1)  # @inspect num_dim
    local_num_dim = int_divide(num_dim, world_size)  # Shard `num_dim`  @inspect local_num_dim

    # Create model (each rank gets 1/world_size of the parameters)
    params = [get_init_params(num_dim, local_num_dim, rank) for i in range(num_layers)]

    # Forward pass
    x = data
    for i in range(num_layers):
        # Compute activations (batch_size x local_num_dim)
        x = x @ params[i]  # Note: this is only on a slice of the parameters
        x = F.gelu(x)

        # Allocate memory for activations (world_size x batch_size x local_num_dim)
        activations = [torch.empty(batch_size, local_num_dim, device=get_device(rank)) for _ in range(world_size)]

        # Send activations via all gather
        dist.all_gather(tensor_list=activations, tensor=x, async_op=False)

        # Concatenate them to get batch_size x num_dim
        x = torch.cat(activations, dim=1)

    print(f"[tensor_parallelism] Rank {rank}: forward pass produced activations {summarize_tensor(x)}", flush=True)

    # Backward pass: homework exercise

    cleanup()


def pipeline_parallelism():
    image("images/pipeline-parallelism.png", width=300)
    text("Sharding strategy: each rank gets subset of layers, transfer all data/activations <br/> 分片策略：每个进程获得层的子集，传输所有数据/激活值")

    data = generate_sample_data()
    spawn(pipeline_parallelism_main, world_size=2, data=data, num_layers=4, num_micro_batches=4)


def pipeline_parallelism_main(rank: int, world_size: int, data: torch.Tensor, num_layers: int, num_micro_batches: int):
    setup(rank, world_size)

    # Use all the data
    data = data.to(get_device(rank))
    batch_size = data.size(0)  # @inspect batch_size
    num_dim = data.size(1)  # @inspect num_dim

    # Split up layers
    local_num_layers = int_divide(num_layers, world_size)  # @inspect local_num_layers

    # Each rank gets a subset of layers
    local_params = [get_init_params(num_dim, num_dim, rank) for i in range(local_num_layers)]

    # Forward pass

    # Break up into micro batches to minimize the bubble
    micro_batch_size = int_divide(batch_size, num_micro_batches)  # @inspect micro_batch_size
    if rank == 0:
        # The data
        micro_batches = data.chunk(chunks=num_micro_batches, dim=0)
    else:
        # Allocate memory for activations
        micro_batches = [torch.empty(micro_batch_size, num_dim, device=get_device(rank)) for _ in range(num_micro_batches)]

    for x in micro_batches:
        # Get activations from previous rank
        if rank - 1 >= 0:
            dist.recv(tensor=x, src=rank - 1)

        # Compute layers assigned to this rank
        for param in local_params:
            x = x @ param
            x = F.gelu(x)

        # Send to the next rank
        if rank + 1 < world_size:
            print(f"[pipeline_parallelism] Rank {rank}: sending {summarize_tensor(x)} to rank {rank + 1}", flush=True)
            dist.send(tensor=x, dst=rank + 1)

    text("Not handled: overlapping communication/computation to eliminate pipeline bubbles <br/> 未处理：重叠通信/计算以消除流水线气泡")

    # Backward pass: homework exercise

    cleanup()

############################################################

def setup(rank: int, world_size: int):
    # Specify where master lives (rank 0), used to coordinate (actual data goes through NCCL)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "15623"

    if torch.cuda.is_available():
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    else:
        dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
