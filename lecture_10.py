from sympy import symbols, oo
from execute_util import text, link, image
from lecture_util import article_link
from references import Reference, llama3, gqa, mla, longformer, sparse_transformer, mistral_7b

# Define symbols corresponding to the shape of the Transformer model
B, S, T, D, F, N, K, H, L, V = symbols("B S T D F N K H L V", positive=True)
c = symbols("c", positive=True)  # Just a constant that helps with taking limits
memory_bandwidth = symbols("memory_bandwidth", positive=True)

scaling_book_transformers = Reference(title="[Scaling book chapter on Transformers]", url="https://jax-ml.github.io/scaling-book/transformers/")
scaling_book_inference = Reference(title="[Scaling book chapter on Transformers]", url="https://jax-ml.github.io/scaling-book/inference/")

def main():
    text("**Inference**: given a **fixed model**, generate responses given prompts <br/> **推理**：给定一个**固定模型**，根据提示生成响应")

    text("### Understanding the inference workload <br/> ### 理解推理工作负载")
    landscape()
    review_transformer()
    review_of_arithmetic_intensity()
    arithmetic_intensity_of_inference()
    throughput_and_latency()

    text("### Taking shortcuts (lossy) <br/> ### 走捷径 (有损)")
    reduce_kv_cache_size()
    alternatives_to_the_transformer()
    quantization()
    model_pruning()

    text("Summary: reduce inference complexity without hurting accuracy <br/> 总结：减少推理复杂度而不影响准确性")

    text("From scratch recipe: <br/> 从零开始配方：")
    text("1. Define faster model architecture <br/> 1. 定义更快的模型架构")
    text("2. Train faster model <br/> 2. 训练更快的模型")

    text("Distillation recipe: <br/> 蒸馏配方：")
    text("1. Define faster model architecture <br/> 1. 定义更快的模型架构")
    text("2. Initialize weights using original model (which has a different architecture) <br/> 2. 使用原始模型的权重初始化 (架构不同)")
    text("3. Repair faster model (distillation) <br/> 3. 修复更快的模型 (蒸馏)")

    text("### Use shortcuts but double check (lossless) <br/> ### 使用捷径但仔细检查 (无损)")
    speculative_sampling()

    text("### Handling dynamic workloads <br/> ### 处理动态工作负载")
    text("Batching over sequences in live traffic is tricky because: <br/> 实时流量中跨序列批处理很棘手，因为：")
    text("1. Requests arrive at different times (waiting for batch is bad for early requests) <br/> 1. 请求在不同时间到达 (等待批处理对先到的请求不利)")
    text("2. Sequences have shared prefixes (e.g., system prompts, generating multiple samples) <br/> 2. 序列有共享前缀 (如系统提示、生成多个样本)")
    text("3. Sequences have different lengths (padding is inefficient) <br/> 3. 序列长度不同 (填充效率低)")

    continuous_batching()
    paged_attention()

    text("### Summary <br/> ### 总结")
    text("- Inference is important (actual use, evaluation, reinforcement learning) <br/> - 推理很重要 (实际使用、评估、强化学习)")
    text("- Different characteristics compared to training (memory-limited, dynamic) <br/> - 与训练相比具有不同特点 (内存受限、动态)")
    text("- Techniques: new architectures, quantization, pruning/distillation, speculative decoding <br/> - 技术：新架构、量化、剪枝/蒸馏、投机解码")
    text("- Ideas from systems (speculative execution, paging) <br/> - 来自系统的思路 (投机执行、分页)")
    text("- New architectures have huge potential for improvement <br/> - 新架构有巨大的改进潜力")


def landscape():
    text("Inference shows up in many places: <br/> 推理出现在许多地方：")
    text("- Actual use (chatbots, code completion, batch data processing) <br/> - 实际使用 (聊天机器人、代码补全、批数据处理)")
    text("- Model evaluation (e.g., on instruction following) <br/> - 模型评估 (如指令遵循)")
    text("- Test-time compute (thinking requires more inference) <br/> - 测试时计算 (思考需要更多推理)")
    text("- Training via reinforcement learning (sample generation, then score) <br/> - 通过强化学习训练 (样本生成，然后评分)")

    text("Why **efficiency** matters: training is one-time cost, inference is repeated many times <br/> 为什么**效率**很重要：训练是一次性成本，推理会重复多次")
    image("images/openai-100b-tokens.png", width=600); link(title=" [tweet]", url="https://x.com/sama/status/1756089361609981993")
    image("images/cursor-1b-lines.png", width=600); link(title=" [tweet]", url="https://x.com/amanrsanger/status/1916968123535880684")

    text("Metrics: <br/> 指标：")
    text("- Time-to-first-token (TTFT): how long user waits before any generation happens (matters for interactive applications) <br/> - 首token时间 (TTFT)：用户在生成开始前等待多久 (对交互式应用很重要)")
    text("- Latency (seconds/token): how fast tokens appear for a user (matters for interactive applications) <br/> - 延迟 (秒/token)：token 对用户出现的速度 (对交互式应用很重要)")
    text("- Throughput (tokens/second): useful for batch processing applications <br/> - 吞吐量 (token/秒)：对批处理应用很有用")

    text("Key considerations in efficiency: <br/> 效率的关键考虑因素：")
    text("- Training (supervised): you see all tokens, can parallelize over sequence (matmul in Transformer) <br/> - 训练 (监督)：你看到所有 token，可以跨序列并行 (Transformer 中的矩阵乘法)")
    text("- Inference: you have to generate sequentially, can't parallelize, so harder to fully utilize compute <br/> - 推理：你必须顺序生成，无法并行，因此更难充分利用计算")

    text("Companies doing inference (a big deal for anyone who has a product or platform): <br/> 做推理的公司 (对任何有产品或平台的人来说都很重要)：")
    text("- Providers serving closed models (OpenAI, Anthropic, Google, etc.) <br/> - 提供闭源模型的服务商 (OpenAI、Anthropic、Google 等)")
    text("- Providers serving open-weight models (Together, Fireworks, DeepInfra, etc.) <br/> - 提供开源权重模型的服务商 (Together、Fireworks、DeepInfra 等)")

    text("Open-source packages: <br/> 开源包：")
    text("- vLLM (Berkeley) "), link(title="[talk]", url="https://www.youtube.com/watch?v=8BaEwoTk8XI")
    text("- Tensor-RT (NVIDIA) "), article_link("https://nvidia.github.io/TensorRT-LLM/overview.html")
    text("- TGI (Hugging Face) "), article_link("https://huggingface.co/docs/text-generation-inference/en/index")


def review_transformer():
    link(scaling_book_transformers)
    image("https://jax-ml.github.io/scaling-book/assets/img/transformer-diagram.png", width=800)
    text("Simplifications (following conventions): `F = 4*D, D = N*H, N = K*G, S = T` <br/> 简化 (遵循约定)：")
    text("FLOPs for a feedforward pass: 6 * (B*T) * (num_params + O(T)) <br/> 前向传播的 FLOPs：")


def review_of_arithmetic_intensity():
    text("Setup: multiply X (B x D) and W (D x F) matrix <br/> 设置：矩阵 X (B x D) 和 W (D x F) 相乘")
    text("Intuition: B is batch size, D is hidden dimension, F is up-projection dimension in MLP <br/> 直观理解：B 是批量大小，D 是隐藏维度，F 是 MLP 中的上投影维度")

    text("Let's do FLOPs and memory read/write accounting for the matrix multiplication (X * W). <br/> 让我们计算矩阵乘法 (X * W) 的 FLOPs 和内存读写。")
    flops = 0
    bytes_transferred = 0

    text("Steps: <br/> 步骤：")
    text("1. Read X (B x D) from HBM <br/> 1. 从 HBM 读取 X (B x D)")
    bytes_transferred += 2*B*D
    text("2. Read W (D x F) from HBM <br/> 2. 从 HBM 读取 W (D x F)")
    bytes_transferred += 2*D*F
    text("3. Compute Y = X (B x D) @ W (D x F) <br/> 3. 计算 Y = X (B x D) @ W (D x F)")
    flops += 2*B*D*F
    text("4. Write Y (B x F) to HBM <br/> 4. 将 Y (B x F) 写入 HBM")
    bytes_transferred += 2*B*F

    text("Let's take stock of the accounting results. <br/> 让我们统计一下计算结果。")
    assert flops == 2*B*D*F
    assert bytes_transferred == 2*B*D + 2*D*F + 2*B*F
    text("Recall that **arithmetic intensity** is how much compute we do per byte transferred (want to be high). <br/> 回想一下，**算术强度**是每字节传输的计算量 (希望高)。")
    intensity = (flops / bytes_transferred).simplify()  # @inspect intensity

    text("Assuming B is much less than D and F, then we can simplify: <br/> 假设 B 远小于 D 和 F，那么我们可以简化：")
    intensity = intensity.subs(D, c*B).subs(F, c*B).limit(c, oo).simplify()  # @inspect intensity
    assert intensity == B

    text("Accelerator intensity of H100: <br/> H100 的加速器强度：")
    flops_per_second = 989e12
    memory_bandwidth = 3.35e12
    accelerator_intensity = flops_per_second / memory_bandwidth  # @inspect accelerator_intensity
    assert round(accelerator_intensity) == 295

    text("If computation intensity > accelerator intensity, **compute-limited** (good) <br/> 如果计算强度 > 加速器强度，**计算受限** (好)")
    text("If computation intensity < accelerator intensity, **memory-limited** (bad) <br/> 如果计算强度 < 加速器强度，**内存受限** (坏)")
    text("Conclusion: compute-limited iff B > 295 <br/> 结论：当且仅当 B > 295 时计算受限")

    text("Extreme case (B = 1, corresponding to matrix-vector product): <br/> 极端情况 (B = 1，对应矩阵-向量乘积)：")
    text("- Arithmetic intensity: 1 <br/> - 算术强度：1")
    text("- Memory-limited (read D x F matrix, perform only 2*D*F FLOPs) <br/> - 内存受限 (读取 D x F 矩阵，仅执行 2*D*F FLOPs)")
    text("- This is basically what happens with generation... <br/> - 这基本上就是生成时发生的情况...")


def arithmetic_intensity_of_inference():
    link(scaling_book_inference)

    image("https://jax-ml.github.io/scaling-book/assets/img/naive-inference-1400.webp", width=800)
    text("Naive inference: to generate each token, feed history into Transformer <br/> 朴素推理：为生成每个 token，将历史输入 Transformer")
    text("Complexity: generating T tokens requires O(T^3) FLOPs (one feedforward pass is O(T^2)) <br/> 复杂度：生成 T 个 token 需要 O(T^3) FLOPs (一次前向传播是 O(T^2))")

    text("Observation: a lot of the work can be shared across prefixes <br/> 观察：很多工作可以在前缀之间共享")
    text("Solution: store **KV cache** in HBM <br/> 解决方案：在 HBM 中存储 **KV 缓存**")
    image("https://jax-ml.github.io/scaling-book/assets/img/cached-inference-1400.webp", width=800)
    text("KV cache: for every sequence (B), token (S), layer (L), head (K), store an H-dimensional vector <br/> KV 缓存：对每个序列 (B)、token (S)、层 (L)、头 (K)，存储一个 H 维向量")

    text("Two stages of inference: <br/> 推理的两个阶段：")
    text("1. **Prefill**: given a prompt, encode into vectors (parallelizable like in training) <br/> 1. **预填充**：给定提示，编码为向量 (像训练一样可以并行)")
    text("2. **Generation**: generate new response tokens (sequential) <br/> 2. **生成**：生成新的响应 token (顺序)")

    text("Let's compute the FLOPs and memory IO for both the MLP and attention layers. <br/> 让我们计算 MLP 和注意力层的 FLOPs 和内存 IO。")
    text("S is the number of tokens we're conditioning on, T is the number of tokens we're generating. <br/> S 是我们条件化的 token 数量，T 是我们生成的 token 数量。")
    text("Later, we'll specialize to prefill (T = S) and generation (T = 1). <br/> 稍后，我们将专门讨论预填充 (T = S) 和生成 (T = 1)。")

    text("### MLP layers (only looking at the matrix multiplications) <br/> ### MLP 层 (仅看矩阵乘法)")
    flops = 0
    bytes_transferred = 0
    text("Steps: <br/> 步骤：")
    text("1. Read X (B x T x D) from HBM <br/> 1. 从 HBM 读取 X (B x T x D)")
    bytes_transferred += 2*B*T*D
    text("2. Read Wup (D x F), Wgate (D x F), Wdown (F x D) from HBM <br/> 2. 从 HBM 读取 Wup (D x F)、Wgate (D x F)、Wdown (F x D)")
    bytes_transferred += 3 * 2*D*F
    text("3. Compute U = X (B x T x D) @ Wup (D x F) <br/> 3. 计算 U = X (B x T x D) @ Wup (D x F)")
    flops += 2*B*T*D*F
    text("4. Write U (B x T x F) to HBM <br/> 4. 将 U (B x T x F) 写入 HBM")
    bytes_transferred += 2*B*T*F
    text("5. Compute G = X (B x T x D) @ Wgate (D x F) <br/> 5. 计算 G = X (B x T x D) @ Wgate (D x F)")
    flops += 2*B*T*D*F
    text("6. Write G (B x T x F) to HBM <br/> 6. 将 G (B x T x F) 写入 HBM")
    bytes_transferred += 2*B*T*F
    text("7. Compute Y = GeLU(G)*U (B x T x F) @ Wdown (F x D) <br/> 7. 计算 Y = GeLU(G)*U (B x T x F) @ Wdown (F x D)")
    flops += 2*B*T*D*F
    text("8. Write Y (B x T x D) to HBM <br/> 8. 将 Y (B x T x D) 写入 HBM")
    bytes_transferred += 2*B*T*D

    text("Let's take stock of the accounting results. <br/> 让我们统计一下计算结果。")
    assert flops == 6*B*T*D*F
    assert bytes_transferred == 4*B*T*D + 4*B*T*F + 6*D*F
    intensity = (flops / bytes_transferred).simplify()  # @inspect intensity
    text("Assume that B*T is much smaller than D and F. <br/> 假设 B*T 远小于 D 和 F。")
    intensity = intensity.subs(D, c*B*T).subs(F, c*B*T).limit(c, oo).simplify()  # @inspect intensity
    assert intensity == B*T

    text("For the two stages: <br/> 对于两个阶段：")
    text("1. Prefill: easy to make compute-limited (good) by making B T large enough <br/> 1. 预填充：通过使 B T 足够大，容易达到计算受限 (好)")
    text("2. Generation: <br/> 2. 生成：")
    text("- Generating one token at a time (T = 1) <br/> - 一次生成一个 token (T = 1)")
    text("- B is number of concurrent requests, hard to make large enough! <br/> - B 是并发请求数，很难做得足够大！")

    text("### Attention layers (focusing on the matrix multiplications with FlashAttention) <br/> ### 注意力层 (关注 FlashAttention 的矩阵乘法)")
    flops = 0
    bytes_transferred = 0
    text("Steps: <br/> 步骤：")
    text("1. Read Q (B x T x D), K (B x S x D), V (B x S x D) from HBM <br/> 1. 从 HBM 读取 Q (B x T x D)、K (B x S x D)、V (B x S x D)")
    bytes_transferred += 2*B*T*D + 2*B*S*D + 2*B*S*D
    text("2. Compute A = Q (B x T x D) @ K (B x S x D) <br/> 2. 计算 A = Q (B x T x D) @ K (B x S x D)")
    flops += 2*B*S*T*D
    text("3. Compute Y = softmax(A) (B x S x T x K x G) @ V (B x S x K x H) <br/> 3. 计算 Y = softmax(A) (B x S x T x K x G) @ V (B x S x K x H)")
    flops += 2*B*S*T*D
    text("4. Write Y (B x T x D) to HBM <br/> 4. 将 Y (B x T x D) 写入 HBM")
    bytes_transferred += 2*B*T*D

    assert flops == 4*B*S*T*D
    assert bytes_transferred == 4*B*S*D + 4*B*T*D
    intensity = (flops / bytes_transferred).simplify()  # @inspect intensity
    assert intensity == S*T / (S + T)

    text("For the two stages: <br/> 对于两个阶段：")
    text("1. Prefill: T = S <br/> 1. 预填充：T = S")
    prefill_intensity = intensity.subs(T, S).simplify()  # @inspect prefill_intensity
    assert prefill_intensity == S/2  # Good!
    text("2. Generation: T = 1 <br/> 2. 生成：T = 1")
    generate_intensity = intensity.subs(T, 1).simplify()  # @inspect generate_intensity
    assert generate_intensity < 1  # Bad!

    text("Unlike MLPs, no dependence on B, so batching doesn't help! <br/> 与 MLP 不同，不依赖于 B，所以批处理没有帮助！")
    text("Why? <br/> 为什么？")
    text("- In MLP layers, every sequence hits the same MLP weights (Wup, Wgate, Wdown don't depend on B) <br/> - 在 MLP 层中，每个序列都命中相同的 MLP 权重 (Wup、Wgate、Wdown 不依赖于 B)")
    text("- In attention layers, every sequence has its own vectors KV cache (Q, K, V all depend on B) <br/> - 在注意力层中，每个序列有自己的 KV 缓存向量 (Q、K、V 都依赖于 B)")

    text("Summary <br/> 总结")
    text("- Prefill is compute-limited, generation is memory-limited <br/> - 预填充是计算受限的，生成是内存受限的")
    text("- MLP intensity is B (requires concurrent requests), attention intensity is 1 (impossible to improve) <br/> - MLP 强度是 B (需要并发请求)，注意力强度是 1 (无法改进)")


def compute_transformer_stats(config):  # @inspect config
    """Return symbols corresponding to various statistics of a Transformer."""
    text("The memory, throughput, and latency depends on the shape of the Transformer. <br/> 内存、吞吐量和延迟取决于 Transformer 的形状。"), text(" "), link("")

    text("Compute the number of parameters in the Transformer: <br/> 计算 Transformer 中的参数数量：")
    num_params = 2*V*D + D*F*3*L + (2*D*N*H + 2*D*K*H)*L
    text("To store parameters, just use bf16 (training requires fp32) <br/> 存储参数只需使用 bf16 (训练需要 fp32)")
    parameter_size = num_params * 2  # 2 for bf16
    
    text("We also don't need gradients and optimizer states since we're not training. <br/> 由于我们不训练，也不需要梯度和优化器状态。")
    text("But we do have to store the KV cache (which are some of the activations) for each sequence (of length S): <br/> 但我们确实需要为每个序列 (长度为 S) 存储 KV 缓存 (这是一些激活)：")
    text("How much we have to store per sequence: <br/> 每个序列需要存储多少：")
    kv_cache_size = S * (K*H) * L * 2 * 2  # 2 for key + value, 2 for bf16

    text("Total memory usage: <br/> 总内存使用量：")
    memory = B * kv_cache_size + parameter_size
    text("Latency is determined by memory IO (read all parameters and KV cache for each step) <br/> 延迟由内存 IO 决定 (每一步读取所有参数和 KV 缓存)")
    latency = memory / memory_bandwidth
    text("Throughput is the inverse of latency, but we're generating B tokens in parallel <br/> 吞吐量是延迟的倒数，但我们在并行生成 B 个 token")
    throughput = B / latency

    # Substitute
    num_params = num_params.subs(config).simplify()  # @inspect num_params
    memory = memory.subs(config).simplify()  # @inspect memory
    latency = latency.subs(config).simplify()  # @inspect latency
    throughput = throughput.subs(config).simplify()  # @inspect throughput

    return num_params, memory, latency, throughput

def llama2_13b_config(args={}):
    return {S: 1024, D: 5120, F: 13824, N: 40, K: 40, H: 128, L: 40, V: 32000, memory_bandwidth: 3.35e12, **args}

def throughput_and_latency():
    text("So we have shown that inference is memory-limited. <br/> 所以我们已经证明推理是内存受限的。")
    text("Let us now compute the theoretical maximum latency and throughput of a single request. <br/> 现在让我们计算单个请求的理论最大延迟和吞吐量。")
    text("Assumption: can overlap compute and communication perfectly and ignore various types of overhead. <br/> 假设：可以完美重叠计算和通信，并忽略各种类型的开销。")

    text("Instantiate latency and throughput for Llama 2 13B on an H100: <br/> 实例化 H100 上 Llama 2 13B 的延迟和吞吐量：")
    config = llama2_13b_config()
    num_params, memory, latency, throughput = compute_transformer_stats(config)

    text("If we use a batch size of 1: <br/> 如果我们使用批量大小为 1：")
    bs1_memory = memory.subs(B, 1).simplify()   # @inspect bs1_memory
    bs1_latency = latency.subs(B, 1).simplify()   # @inspect bs1_latency
    bs1_throughput = throughput.subs(B, 1).simplify()   # @inspect bs1_throughput

    text("If we use a batch size of 64 (worse latency, better throughput): <br/> 如果我们使用批量大小为 64 (延迟更差，吞吐量更好)：")
    bs64_memory = memory.subs(B, 64).simplify()   # @inspect bs64_memory
    bs64_latency = latency.subs(B, 64).simplify()   # @inspect bs64_latency
    bs64_throughput = throughput.subs(B, 64).simplify()   # @inspect bs64_throughput

    text("If we use a batch size of 256: <br/> 如果我们使用批量大小为 256：")
    bs256_memory = memory.subs(B, 256).simplify()   # @inspect bs256_memory
    bs256_latency = latency.subs(B, 256).simplify()   # @inspect bs256_latency
    bs256_throughput = throughput.subs(B, 256).simplify()   # @inspect bs256_throughput
    text("Doesn't fit into memory, but throughput gains are diminishing too... <br/> 无法装入内存，但吞吐量收益也在递减...")

    text("**Tradeoff** between latency and throughput: <br/> **延迟和吞吐量之间的权衡**：")
    text("1. Smaller batch sizes yields better latency but worse throughput <br/> 1. 较小的批量大小产生更好的延迟但更差的吞吐量")
    text("2. Larger batch sizes yields better throughput but worse latency <br/> 2. 较大的批量大小产生更好的吞吐量但更差的延迟")

    text("Easy parallelism: if you launch M copies of the model, latency is the same, throughput increases by M! <br/> 简单并行：如果你启动 M 个模型副本，延迟相同，吞吐量增加 M 倍！")
    text("Harder parallelism: shard the model and the KV cache "), link(scaling_book_inference)

    text("Note: time-to-first-token (TTFT) is essentially a function of prefill <br/> 注意：首 token 时间 (TTFT) 本质上是预填充的函数")
    text("Use smaller batch sizes during prefill for faster TTFT <br/> 在预填充期间使用较小的批量大小以获得更快的 TTFT")
    text("Use larger batch sizes during generation to improve throughput <br/> 在生成期间使用较大的批量大小以提高吞吐量")


def reduce_kv_cache_size():
    text("Recall that memory is the bottleneck for inference. <br/> 回想一下，内存是推理的瓶颈。")
    text("So let's try to reduce the size of the KV cache <br/> 所以让我们尝试减少 KV 缓存的大小")
    text("...but make sure we don't lose too much accuracy. <br/> ...但要确保不会损失太多准确性。")

    text("### Grouped-query attention (GQA) <br/> ### 分组查询注意力 (GQA) "), link(gqa)
    image("https://jax-ml.github.io/scaling-book/assets/img/gmqa.png", width=800)
    text("Idea: N query heads, but only K key and value heads, each interacting with N/K query heads <br/> 想法：N 个查询头，但只有 K 个键和值头，每个与 N/K 个查询头交互")
    text("Multi-headed attention (MHA): K=N <br/> 多头注意力 (MHA)：K=N")
    text("Multi-query attention (MQA): K=1 <br/> 多查询注意力 (MQA)：K=1")
    text("Group-query attention (GQA): K is somewhere in between <br/> 分组查询注意力 (GQA)：K 介于两者之间")

    text("Latency/throughput improvements: <br/> 延迟/吞吐量改进：")
    image("images/gqa-speed.png", width=500); text(" "); link(gqa)
    text("Reduce the KV cache by a factor of N/K <br/> 将 KV 缓存减少 N/K 倍")
    config = llama2_13b_config({K: 40, B: 64})  # Original Llama 2 13B
    k40_num_params, k40_memory, k40_latency, k40_throughput = compute_transformer_stats(config)  # @inspect k40_memory, @inspect k40_latency, @inspect k40_throughput

    config = llama2_13b_config({K: 8, B: 64})  # Use GQA with 1:5 ratio
    k8_num_params, k8_memory, k8_latency, k8_throughput = compute_transformer_stats(config)  # @inspect k8_memory, @inspect k8_latency, @inspect k8_throughput

    text("This also means we can use a larger batch size: <br/> 这也意味着我们可以使用更大的批量大小：")
    config = llama2_13b_config({K: 8, B: 256})  # Increase batch size
    k8_bs_num_params, k8_bs_memory, k8_bs_latency, k8_bs_throughput = compute_transformer_stats(config)  # @inspect k8_bs_memory, @inspect k8_bs_latency, @inspect k8_bs_throughput
    text("Worse latency, but better throughput (and it fits in memory now!). <br/> 延迟更差，但吞吐量更好 (而且现在可以装入内存了！)。")

    text("Check that accuracy doesn't drop: <br/> 检查准确性是否下降："); link(gqa)
    image("images/gqa-accuracy.png", width=800)

    text("### Multi-head latent attention (MLA) <br/> ### 多头潜在注意力 (MLA) "), link(mla)
    image("images/mla-schema.png", width=800)
    text("Key idea: project down each key and value vector from N*H dimensions to C dimensions <br/> 关键思想：将每个键和值向量从 N*H 维投影到 C 维")
    text("DeepSeek v2: reduce N*H = 16384 to C = 512 <br/> DeepSeek v2：将 N*H = 16384 减少到 C = 512")
    text("Wrinkle: MLA is not compatible with RoPE, so need to add additional 64 dimensions for RoPE, so 512 + 64 = 576 total dimensions <br/> 问题：MLA 与 RoPE 不兼容，所以需要为 RoPE 增加额外的 64 维，所以总共 512 + 64 = 576 维")
    text("Latency/throughput improvements follow similarly from the KV cache reduction as argued earlier <br/> 延迟/吞吐量改进与之前讨论的 KV 缓存减少类似")

    text("Let's now check the accuracy. <br/> 现在让我们检查准确性。")
    text("First, MHA is better than GQA (though more expensive) [Table 8] <br/> 首先，MHA 比 GQA 更好 (虽然更昂贵) [表 8] "); link(mla)
    image("images/mla-accuracy.png", width=800)
    text("Second, MLA is a bit better than MHA (and much cheaper) [Table 9] <br/> 其次，MLA 比 MHA 稍好 (且便宜得多) [表 9] "); link(mla)
    image("images/mla-accuracy2.png", width=800)

    text("### Cross-layer attention (CLA) <br/> ### 跨层注意力 (CLA) "), link("https://arxiv.org/abs/2405.12981")
    image("images/cla-diagram.png", width=500)
    text("Idea: share KVs across **layers** (just as GQA shares KVs across heads) <br/> 想法：在**层**之间共享 KV (就像 GQA 在头之间共享 KV)")
    text("Empirically improves the pareto frontier of accuracy and KV cache size (latency and throughput) <br/> 经验性地改善了准确性和 KV 缓存大小 (延迟和吞吐量) 的帕累托前沿")
    image("images/cla-results.png", width=700)

    text("### Local attention <br/> ### 局部注意力 "), link(longformer), link(sparse_transformer), link(mistral_7b)
    image("images/longformer-attention.png", width=800)
    text("Idea: just look at the local context, which is most relevant for modeling <br/> 想法：只看局部上下文，这对建模最相关")
    text("Effective context scales linearly with the number of layers <br/> 有效上下文随层数线性扩展")
    text("KV cache is independent of sequence length! <br/> KV 缓存与序列长度无关！")

    text("Problem: this can still hurt accuracy <br/> 问题：这仍然可能影响准确性")
    text("Solution: interleave local attention with global attention (hybrid layers) <br/> 解决方案：将局部注意力与全局注意力交错 (混合层)")
    text("Example: character.ai uses 1 global layer every 6 layers (in addition to CLA) "), article_link("https://research.character.ai/optimizing-inference/")
    image("https://research.character.ai/content/images/2024/06/figure1-2-1.png", width=800)

    text("Summary: <br/> 总结：")
    text("- Goal: reduce the KV cache size (since inference is memory-limited) without hurting accuracy <br/> - 目标：减少 KV 缓存大小 (因为推理是内存受限的) 而不影响准确性")
    text("- Lower-dimensional KV cache (GQA, MLA, shared KV cache) <br/> - 低维 KV 缓存 (GQA、MLA、共享 KV 缓存)")
    text("- Local attention on some of the layers <br/> - 某些层上的局部注意力")


def alternatives_to_the_transformer():
    text("We have shown that tweaking the architecture of the Transformer, we can improve latency and throughput. <br/> 我们已经证明，通过调整 Transformer 的架构，我们可以改善延迟和吞吐量。")
    text("Attention + autoregression is fundamentally memory-limited (Transformers were not designed with inference in mind). <br/> 注意力 + 自回归本质上是内存受限的 (Transformer 在设计时没有考虑推理)。")
    text("Can we substantially improve things if we go beyond the Transformer? <br/> 如果我们超越 Transformer，能否大幅改善？")
    text("We will discuss two directions: state-space models and diffusion models. <br/> 我们将讨论两个方向：状态空间模型和扩散模型。")

    text("## State-space models <br/> ## 状态空间模型")
    link(title="[presentation from CS229S]", url="https://docs.google.com/presentation/d/1wrQO4uzwWr73SGj7aFxeVR9Cz0PY-mzJipn12enM39k/edit#slide=id.p")
    text("- Idea: from signal processing to model long-context sequences in a sub-quadratic time <br/> - 想法：从信号处理到以亚二次时间建模长上下文序列")
    text("- S4: based on classic state space models, good at synthetic long-context tasks "), link("https://arxiv.org/abs/2111.00396")
    image("images/s4-summary.png", width=800)
    text("- Weaknesses: bad at solving associative recall tasks important for language (where Transformers do well) <br/> - 弱点：不擅长解决对语言重要的联想回忆任务 (Transformer 做得好的地方)")
    image("images/based-associative-recall.png", width=400)
    text("- Mamba: allow SSM parameters to be input-dependent, match Transformers at 1B scale "), link("https://arxiv.org/abs/2312.00752")
    text("- Jamba: interleave Transformer-Mamba layers (1:7 ratio) with a 52B MoE "), link("https://arxiv.org/abs/2403.19887")
    image("images/jamba-architecture.png", width=400)
    text("- BASED: use linear attention + local attention "), link("https://arxiv.org/abs/2402.18668")
    image("images/based-attention.png", width=400)
    text("- MiniMax-01: use linear attention + full attention (456B parameter MoE) "), link("https://arxiv.org/pdf/2501.08313")

    text("Takeaways: <br/> 要点：")
    text("- Linear + local attention (still need some full attention) yield serious SOTA models <br/> - 线性 + 局部注意力 (仍需要一些完整注意力) 产生严肃的 SOTA 模型")
    text("- Replace O(T) KV cache with O(1) state => much more efficient for inference <br/> - 用 O(1) 状态替换 O(T) KV 缓存 => 推理效率高得多")

    text("### Diffusion models <br/> ### 扩散模型")
    text("- Popular for image generation, but harder to get working for text generation "), link("https://arxiv.org/abs/2205.14217")
    image("images/diffusion-lm.png", width=700)
    text("- Idea: generate each token in parallel (not autoregressively), refine multiple time steps <br/> - 想法：并行生成每个 token (非自回归)，多时间步细化")
    text("- Start with random noise (over entire sequence), iteratively refine it <br/> - 从随机噪声开始 (覆盖整个序列)，迭代细化")
    text("- Results from Inception Labs "), article_link("https://www.inceptionlabs.ai/news")
    link(title="[demo video]", url="https://x.com/i/status/1894847919624462794")
    text("Much faster on coding benchmarks: <br/> 在编码基准测试上快得多：")
    image("https://framerusercontent.com/images/K2zvhtaTsz5ehDFoWx6KQHOqCyk.jpg", width=800)

    text("Overall, significant gains in inference to be made with more radical architecture changes! <br/> 总的来说，通过更激进的架构改变可以在推理方面获得显著收益！")


def quantization():
    text("Key idea: reduce the precision of numbers <br/> 关键思想：降低数字精度")
    text("Less memory means higher latency/throughput (since inference is memory-limited). <br/> 更少内存意味着更高延迟/吞吐量 (因为推理是内存受限的)。")
    text("Of course we have to worry about accuracy... <br/> 当然我们必须担心准确性...")

    image("https://www.datocms-assets.com/104802/1709770809-twitter-post-20.png", width=400), article_link("https://www.baseten.co/blog/fp8-efficient-model-inference-with-8-bit-floating-point-numbers/")
    text("- fp32 (4 bytes): needed for parameters and optimizer states during training <br/> - fp32 (4 字节)：训练期间参数和优化器状态所需")
    text("- bf16 (2 bytes): default for inference <br/> - bf16 (2 字节)：推理默认值")
    text("- fp8 (1 byte) [-240, 240] for e4m3 on H100s: can train if you dare "), link("https://arxiv.org/pdf/2310.18313")
    text("- int8 (1 byte) [-128, 127]: less accurate but cheaper than fp8, but for inference only "), link("https://arxiv.org/pdf/2303.17951")
    text("- int4 (0.5 bytes) [-8, 7]: cheaper, even less accurate "), link("https://arxiv.org/pdf/2303.17951")

    text("Quantization-aware training (QAT): train with quantization, but doesn't scale up <br/> 量化感知训练 (QAT)：用量化训练，但无法扩展")
    text("Post-training quantization (PTQ): run on sample data to determine scale and zero point for each layer or tensor <br/> 训练后量化 (PTQ)：在样本数据上运行以确定每层或张量的缩放和零点")
    link(title="[Overview of approaches]", url="https://apxml.com/posts/llm-quantization-techniques-explained")

    text("### LLM.int8() <br/> ### LLM.int8()")
    link("https://arxiv.org/abs/2208.07339"), article_link("https://huggingface.co/blog/hf-bitsandbytes-integration")
    text("Standard quantization (scale by max of absolute values): <br/> 标准量化 (按最大绝对值缩放)：")
    image("https://huggingface.co/blog/assets/96_hf_bitsandbytes_integration/quant-freeze.png", width=500)
    text("Problem: outliers (which appear in larger networks) screw everything up <br/> 问题：异常值 (出现在较大网络中) 搞砸一切")
    text("Solution: extract outliers and process them in fp16 <br/> 解决方案：提取异常值并在 fp16 中处理")
    image("https://huggingface.co/blog/assets/96_hf_bitsandbytes_integration/Mixed-int8.gif", width=600)
    text("It works well (but is 15-23% slower than fp16): <br/> 它工作良好 (但比 fp16 慢 15-23%)：")
    image("images/llm-int8-bloom.png", width=500)

    text("### Activation-aware quantization <br/> ### 激活感知量化")
    link("https://arxiv.org/abs/2306.00978")
    text("Idea: select which weights (0.1-1%) to keep in high precision based on activations <br/> 想法：根据激活选择哪些权重 (0.1-1%) 保持高精度")
    text("fp16 -> int3 produces 4x lower memory, 3.2x speedup <br/> fp16 -> int3 产生 4 倍更低内存，3.2 倍加速")
    image("images/awq-schema.png", width=800)


def model_pruning():
    text("Key idea: just rip out parts of an expensive model to make it cheaper <br/> 关键思想：直接移除昂贵模型的部分使其更便宜")
    text("...and then fix it up. <br/> ...然后修复它。")

    text("Paper from NVIDIA "), link("https://arxiv.org/abs/2407.14679")
    image("images/pruning-kd-loop.png", width=600)
    text("Algorithm: <br/> 算法：")
    text("1. Identify important {layer, head, hidden dimension} on a small calibration dataset (1024 samples) <br/> 1. 在小校准数据集 (1024 样本) 上识别重要的 {层、头、隐藏维度}")
    text("2. Remove unimportant layers to get a smaller model <br/> 2. 移除不重要的层以获得更小的模型")
    text("3. Distill the original model into pruned model <br/> 3. 将原始模型蒸馏到剪枝模型")

    text("Results: <br/> 结果：")
    image("images/pruning-kd.png", width=500)


def speculative_sampling():
    text("Recall the two stages of inference: <br/> 回想推理的两个阶段：")
    text("- Prefill: given a sequence, encode tokens in parallel (compute-limited) [note: also gives you probabilities] <br/> - 预填充：给定序列，并行编码 token (计算受限) [注：也给你概率]")
    text("- Generation: generate one token at a time (memory-limited) <br/> - 生成：一次生成一个 token (内存受限)")
    text("In other words, checking is faster than generation. <br/> 换句话说，检查比生成快。")

    text("Speculative sampling <br/> 投机采样 "); link("https://arxiv.org/abs/2211.17192"); link("https://arxiv.org/abs/2302.01318")
    text("- Use a cheaper **draft model** p to guess a few tokens (e.g., 4) <br/> - 使用更便宜的**草稿模型** p 猜测几个 token (如 4)")
    text("- Evaluate with target model q (process tokens in parallel), and accept if it looks good <br/> - 用目标模型 q 评估 (并行处理 token)，如果看起来不错就接受")
    link(title="[Speculative sampling video]", url="https://storage.googleapis.com/gweb-research2023-media/media/SpeculativeDecoding-1-Illustration.mp4")
    article_link("https://research.google/blog/looking-back-at-speculative-decoding/")

    image("images/speculative-sampling-algorithm.png", width=600)
    text("This is modified rejection sampling with proposal p and target q <br/> 这是用提议 p 和目标 q 的改进拒绝采样")
    text("Modification: always generate at least one candidate (rejection sampling will keep looping) <br/> 修改：始终生成至少一个候选 (拒绝采样会一直循环)")
    text("Key property: guaranteed to be an **exact sample** from the target model! <br/> 关键属性：保证是来自目标模型的**精确样本**！")

    text("Proof by example: assume two vocabulary elements {A, B} <br/> 示例证明：假设两个词汇元素 {A, B}")
    text("- Target model probabilities: [q(A), q(B)] <br/> - 目标模型概率：[q(A), q(B)]")
    text("- Draft model probabilities: [p(A), p(B)] <br/> - 草稿模型概率：[p(A), p(B)]")
    text("- Assume p(A) > q(A) [draft model oversamples A]. <br/> - 假设 p(A) > q(A) [草稿模型过度采样 A]。")
    text("- Therefore p(B) < q(B) [draft model undersamples B]. <br/> - 因此 p(B) < q(B) [草稿模型欠采样 B]。")
    text("- Residual probabilities max(q-p, 0): [0, 1] <br/> - 残差概率 max(q-p, 0)：[0, 1]")
    text("Compute the probabilities of speculatively sampling a token: <br/> 计算投机采样 token 的概率：")
    text("- P[sampling A] = p(A) * (q(A) / p(A)) + p(B) * 1 * 0 = q(A) <br/> - P[采样 A] = p(A) * (q(A) / p(A)) + p(B) * 1 * 0 = q(A)")
    text("- P[sampling B] = p(B) * 1 + p(A) * (1 - q(A) / p(A)) * 1 = q(B) <br/> - P[采样 B] = p(B) * 1 + p(A) * (1 - q(A) / p(A)) * 1 = q(B)")

    image("images/speculative-sampling-results.png", width=600)
    image("images/speculative-sampling-stats.png", width=600)

    text("In practice: <br/> 在实践中：")
    text("- Target model has 70B parameters, draft model has 8B parameters <br/> - 目标模型有 70B 参数，草稿模型有 8B 参数")
    text("- Target model has 8B parameters, draft model has 1B parameters <br/> - 目标模型有 8B 参数，草稿模型有 1B 参数")
    text("- Try to make draft model as close to target (distillation) <br/> - 尝试使草稿模型尽可能接近目标 (蒸馏)")

    text("Extensions to improve the draft model: <br/> 改进草稿模型的扩展：")
    text("- Medusa: draft model generates multiple tokens in parallel "), link("https://arxiv.org/abs/2401.10774")
    text("- EAGLE: draft model takes high-level features from target model "), link("https://arxiv.org/pdf/2401.15077")
    image("images/medusa-eagle.png", width=600)

    text("Summary: <br/> 总结：")
    text("- Exact sampling from target model (thanks to math)! <br/> - 来自目标模型的精确采样 (感谢数学)！")
    text("- Exploits asymmetry between checking and generation <br/> - 利用检查和生成之间的不对称性")
    text("- Lots of room for innovation on the draft model (involves training) <br/> - 草稿模型有很多创新空间 (涉及训练)")


def continuous_batching():
    link(title="Orca: A Distributed Serving System for Transformer-Based Generative Models", url="https://www.usenix.org/system/files/osdi22-yu.pdf"), link(title="[talk]", url="https://www.youtube.com/watch?v=Ob9PPLxETYU")

    text("Problem: <br/> 问题：")
    text("- Training: get a dense block of tokens (batch size x sequence length) <br/> - 训练：获得密集的 token 块 (批量大小 x 序列长度)")
    text("- Inference: requests arrive and finish at different times, so you have a ragged array <br/> - 推理：请求在不同时间到达和完成，所以你有一个不规则数组")
    image("https://images.ctfassets.net/xjan103pcp94/1LJioEsEdQQpDCxYNWirU6/82b9fbfc5b78b10c1d4508b60e72fdcf/cb_02_diagram-static-batching.png", width=600)

    text("Solution: iteration-level scheduling <br/> 解决方案：迭代级调度")
    text("- Decode step by step <br/> - 逐步解码")
    text("- Add new requests to the batch as they arrive (so don't have to wait until generation completes) <br/> - 新请求到达时添加到批次 (所以不必等到生成完成)")

    text("Problem: <br/> 问题：")
    text("- Batching only works when all sequences have the same dimensionality (right?) <br/> - 批处理仅在所有序列具有相同维度时才有效 (对吗？)")
    text("- But each request might have a different length <br/> - 但每个请求可能有不同的长度")

    text("Solution: selective batching <br/> 解决方案：选择性批处理")
    text("- Training: when all sequences of the same length, operate on a B x S x H tensor <br/> - 训练：当所有序列长度相同时，在 B x S x H 张量上操作")
    text("- But we might have different lengths: [3, H], [9, H], [5, H], etc. <br/> - 但我们可能有不同的长度：[3, H]、[9, H]、[5, H] 等")
    text("- Attention computation: process each sequence separately <br/> - 注意力计算：分别处理每个序列")
    text("- Non-attention computation: concatenate all the sequences together to [3 + 9 + 5, H] <br/> - 非注意力计算：将所有序列连接在一起为 [3 + 9 + 5, H]")


def paged_attention():
    text("Paper that introduced vLLM in addition to PagedAttention <br/> 介绍 vLLM 和 PagedAttention 的论文 "), link("https://arxiv.org/pdf/2309.06180.pdf")

    text("Previous status quo: <br/> 之前的状态：")
    text("- Request comes in <br/> - 请求进来")
    text("- Allocate section of KV cache for prompt and response (up to a max length) <br/> - 为提示和响应分配 KV 缓存段 (最大长度)")
    image("images/paged-attention-fragmentation.png", width=800)
    text("Problem: fragmentation (what happens to your hard drive) <br/> 问题：碎片 (就像你的硬盘)")
    text("- But this is wasteful since we might generate much fewer tokens (internal fragmentation)! <br/> - 但这是浪费的，因为我们可能生成更少的 token (内部碎片)！")
    text("- Might be extra unused space between sections (external fragmentation)! <br/> - 段之间可能有额外的未使用空间 (外部碎片)！")

    text("Solution: PagedAttention (remember operating systems) <br/> 解决方案：PagedAttention (记得操作系统)")
    text("- Divide the KV cache of a sequence into non-contiguous **blocks** <br/> - 将序列的 KV 缓存划分为不连续的**块**")
    image("images/paged-attention-blocks.png", width=400)

    text("Two requests share the KV caches: <br/> 两个请求共享 KV 缓存：")
    image("images/paged-attention-logical.png", width=800)

    text("In general, multiples types of sharing KV caches across sequences: <br/> 一般来说，跨序列共享 KV 缓存的多种类型：")
    image("images/paged-attention-sharing.png", width=600)
    text("- Sharing the system prompt <br/> - 共享系统提示")
    text("- Sampling multiple responses per prompt (e.g., for program synthesis) <br/> - 每个提示采样多个响应 (如用于程序合成)")

    text("Solution: share prefixes, copy-on-write at the block level <br/> 解决方案：共享前缀，块级写时复制")
    image("images/paged-attention-parallel.png", width=600)

    text("Other vLLM optimizations: <br/> 其他 vLLM 优化：")
    text("- Kernel to fuse block read and attention (reduce kernel launch overhead) <br/> - 融合块读取和注意力的内核 (减少内核启动开销)")
    text("- Use latest kernels (FlashAttention, FlashDecoding) <br/> - 使用最新内核 (FlashAttention、FlashDecoding)")
    text("- Use CUDA graphs to avoid kernel launch overhead <br/> - 使用 CUDA 图避免内核启动开销")

    text("Summary: use ideas from operating systems (paging) to make use of memory for dynamic workloads <br/> 总结：使用操作系统的思路 (分页) 来利用内存处理动态工作负载")


if __name__ == "__main__":
    main()
