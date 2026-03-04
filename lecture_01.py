import regex
from abc import ABC
from dataclasses import dataclass
from collections import defaultdict
import random

from execute_util import link, image, text
from lecture_util import article_link, x_link, youtube_link
from references import gpt_3, gpt4, shannon1950, bengio2003, susketver2014, \
    bahdanau2015_attention, transformer_2017, gpt2, t5, kaplan_scaling_laws_2020, \
    the_pile, gpt_j, opt_175b, bloom, palm, chinchilla, llama, mistral_7b, \
    instruct_gpt, dpo, adamw2017, lima, deepseek_v3, adam2014, grpo, ppo2017, muon, \
    large_batch_training_2018, wsd_2024, cosine_learning_rate_2017, olmo_7b, moe_2017, \
    megatron_lm_2019, shazeer_2020, elmo, bert, qwen_2_5, deepseek_r1, moe_2017, \
    rms_norm_2019, rope_2021, soap, gqa, mla, deepseek_67b, deepseek_v2, brants2007, \
    layernorm_2016, pre_post_norm_2020, llama2, llama3, olmo2, \
    megabyte, byt5, blt, tfree, sennrich_2016, zero_2019, gpipe_2018
from data import get_common_crawl_urls, read_common_crawl, write_documents, markdownify_documents
from model_util import query_gpt4o

import tiktoken

def main():
    welcome()
    why_this_course_exists()
    current_landscape()

    what_is_this_program()

    course_logistics()
    course_components()

    tokenization()

    text("Next time: PyTorch building blocks, resource accounting <br/> 下次内容：PyTorch 构建模块、资源核算。")


def welcome():
    text("## CS336: Language Models From Scratch (Spring 2025) <br/> CS336：从零构建语言模型（2025年春季）"),

    image("images/course-staff.png", width=600)

    text("This is the second offering of CS336. <br/> 这是 CS336 的第二次开课。")
    text("Stanford edition has grown by 50%. <br/> 斯坦福版本增长了 50%。")
    text("Lectures will be posted on YouTube and be made available to the whole world. <br/> 课程将发布到 YouTube，向全世界开放。")


def why_this_course_exists():
    text("## Why did we make this course? <br/> 我们为什么要开设这门课？")

    text("Let's ask GPT-4 <br/> 让我们问问 GPT-4 "), link(gpt4)
    response = query_gpt4o(prompt="Why teach a course on building language models from scratch? Answer in one sentence.")  # @inspect response
    
    text("Problem: researchers are becoming **disconnected** from the underlying technology. <br/> 问题：研究人员正在与底层技术**脱节**。")
    text("8 years ago, researchers would implement and train their own models. <br/> 8 年前，研究人员会自己实现和训练模型。")
    text("6 years ago, researchers would download a model (e.g., BERT) and fine-tune it. <br/> 6 年前，研究人员会下载模型（如 BERT）并进行微调。")
    text("Today, researchers just prompt a proprietary model (e.g., GPT-4/Claude/Gemini). <br/> 如今，研究人员只是在提示闭源模型（如 GPT-4/Claude/Gemini）。")

    text("Moving up levels of abstractions boosts productivity, but <br/> 提升抽象层次能提高生产力，但")
    text("- These abstractions are leaky (in contrast to programming languages or operating systems). <br/> - 这些抽象是有漏洞的（与编程语言或操作系统不同）。")
    text("- There is still fundamental research to be done that require tearing up the stack. <br/> - 仍有需要打破整个技术栈才能完成的基础研究。")

    text("**Full understanding** of this technology is necessary for **fundamental research**. <br/> **深入理解**这项技术对于**基础研究**是必要的。")

    text("This course: **understanding via building** <br/> 这门课：**通过构建来理解**")
    text("But there's one small problem... <br/> 但有一个小问题...")

    text("## The industrialization of language models <br/> 语言模型的工业化")
    image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/cc/Industrialisation.jpg/440px-Industrialisation.jpg", width=400)

    text("GPT-4 supposedly has 1.8T parameters. <br/> GPT-4 据称有 1.8T 参数。"), article_link("https://www.hpcwire.com/2024/03/19/the-generative-ai-future-is-now-nvidias-huang-says")
    text("GPT-4 supposedly cost $100M to train. <br/> GPT-4 据称训练成本为 1 亿美元。"), article_link("https://www.wired.com/story/openai-ceo-sam-altman-the-age-of-giant-ai-models-is-already-over/")
    text("xAI builds cluster with 200,000 H100s to train Grok. <br/> xAI 构建了 200,000 个 H100 的集群来训练 Grok。"), article_link("https://www.tomshardware.com/pc-components/gpus/elon-musk-is-doubling-the-worlds-largest-ai-gpu-cluster-expanding-colossus-gpu-cluster-to-200-000-soon-has-floated-300-000-in-the-past")
    text("Stargate (OpenAI, NVIDIA, Oracle) invests $500B over 4 years. <br/> Stargate（OpenAI、NVIDIA、Oracle）在 4 年内投资 5000 亿美元。"), article_link("https://openai.com/index/announcing-the-stargate-project/")

    text("Also, there are no public details on how frontier models are built. <br/> 此外，前沿模型的构建方式没有公开细节。")
    text("From the GPT-4 technical report <br/> 来自 GPT-4 技术报告 "), link(gpt4), text(":")
    image("images/gpt4-no-details.png", width=600)

    text("## More is different <br/> 越多越不同")
    text("Frontier models are out of reach for us. <br/> 前沿模型对我们来说遥不可及。")
    text("But building small language models (<1B parameters in this class) might not be representative of large language models. <br/> 但构建小型语言模型（本课程中 <1B 参数）可能无法代表大型语言模型。")

    text("Example 1: fraction of FLOPs spent in attention versus MLP changes with scale. <br/> 示例 1：注意力与 MLP 的 FLOPs 占比随规模变化。"), x_link("https://x.com/stephenroller/status/1579993017234382849")
    image("images/roller-flops.png", width=400)
    text("Example 2: emergence of behavior with scale <br/> 示例 2：行为随规模涌现 "), link("https://arxiv.org/pdf/2206.07682")
    image("images/wei-emergence-plot.png", width=600)

    text("## What can we learn in this class that transfers to frontier models? <br/> 在这门课中我们能学到什么可以迁移到前沿模型的知识？")
    text("There are three types of knowledge: <br/> 有三类知识：")
    text("- **Mechanics**: how things work (what a Transformer is, how model parallelism leverages GPUs) <br/> - **机制**：事物如何运作（什么是 Transformer，模型并行如何利用 GPU）")
    text("- **Mindset**: squeezing the most out of the hardware, taking scale seriously (scaling laws) <br/> - **思维方式**：最大化利用硬件，认真对待规模（缩放定律）")
    text("- **Intuitions**: which data and modeling decisions yield good accuracy <br/> - **直觉**：哪些数据和建模决策能带来好的准确率")

    text("We can teach mechanics and mindset (these do transfer). <br/> 我们可以教授机制和思维方式（这些是可迁移的）。")
    text("We can only partially teach intuitions (do not necessarily transfer across scales). <br/> 我们只能部分教授直觉（不一定能跨规模迁移）。")

    text("## Intuitions? 🤷 <br/> 直觉？🤷")
    text("Some design decisions are simply not (yet) justifiable and just come from experimentation. <br/> 一些设计决策（目前）无法解释，只是来自实验。")
    text("Example: Noam Shazeer paper that introduced SwiGLU <br/> 示例：Noam Shazeer 引入 SwiGLU 的论文 "), link(shazeer_2020)
    image("images/divine-benevolence.png", width=600)

    text("## The bitter lesson <br/> 苦涩的教训")
    text("Wrong interpretation: scale is all that matters, algorithms don't matter. <br/> 错误解读：只有规模重要，算法不重要。")
    text("Right interpretation: algorithms that scale is what matters. <br/> 正确解读：能够扩展的算法才是重要的。")
    text("### accuracy = efficiency x resources <br/> ### 准确率 = 效率 × 资源")
    text("In fact, efficiency is way more important at larger scale (can't afford to be wasteful). <br/> 事实上，在更大规模下效率更加重要（无法承受浪费）。")
    link("https://arxiv.org/abs/2005.04305"), text(" showed 44x algorithmic efficiency on ImageNet between 2012 and 2019 <br/> 展示了 2012 年至 2019 年间 ImageNet 上 44 倍的算法效率提升")

    text("Framing: what is the best model one can build given a certain compute and data budget? <br/> 问题框架：在给定的计算和数据预算下，能构建的最佳模型是什么？")
    text("In other words, **maximize efficiency**! <br/> 换句话说，**最大化效率**！")


def current_landscape():
    text("## Pre-neural (before 2010s) <br/> 神经网络之前（2010 年代之前）")
    text("- Language model to measure the entropy of English <br/> - 用语言模型测量英语的熵 "), link(shannon1950)
    text("- Lots of work on n-gram language models (for machine translation, speech recognition) <br/> - 大量 n-gram 语言模型的工作（用于机器翻译、语音识别）"), link(brants2007)

    text("## Neural ingredients (2010s) <br/> 神经网络要素（2010 年代）")
    text("- First neural language model <br/> - 第一个神经语言模型 "), link(bengio2003)
    text("- Sequence-to-sequence modeling (for machine translation) <br/> - 序列到序列建模（用于机器翻译）"), link(susketver2014)
    text("- Adam optimizer <br/> - Adam 优化器 "), link(adam2014)
    text("- Attention mechanism (for machine translation) <br/> - 注意力机制（用于机器翻译）"), link(bahdanau2015_attention)
    text("- Transformer architecture (for machine translation) <br/> - Transformer 架构（用于机器翻译）"), link(transformer_2017)
    text("- Mixture of experts <br/> - 混合专家 "), link(moe_2017)
    text("- Model parallelism <br/> - 模型并行 "), link(gpipe_2018), link(zero_2019), link(megatron_lm_2019)

    text("## Early foundation models (late 2010s) <br/> 早期基础模型（2010 年代后期）")
    text("- ELMo: pretraining with LSTMs, fine-tuning helps tasks <br/> - ELMo：使用 LSTM 预训练，微调有助于任务 "), link(elmo)
    text("- BERT: pretraining with Transformer, fine-tuning helps tasks <br/> - BERT：使用 Transformer 预训练，微调有助于任务 "), link(bert)
    text("- Google's T5 (11B): cast everything as text-to-text <br/> - Google 的 T5（11B）：将所有内容转换为文本到文本 "), link(t5)

    text("## Embracing scaling, more closed <br/> 拥抱规模化，更加封闭")
    text("- OpenAI's GPT-2 (1.5B): fluent text, first signs of zero-shot, staged release <br/> - OpenAI 的 GPT-2（1.5B）：流畅的文本，首次出现零样本迹象，分阶段发布 "), link(gpt2)
    text("- Scaling laws: provide hope / predictability for scaling <br/> - 缩放定律：为规模化提供希望/可预测性 "), link(kaplan_scaling_laws_2020)
    text("- OpenAI's GPT-3 (175B): in-context learning, closed <br/> - OpenAI 的 GPT-3（175B）：上下文学习，闭源 "), link(gpt_3)
    text("- Google's PaLM (540B): massive scale, undertrained <br/> - Google 的 PaLM（540B）：大规模，训练不足 "), link(palm)
    text("- DeepMind's Chinchilla (70B): compute-optimal scaling laws <br/> - DeepMind 的 Chinchilla（70B）：计算最优的缩放定律 "), link(chinchilla)

    text("## Open models <br/> 开放模型")
    text("- EleutherAI's open datasets (The Pile) and models (GPT-J) <br/> - EleutherAI 的开放数据集（The Pile）和模型（GPT-J）"), link(the_pile), link(gpt_j)
    text("- Meta's OPT (175B): GPT-3 replication, lots of hardware issues <br/> - Meta 的 OPT（175B）：GPT-3 复制，大量硬件问题 "), link(opt_175b)
    text("- Hugging Face / BigScience's BLOOM: focused on data sourcing <br/> - Hugging Face / BigScience 的 BLOOM：专注于数据来源 "), link(bloom)
    text("- Meta's Llama models <br/> - Meta 的 Llama 模型 "), link(llama), link(llama2), link(llama3)
    text("- Alibaba\'s Qwen models <br/> - 阿里巴巴的 Qwen 模型 "), link(qwen_2_5)
    text("- DeepSeek\'s models <br/> - DeepSeek 的模型 "), link(deepseek_67b), link(deepseek_v2), link(deepseek_v3)
    text("- AI2's OLMo 2 <br/> - AI2 的 OLMo 2 "), link(olmo_7b), link(olmo2),

    text("## Levels of openness <br/> 开放程度")
    text("- Closed models (e.g., GPT-4o): API access only <br/> - 闭源模型（如 GPT-4o）：仅 API 访问 "), link(gpt4)
    text("- Open-weight models (e.g., DeepSeek): weights available, paper with architecture details, some training details, no data details <br/> - 开放权重模型（如 DeepSeek）：权重可用，论文包含架构细节、部分训练细节，无数据细节 "), link(deepseek_v3)
    text("- Open-source models (e.g., OLMo): weights and data available, paper with most details (but not necessarily the rationale, failed experiments) <br/> - 开源模型（如 OLMo）：权重和数据可用，论文包含大部分细节（但不一定包含原因和失败实验）"), link(olmo_7b)

    text("## Today's frontier models <br/> 当今的前沿模型")
    text("- OpenAI's o3 <br/> - OpenAI 的 o3 "), link("https://openai.com/index/openai-o3-mini/")
    text("- Anthropic's Claude Sonnet 3.7 <br/> - Anthropic 的 Claude Sonnet 3.7 "), link("https://www.anthropic.com/news/claude-3-7-sonnet")
    text("- xAI's Grok 3 <br/> - xAI 的 Grok 3 "), link("https://x.ai/news/grok-3")
    text("- Google's Gemini 2.5 <br/> - Google 的 Gemini 2.5 "), link("https://blog.google/technology/google-deepmind/gemini-model-thinking-updates-march-2025/")
    text("- Meta's Llama 3.3 <br/> - Meta 的 Llama 3.3 "), link("https://ai.meta.com/blog/meta-llama-3/")
    text("- DeepSeek's r1 <br/> - DeepSeek 的 r1 "), link(deepseek_r1)
    text("- Alibaba's Qwen 2.5 Max <br/> - 阿里巴巴的 Qwen 2.5 Max "), link("https://qwenlm.github.io/blog/qwen2.5-max/")
    text("- Tencent's Hunyuan-T1 <br/> - 腾讯的 Hunyuan-T1 "), link("https://tencent.github.io/llm.hunyuan.T1/README_EN.html")


def what_is_this_program():
    text("This is an *executable lecture*, a program whose execution delivers the content of a lecture. <br/> 这是一个*可执行讲义*，一个通过执行来呈现讲座内容的程序。")
    text("Executable lectures make it possible to: <br/> 可执行讲义使得以下成为可能：")
    text("- view and run code (since everything is code!), <br/> - 查看和运行代码（因为一切都是代码！），")
    total = 0  # @inspect total
    for x in [1, 2, 3]:  # @inspect x
        total += x  # @inspect total
    text("- see the hierarchical structure of the lecture, and <br/> - 查看讲座的层次结构，以及")
    text("- jump to definitions and concepts: <br/> - 跳转到定义和概念："), link(supervised_finetuning)


def course_logistics():
    text("All information online: <br/> 所有信息在线："), link("https://stanford-cs336.github.io/spring2025/")

    text("This is a 5-unit class. <br/> 这是一门 5 学分的课程。")
    text("Comment from Spring 2024 course evaluation: *The entire assignment was approximately the same amount of work as all 5 assignments from CS 224n plus the final project. And that's just the first homework assignment.* <br/> 来自 2024 年春季课程评价的评论：*整个作业的工作量大约相当于 CS 224n 的所有 5 次作业加上期末项目。而这只是第一次作业。*")

    text("## Why you should take this course <br/> 为什么你应该选这门课")
    text("- You have an obsessive need to understand how things work. <br/> - 你有一种执着的需求，想要理解事物是如何运作的。")
    text("- You want to build up your research engineering muscles. <br/> - 你想要锻炼你的研究工程能力。")

    text("## Why you should not take this course <br/> 为什么你不应该选这门课")
    text("- You actually want to get research done this quarter.<br>(Talk to your advisor.) <br/> - 你这个学期真的想完成研究。<br>（和你的导师谈谈。）")
    text("- You are interested in learning about the hottest new techniques in AI (e.g., multimodality, RAG, etc.).<br>(You should take a seminar class for that.) <br/> - 你对学习 AI 中最热门的新技术感兴趣（如多模态、RAG 等）。<br>（你应该选一门研讨课。）")
    text("- You want to get good results on your own application domain.<br>(You should just prompt or fine-tune an existing model.) <br/> - 你想在自己的应用领域获得好的结果。<br>（你应该直接提示或微调现有模型。）")

    text("## How you can follow along at home <br/> 如何在家自学")
    text("- All lecture materials and assignments will be posted online, so feel free to follow on your own. <br/> - 所有讲座材料和作业都会在线发布，所以你可以自己跟进。")
    text("- Lectures are recorded via [CGOE, formally SCPD](https://cgoe.stanford.edu/) and be made available on YouTube (with some lag). <br/> - 讲座通过 [CGOE，原名 SCPD](https://cgoe.stanford.edu/) 录制，并在 YouTube 上发布（有一些延迟）。")
    text("- We plan to offer this class again next year. <br/> - 我们计划明年再次开设这门课。")

    text("## Assignments <br/> 作业")
    text("- 5 assignments (basics, systems, scaling laws, data, alignment). <br/> - 5 次作业（基础、系统、缩放定律、数据、对齐）。")
    text("- No scaffolding code, but we provide unit tests and adapter interfaces to help you check correctness. <br/> - 没有脚手架代码，但我们提供单元测试和适配器接口来帮助你检查正确性。")
    text("- Implement locally to test for correctness, then run on cluster for benchmarking (accuracy and speed). <br/> - 在本地实现以测试正确性，然后在集群上运行进行基准测试（准确率和速度）。")
    text("- Leaderboard for some assignments (minimize perplexity given training budget). <br/> - 部分作业有排行榜（在给定训练预算下最小化困惑度）。")
    text("- AI tools (e.g., CoPilot, Cursor) can take away from learning, so use at your own risk. <br/> - AI 工具（如 CoPilot、Cursor）可能会削弱学习效果，请自行斟酌使用。")

    text("## Cluster <br/> 集群")
    text("- Thanks to Together AI for providing a compute cluster. 🙏 <br/> - 感谢 Together AI 提供计算集群。🙏")
    text("- Please read [the guide](https://docs.google.com/document/d/1BSSig7zInyjDKcbNGftVxubiHlwJ-ZqahQewIzBmBOo/edit) on how to use the cluster. <br/> - 请阅读[使用指南](https://docs.google.com/document/d/1BSSig7zInyjDKcbNGftVxubiHlwJ-ZqahQewIzBmBOo/edit)了解如何使用集群。")
    text("- Start your assignments early, since the cluster will fill up close to the deadline! <br/> - 尽早开始作业，因为集群在截止日期前会变得很繁忙！")


def course_components():
    text("## It's all about efficiency <br/> 一切都是关于效率")
    text("Resources: data + hardware (compute, memory, communication bandwidth) <br/> 资源：数据 + 硬件（计算、内存、通信带宽）")
    text("How do you train the best model given a fixed set of resources? <br/> 在给定固定资源的情况下，如何训练出最好的模型？")
    text("Example: given a Common Crawl dump and 32 H100s for 2 weeks, what should you do? <br/> 示例：给定一个 Common Crawl 数据转储和 32 个 H100 用 2 周，你应该怎么做？")

    text("Design decisions: <br/> 设计决策：")
    image("images/design-decisions.png", width=800)

    text("## Overview of the course <br/> 课程概览")
    basics()
    systems()
    scaling_laws()
    data()
    alignment()

    text("## Efficiency drives design decisions <br/> 效率驱动设计决策")

    text("Today, we are compute-constrained, so design decisions will reflect squeezing the most out of given hardware. <br/> 如今，我们受到计算资源的限制，因此设计决策将反映出如何最大化利用给定的硬件。")
    text("- Data processing: avoid wasting precious compute updating on bad / irrelevant data <br/> - 数据处理：避免在低质量/无关数据上浪费宝贵的计算资源")
    text("- Tokenization: working with raw bytes is elegant, but compute-inefficient with today's model architectures. <br/> - 分词：使用原始字节很优雅，但对于当今的模型架构来说计算效率低。")
    text("- Model architecture: many changes motivated by reducing memory or FLOPs (e.g., sharing KV caches, sliding window attention) <br/> - 模型架构：许多变化是为了减少内存或 FLOPs（如共享 KV 缓存、滑动窗口注意力）")
    text("- Training: we can get away with a single epoch! <br/> - 训练：我们只需要一个 epoch！")
    text("- Scaling laws: use less compute on smaller models to do hyperparameter tuning <br/> - 缩放定律：在较小模型上使用较少计算进行超参数调优")
    text("- Alignment: if tune model more to desired use cases, require smaller base models <br/> - 对齐：如果将模型更多地调优到期望的用例，则需要更小的基础模型")

    text("Tomorrow, we will become data-constrained... <br/> 明天，我们将变成数据受限...")


class Tokenizer(ABC):
    """Abstract interface for a tokenizer."""
    def encode(self, string: str) -> list[int]:
        raise NotImplementedError

    def decode(self, indices: list[int]) -> str:
        raise NotImplementedError


def basics():
    text("Goal: get a basic version of the full pipeline working <br/> 目标：让完整流程的基本版本运行起来")
    text("Components: tokenization, model architecture, training <br/> 组件：分词、模型架构、训练")

    text("## Tokenization <br/> 分词")
    text("Tokenizers convert between strings and sequences of integers (tokens) <br/> 分词器在字符串和整数序列（token）之间转换")
    image("images/tokenized-example.png", width=600) 
    text("Intuition: break up string into popular segments <br/> 直觉：将字符串分割成常见的片段")

    text("This course: Byte-Pair Encoding (BPE) tokenizer <br/> 本课程：字节对编码（BPE）分词器 "), link(sennrich_2016)

    text("Tokenizer-free approaches: <br/> 无分词器方法："), link(byt5), link(megabyte), link(blt), link(tfree)
    text("Use bytes directly, promising, but have not yet been scaled up to the frontier. <br/> 直接使用字节，很有前景，但尚未扩展到前沿规模。")
    
    text("## Architecture <br/> 架构")
    text("Starting point: original Transformer <br/> 起点：原始 Transformer "), link(transformer_2017)
    image("images/transformer-architecture.png", width=500)

    text("Variants: <br/> 变体：")
    text("- Activation functions: ReLU, SwiGLU <br/> - 激活函数：ReLU、SwiGLU "), link(shazeer_2020)
    text("- Positional encodings: sinusoidal, RoPE <br/> - 位置编码：正弦、RoPE "), link(rope_2021)
    text("- Normalization: LayerNorm, RMSNorm <br/> - 归一化：LayerNorm、RMSNorm "), link(layernorm_2016), link(rms_norm_2019)
    text("- Placement of normalization: pre-norm versus post-norm <br/> - 归一化位置：前置归一化 vs 后置归一化 "), link(pre_post_norm_2020)
    text("- MLP: dense, mixture of experts <br/> - MLP：密集、混合专家 "), link(moe_2017)
    text("- Attention: full, sliding window, linear <br/> - 注意力：全注意力、滑动窗口、线性 "), link(mistral_7b), link("https://arxiv.org/abs/2006.16236")
    text("- Lower-dimensional attention: group-query attention (GQA), multi-head latent attention (MLA) <br/> - 低维注意力：分组查询注意力（GQA）、多头潜在注意力（MLA）"), link(gqa), link(mla)
    text("- State-space models: Hyena <br/> - 状态空间模型：Hyena "), link("https://arxiv.org/abs/2302.10866")

    text("## Training <br/> 训练")
    text("- Optimizer (e.g., AdamW, Muon, SOAP) <br/> - 优化器（如 AdamW、Muon、SOAP）"), link(adam2014), link(adamw2017), link(muon), link(soap)
    text("- Learning rate schedule (e.g., cosine, WSD) <br/> - 学习率调度（如余弦、WSD）"), link(cosine_learning_rate_2017), link(wsd_2024)
    text("- Batch size (e..g, critical batch size) <br/> - 批次大小（如临界批次大小）"), link(large_batch_training_2018)
    text("- Regularization (e.g., dropout, weight decay) <br/> - 正则化（如 dropout、权重衰减）")
    text("- Hyperparameters (number of heads, hidden dimension): grid search <br/> - 超参数（头数、隐藏维度）：网格搜索")

    text("## Assignment 1 <br/> 作业 1")
    link(title="[GitHub]", url="https://github.com/stanford-cs336/assignment1-basics"), link(title="[PDF]", url="https://github.com/stanford-cs336/assignment1-basics/blob/main/cs336_spring2025_assignment1_basics.pdf")
    text("- Implement BPE tokenizer <br/> - 实现 BPE 分词器")
    text("- Implement Transformer, cross-entropy loss, AdamW optimizer, training loop <br/> - 实现 Transformer、交叉熵损失、AdamW 优化器、训练循环")
    text("- Train on TinyStories and OpenWebText <br/> - 在 TinyStories 和 OpenWebText 上训练")
    text("- Leaderboard: minimize OpenWebText perplexity given 90 minutes on a H100 <br/> - 排行榜：在 H100 上用 90 分钟最小化 OpenWebText 困惑度 "), link(title="[last year's leaderboard]", url="https://github.com/stanford-cs336/spring2024-assignment1-basics-leaderboard")


def systems():
    text("Goal: squeeze the most out of the hardware <br/> 目标：最大化利用硬件")
    text("Components: kernels, parallelism, inference <br/> 组件：内核、并行、推理")

    text("## Kernels <br/> 内核")
    text("What a GPU (A100) looks like: <br/> GPU（A100）的样子：")
    image("https://miro.medium.com/v2/resize:fit:2000/format:webp/1*6xoBKi5kL2dZpivFe1-zgw.jpeg", width=800)
    text("Analogy: warehouse : DRAM :: factory : SRAM <br/> 类比：仓库 : DRAM :: 工厂 : SRAM")
    image("https://horace.io/img/perf_intro/factory_bandwidth.png", width=400)
    text("Trick: organize computation to maximize utilization of GPUs by minimizing data movement <br/> 技巧：组织计算以通过最小化数据移动来最大化 GPU 利用率")
    text("Write kernels in CUDA/**Triton**/CUTLASS/ThunderKittens <br/> 用 CUDA/**Triton**/CUTLASS/ThunderKittens 编写内核")

    text("## Parallelism <br/> 并行")
    text("What if we have multiple GPUs (8 A100s)? <br/> 如果我们有多个 GPU（8 个 A100）怎么办？")
    image("https://www.fibermall.com/blog/wp-content/uploads/2024/09/the-hardware-topology-of-a-typical-8xA100-GPU-host.png", width=500)
    text("Data movement between GPUs is even slower, but same 'minimize data movement' principle holds <br/> GPU 之间的数据移动更慢，但同样的'最小化数据移动'原则仍然适用")
    text("Use collective operations (e.g., gather, reduce, all-reduce) <br/> 使用集合操作（如 gather、reduce、all-reduce）")
    text("Shard (parameters, activations, gradients, optimizer states) across GPUs <br/> 在 GPU 之间分片（参数、激活、梯度、优化器状态）")
    text("How to split computation: {data,tensor,pipeline,sequence} parallelism <br/> 如何分割计算：{数据,张量,流水线,序列}并行")
    
    text("## Inference <br/> 推理")
    text("Goal: generate tokens given a prompt (needed to actually use models!) <br/> 目标：给定提示生成 token（实际使用模型所需！）")
    text("Inference is also needed for reinforcement learning, test-time compute, evaluation <br/> 推理也用于强化学习、测试时计算、评估")
    text("Globally, inference compute (every use) exceeds training compute (one-time cost) <br/> 全局来看，推理计算（每次使用）超过训练计算（一次性成本）")
    text("Two phases: prefill and decode <br/> 两个阶段：预填充和解码")
    image("images/prefill-decode.png", width=500)
    text("Prefill (similar to training): tokens are given, can process all at once (compute-bound) <br/> 预填充（类似训练）：token 已给定，可以一次性处理（计算受限）")
    text("Decode: need to generate one token at a time (memory-bound) <br/> 解码：需要一次生成一个 token（内存受限）")
    text("Methods to speed up decoding: <br/> 加速解码的方法：")
    text("- Use cheaper model (via model pruning, quantization, distillation) <br/> - 使用更便宜的模型（通过模型剪枝、量化、蒸馏）")
    text("- Speculative decoding: use a cheaper \"draft\" model to generate multiple tokens, then use the full model to score in parallel (exact decoding!) <br/> - 投机解码：使用更便宜的'草稿'模型生成多个 token，然后使用完整模型并行评分（精确解码！）")
    text("- Systems optimizations: KV caching, batching <br/> - 系统优化：KV 缓存、批处理")

    text("## Assignment 2 <br/> 作业 2")
    link(title="[GitHub from 2024]", url="https://github.com/stanford-cs336/spring2024-assignment2-systems"), link(title="[PDF from 2024]", url="https://github.com/stanford-cs336/spring2024-assignment2-systems/blob/master/cs336_spring2024_assignment2_systems.pdf")
    text("- Implement a fused RMSNorm kernel in Triton <br/> - 用 Triton 实现融合的 RMSNorm 内核")
    text("- Implement distributed data parallel training <br/> - 实现分布式数据并行训练")
    text("- Implement optimizer state sharding <br/> - 实现优化器状态分片")
    text("- Benchmark and profile the implementations <br/> - 对实现进行基准测试和性能分析")


def scaling_laws():
    text("Goal: do experiments at small scale, predict hyperparameters/loss at large scale <br/> 目标：在小规模进行实验，预测大规模的超参数/损失")
    text("Question: given a FLOPs budget ($C$), use a bigger model ($N$) or train on more tokens ($D$)? <br/> 问题：给定 FLOPs 预算（$C$），使用更大的模型（$N$）还是在更多 token 上训练（$D$）？")
    text("Compute-optimal scaling laws: <br/> 计算最优的缩放定律："), link(kaplan_scaling_laws_2020), link(chinchilla)
    image("images/chinchilla-isoflop.png", width=800)
    text("TL;DR: $D^* = 20 N^*$ (e.g., 1.4B parameter model should be trained on 28B tokens) <br/> 简而言之：$D^* = 20 N^*$（例如，1.4B 参数的模型应该在 28B token 上训练）")
    text("But this doesn't take into account inference costs! <br/> 但这没有考虑推理成本！")

    text("## Assignment 3 <br/> 作业 3")
    link(title="[GitHub from 2024]", url="https://github.com/stanford-cs336/spring2024-assignment3-scaling"), link(title="[PDF from 2024]", url="https://github.com/stanford-cs336/spring2024-assignment3-scaling/blob/master/cs336_spring2024_assignment3_scaling.pdf")
    text("- We define a training API (hyperparameters -> loss) based on previous runs <br/> - 我们基于之前的运行定义一个训练 API（超参数 -> 损失）")
    text("- Submit \"training jobs\" (under a FLOPs budget) and gather data points <br/> - 提交'训练任务'（在 FLOPs 预算内）并收集数据点")
    text("- Fit a scaling law to the data points <br/> - 将缩放定律拟合到数据点")
    text("- Submit predictions for scaled up hyperparameters <br/> - 提交扩展后超参数的预测")
    text("- Leaderboard: minimize loss given FLOPs budget <br/> - 排行榜：在给定 FLOPs 预算下最小化损失")


def data():
    text("Question: What capabilities do we want the model to have? <br/> 问题：我们希望模型具有什么能力？")
    text("Multilingual? Code? Math? <br/> 多语言？代码？数学？")
    image("https://ar5iv.labs.arxiv.org/html/2101.00027/assets/pile_chart2.png", width=600)

    text("## Evaluation <br/> 评估")
    text("- Perplexity: textbook evaluation for language models <br/> - 困惑度：语言模型的教科书式评估")
    text("- Standardized testing (e.g., MMLU, HellaSwag, GSM8K) <br/> - 标准化测试（如 MMLU、HellaSwag、GSM8K）")
    text("- Instruction following (e.g., AlpacaEval, IFEval, WildBench) <br/> - 指令遵循（如 AlpacaEval、IFEval、WildBench）")
    text("- Scaling test-time compute: chain-of-thought, ensembling <br/> - 扩展测试时计算：思维链、集成")
    text("- LM-as-a-judge: evaluate generative tasks <br/> - LM 作为评判者：评估生成任务")
    text("- Full system: RAG, agents <br/> - 完整系统：RAG、智能体")

    text("## Data curation <br/> 数据整理")
    text("- Data does not just fall from the sky. <br/> - 数据不会从天上掉下来。")
    # look_at_web_data()
    text("- Sources: webpages crawled from the Internet, books, arXiv papers, GitHub code, etc. <br/> - 来源：从互联网爬取的网页、书籍、arXiv 论文、GitHub 代码等。")
    text("- Appeal to fair use to train on copyright data? <br/> - 援引合理使用来训练版权数据？"), link("https://arxiv.org/pdf/2303.15715.pdf")
    text("- Might have to license data (e.g., Google with Reddit data) <br/> - 可能需要授权数据（例如，Google 与 Reddit 数据）"), article_link("https://www.reuters.com/technology/reddit-ai-content-licensing-deal-with-google-sources-say-2024-02-22/")
    text("- Formats: HTML, PDF, directories (not text!) <br/> - 格式：HTML、PDF、目录（不是文本！）")

    text("## Data processing <br/> 数据处理")
    text("- Transformation: convert HTML/PDF to text (preserve content, some structure, rewriting) <br/> - 转换：将 HTML/PDF 转换为文本（保留内容、部分结构、重写）")
    text("- Filtering: keep high quality data, remove harmful content (via classifiers) <br/> - 过滤：保留高质量数据，移除有害内容（通过分类器）")
    text("- Deduplication: save compute, avoid memorization; use Bloom filters or MinHash <br/> - 去重：节省计算，避免记忆；使用布隆过滤器或 MinHash")

    text("## Assignment 4 <br/> 作业 4")
    link(title="[GitHub from 2024]", url="https://github.com/stanford-cs336/spring2024-assignment4-data"), link(title="[PDF from 2024]", url="https://github.com/stanford-cs336/spring2024-assignment4-data/blob/master/cs336_spring2024_assignment4_data.pdf")
    text("- Convert Common Crawl HTML to text <br/> - 将 Common Crawl HTML 转换为文本")
    text("- Train classifiers to filter for quality and harmful content <br/> - 训练分类器来过滤质量和有害内容")
    text("- Deduplication using MinHash <br/> - 使用 MinHash 去重")
    text("- Leaderboard: minimize perplexity given token budget <br/> - 排行榜：在给定 token 预算下最小化困惑度")


# def look_at_web_data():
#     urls = get_common_crawl_urls()[:3]  # @inspect urls
#     documents = list(read_common_crawl(urls[1], limit=300))
#     random.seed(40)
#     random.shuffle(documents)
#     documents = markdownify_documents(documents[:10])
#     write_documents(documents, "var/sample-documents.txt")
#     link(title="[sample documents]", url="var/sample-documents.txt")
#     text("It's a wasteland out there!  Need to really process the data.")


def alignment():
    text("So far, a **base model** is raw potential, very good at completing the next token. <br/> 到目前为止，**基础模型**是原始潜力，非常擅长完成下一个 token。")
    text("Alignment makes the model actually useful. <br/> 对齐使模型真正有用。")

    text("Goals of alignment: <br/> 对齐的目标：")
    text("- Get the language model to follow instructions <br/> - 让语言模型遵循指令")
    text("- Tune the style (format, length, tone, etc.) <br/> - 调整风格（格式、长度、语气等）")
    text("- Incorporate safety (e.g., refusals to answer harmful questions) <br/> - 纳入安全性（例如，拒绝回答有害问题）")

    text("Two phases: <br/> 两个阶段：")
    supervised_finetuning()
    learning_from_feedback()

    text("## Assignment 5 <br/> 作业 5")
    link(title="[GitHub from 2024]", url="https://github.com/stanford-cs336/spring2024-assignment5-alignment"), link(title="[PDF from 2024]", url="https://github.com/stanford-cs336/spring2024-assignment5-alignment/blob/master/cs336_spring2024_assignment5_alignment.pdf")
    text("- Implement supervised fine-tuning <br/> - 实现监督微调")
    text("- Implement Direct Preference Optimization (DPO) <br/> - 实现直接偏好优化（DPO）")
    text("- Implement Group Relative Preference Optimization (GRPO) <br/> - 实现组相对偏好优化（GRPO）")


@dataclass(frozen=True)
class Turn:
    role: str
    content: str


@dataclass(frozen=True)
class ChatExample:
    turns: list[Turn]


@dataclass(frozen=True)
class PreferenceExample:
    history: list[Turn]
    response_a: str
    response_b: str
    chosen: str


def supervised_finetuning():
    text("## Supervised finetuning (SFT) <br/> 监督微调（SFT）")

    text("Instruction data: (prompt, response) pairs <br/> 指令数据：（提示，响应）对")
    sft_data: list[ChatExample] = [
        ChatExample(
            turns=[
                Turn(role="system", content="You are a helpful assistant."),
                Turn(role="user", content="What is 1 + 1?"),
                Turn(role="assistant", content="The answer is 2."),
            ],
        ),
    ]
    text("Data often involves human annotation. <br/> 数据通常涉及人工标注。")
    text("Intuition: base model already has the skills, just need few examples to surface them. <br/> 直觉：基础模型已经具备技能，只需要少量示例来激发它们。"), link(lima)
    text("Supervised learning: fine-tune model to maximize p(response | prompt). <br/> 监督学习：微调模型以最大化 p(response | prompt)。")


def learning_from_feedback():
    text("Now we have a preliminary instruction following model. <br/> 现在我们有了一个初步的指令遵循模型。")
    text("Let's make it better without expensive annotation. <br/> 让我们在没有昂贵标注的情况下改进它。")
    
    text("## Preference data <br/> 偏好数据")
    text("Data: generate multiple responses using model (e.g., [A, B]) to a given prompt. <br/> 数据：使用模型对给定提示生成多个响应（例如，[A, B]）。")
    text("User provides preferences (e.g., A < B or A > B). <br/> 用户提供偏好（例如，A < B 或 A > B）。")
    preference_data: list[PreferenceExample] = [
        PreferenceExample(
            history=[
                Turn(role="system", content="You are a helpful assistant."),
                Turn(role="user", content="What is the best way to train a language model?"),
            ],
            response_a="You should use a large dataset and train for a long time.",
            response_b="You should use a small dataset and train for a short time.",
            chosen="a",
        )
    ]

    text("## Verifiers <br/> 验证器")
    text("- Formal verifiers (e.g., for code, math) <br/> - 形式验证器（例如，用于代码、数学）")
    text("- Learned verifiers: train against an LM-as-a-judge <br/> - 学习的验证器：针对 LM 作为评判者进行训练")

    text("## Algorithms <br/> 算法")
    text("- Proximal Policy Optimization (PPO) from reinforcement learning <br/> - 来自强化学习的近端策略优化（PPO）"), link(ppo2017), link(instruct_gpt)
    text("- Direct Policy Optimization (DPO): for preference data, simpler <br/> - 直接策略优化（DPO）：用于偏好数据，更简单 "), link(dpo)
    text("- Group Relative Preference Optimization (GRPO): remove value function <br/> - 组相对偏好优化（GRPO）：移除价值函数 "), link(grpo)


############################################################
# Tokenization

# https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py#L23
GPT2_TOKENIZER_REGEX = \
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def tokenization():
    text("This unit was inspired by Andrej Karpathy's video on tokenization; check it out! <br/> 本单元受 Andrej Karpathy 关于分词的视频启发；去看看吧！"), youtube_link("https://www.youtube.com/watch?v=zduSFxRajkE")

    intro_to_tokenization()
    tokenization_examples()
    character_tokenizer()
    byte_tokenizer()
    word_tokenizer()
    bpe_tokenizer()

    text("## Summary <br/> 总结")
    text("- Tokenizer: strings <-> tokens (indices) <br/> - 分词器：字符串 <-> token（索引）")
    text("- Character-based, byte-based, word-based tokenization highly suboptimal <br/> - 基于字符、字节、单词的分词高度次优")
    text("- BPE is an effective heuristic that looks at corpus statistics <br/> - BPE 是一种有效的启发式方法，利用语料库统计")
    text("- Tokenization is a necessary evil, maybe one day we'll just do it from bytes... <br/> - 分词是必要之恶，也许有一天我们能直接从字节开始...")

@dataclass(frozen=True)
class BPETokenizerParams:
    """All you need to specify a BPETokenizer."""
    vocab: dict[int, bytes]     # index -> bytes
    merges: dict[tuple[int, int], int]  # index1,index2 -> new_index



class CharacterTokenizer(Tokenizer):
    """Represent a string as a sequence of Unicode code points."""
    def encode(self, string: str) -> list[int]:
        return list(map(ord, string))

    def decode(self, indices: list[int]) -> str:
        return "".join(map(chr, indices))


class ByteTokenizer(Tokenizer):
    """Represent a string as a sequence of bytes."""
    def encode(self, string: str) -> list[int]:
        string_bytes = string.encode("utf-8")  # @inspect string_bytes
        indices = list(map(int, string_bytes))  # @inspect indices
        return indices

    def decode(self, indices: list[int]) -> str:
        string_bytes = bytes(indices)  # @inspect string_bytes
        string = string_bytes.decode("utf-8")  # @inspect string
        return string


def merge(indices: list[int], pair: tuple[int, int], new_index: int) -> list[int]:  # @inspect indices, @inspect pair, @inspect new_index
    """Return `indices`, but with all instances of `pair` replaced with `new_index`."""
    new_indices = []  # @inspect new_indices
    i = 0  # @inspect i
    while i < len(indices):
        if i + 1 < len(indices) and indices[i] == pair[0] and indices[i + 1] == pair[1]:
            new_indices.append(new_index)
            i += 2
        else:
            new_indices.append(indices[i])
            i += 1
    return new_indices


class BPETokenizer(Tokenizer):
    """BPE tokenizer given a set of merges and a vocabulary."""
    def __init__(self, params: BPETokenizerParams):
        self.params = params

    def encode(self, string: str) -> list[int]:
        indices = list(map(int, string.encode("utf-8")))  # @inspect indices
        # Note: this is a very slow implementation
        for pair, new_index in self.params.merges.items():  # @inspect pair, @inspect new_index
            indices = merge(indices, pair, new_index)
        return indices

    def decode(self, indices: list[int]) -> str:
        bytes_list = list(map(self.params.vocab.get, indices))  # @inspect bytes_list
        string = b"".join(bytes_list).decode("utf-8")  # @inspect string
        return string


def get_compression_ratio(string: str, indices: list[int]) -> float:
    """Given `string` that has been tokenized into `indices`, ."""
    num_bytes = len(bytes(string, encoding="utf-8"))  # @inspect num_bytes
    num_tokens = len(indices)                       # @inspect num_tokens
    return num_bytes / num_tokens


def get_gpt2_tokenizer():
    # Code: https://github.com/openai/tiktoken
    # You can use cl100k_base for the gpt3.5-turbo or gpt4 tokenizer
    return tiktoken.get_encoding("gpt2")


def intro_to_tokenization():
    text("Raw text is generally represented as Unicode strings. <br/> 原始文本通常表示为 Unicode 字符串。")
    string = "Hello, 🌍! 你好!"

    text("A language model places a probability distribution over sequences of tokens (usually represented by integer indices). <br/> 语言模型在 token 序列上放置概率分布（通常用整数索引表示）。")
    indices = [15496, 11, 995, 0]

    text("So we need a procedure that *encodes* strings into tokens. <br/> 所以我们需要一个将字符串*编码*为 token 的过程。")
    text("We also need a procedure that *decodes* tokens back into strings. <br/> 我们还需要一个将 token *解码*回字符串的过程。")
    text("A <br/> 一个 "), link(Tokenizer), text(" is a class that implements the encode and decode methods. <br/>  是一个实现编码和解码方法的类。")
    text("The **vocabulary size** is number of possible tokens (integers). <br/> **词汇表大小**是可能的 token（整数）数量。")


def tokenization_examples():
    text("To get a feel for how tokenizers work, play with this <br/> 要了解分词器如何工作，可以玩一下这个 "), link(title="interactive site", url="https://tiktokenizer.vercel.app/?encoder=gpt2")

    text("## Observations <br/> 观察")
    text("- A word and its preceding space are part of the same token (e.g., \" world\"). <br/> - 一个词及其前面的空格是同一个 token 的一部分（例如，\" world\"）。")
    text("- A word at the beginning and in the middle are represented differently (e.g., \"hello hello\"). <br/> - 开头和中间的单词表示方式不同（例如，\"hello hello\"）。")
    text("- Numbers are tokenized into every few digits. <br/> - 数字每隔几位被分词。")

    text("Here's the GPT-2 tokenizer from OpenAI (tiktoken) in action. <br/> 这是 OpenAI 的 GPT-2 分词器（tiktoken）的实际运行。")
    tokenizer = get_gpt2_tokenizer()
    string = "Hello, 🌍! 你好!"  # @inspect string

    text("Check that encode() and decode() roundtrip: <br/> 检查 encode() 和 decode() 是否能往返：")
    indices = tokenizer.encode(string)  # @inspect indices
    reconstructed_string = tokenizer.decode(indices)  # @inspect reconstructed_string
    assert string == reconstructed_string
    compression_ratio = get_compression_ratio(string, indices)  # @inspect compression_ratio


def character_tokenizer():
    text("## Character-based tokenization <br/> 基于字符的分词")

    text("A Unicode string is a sequence of Unicode characters. <br/> Unicode 字符串是 Unicode 字符的序列。")
    text("Each character can be converted into a code point (integer) via `ord`. <br/> 每个字符可以通过 `ord` 转换为码点（整数）。")
    assert ord("a") == 97
    assert ord("🌍") == 127757
    text("It can be converted back via `chr`. <br/> 可以通过 `chr` 转换回来。")
    assert chr(97) == "a"
    assert chr(127757) == "🌍"

    text("Now let's build a `Tokenizer` and make sure it round-trips: <br/> 现在让我们构建一个 `Tokenizer` 并确保它能往返：")
    tokenizer = CharacterTokenizer()
    string = "Hello, 🌍! 你好!"  # @inspect string
    indices = tokenizer.encode(string)  # @inspect indices
    reconstructed_string = tokenizer.decode(indices)  # @inspect reconstructed_string
    assert string == reconstructed_string

    text("There are approximately 150K Unicode characters. <br/> 大约有 15 万个 Unicode 字符。"), link(title="[Wikipedia]", url="https://en.wikipedia.org/wiki/List_of_Unicode_characters")
    vocabulary_size = max(indices) + 1  # This is a lower bound @inspect vocabulary_size
    text("Problem 1: this is a very large vocabulary. <br/> 问题 1：这是一个非常大的词汇表。")
    text("Problem 2: many characters are quite rare (e.g., 🌍), which is inefficient use of the vocabulary. <br/> 问题 2：许多字符非常罕见（例如，🌍），这是词汇表的低效使用。")
    compression_ratio = get_compression_ratio(string, indices)  # @inspect compression_ratio


def byte_tokenizer():
    text("## Byte-based tokenization <br/> 基于字节的分词")

    text("Unicode strings can be represented as a sequence of bytes, which can be represented by integers between 0 and 255. <br/> Unicode 字符串可以表示为字节序列，字节可以用 0 到 255 之间的整数表示。")
    text("The most common Unicode encoding is <br/> 最常见的 Unicode 编码是 "), link(title="UTF-8", url="https://en.wikipedia.org/wiki/UTF-8")

    text("Some Unicode characters are represented by one byte: <br/> 一些 Unicode 字符用一个字节表示：")
    assert bytes("a", encoding="utf-8") == b"a"
    text("Others take multiple bytes: <br/> 其他字符需要多个字节：")
    assert bytes("🌍", encoding="utf-8") == b"\xf0\x9f\x8c\x8d"

    text("Now let's build a `Tokenizer` and make sure it round-trips: <br/> 现在让我们构建一个 `Tokenizer` 并确保它能往返：")
    tokenizer = ByteTokenizer()
    string = "Hello, 🌍! 你好!"  # @inspect string
    indices = tokenizer.encode(string)  # @inspect indices
    reconstructed_string = tokenizer.decode(indices)  # @inspect reconstructed_string
    assert string == reconstructed_string

    text("The vocabulary is nice and small: a byte can represent 256 values. <br/> 词汇表很小很好：一个字节可以表示 256 个值。")
    vocabulary_size = 256  # @inspect vocabulary_size
    text("What about the compression rate? <br/> 压缩率如何？")
    compression_ratio = get_compression_ratio(string, indices)  # @inspect compression_ratio
    assert compression_ratio == 1
    text("The compression ratio is terrible, which means the sequences will be too long. <br/> 压缩率很糟糕，这意味着序列会太长。")
    text("Given that the context length of a Transformer is limited (since attention is quadratic), this is not looking great... <br/> 鉴于 Transformer 的上下文长度有限（因为注意力是二次的），这看起来不太好...")


def word_tokenizer():
    text("## Word-based tokenization <br/> 基于单词的分词")

    text("Another approach (closer to what was done classically in NLP) is to split strings into words. <br/> 另一种方法（更接近经典 NLP 的做法）是将字符串分割成单词。")
    string = "I'll say supercalifragilisticexpialidocious!"

    segments = regex.findall(r"\w+|.", string)  # @inspect segments
    text("This regular expression keeps all alphanumeric characters together (words). <br/> 这个正则表达式将所有字母数字字符保持在一起（单词）。")

    text("Here is a fancier version: <br/> 这是一个更复杂的版本：")
    pattern = GPT2_TOKENIZER_REGEX  # @inspect pattern
    segments = regex.findall(pattern, string)  # @inspect segments

    text("To turn this into a `Tokenizer`, we need to map these segments into integers. <br/> 要将其转换为 `Tokenizer`，我们需要将这些片段映射为整数。")
    text("Then, we can build a mapping from each segment into an integer. <br/> 然后，我们可以构建每个片段到整数的映射。")

    text("But there are problems: <br/> 但有一些问题：")
    text("- The number of words is huge (like for Unicode characters). <br/> - 单词数量巨大（像 Unicode 字符一样）。")
    text("- Many words are rare and the model won't learn much about them. <br/> - 许多单词很罕见，模型不会学到太多关于它们的知识。")
    text("- This doesn't obviously provide a fixed vocabulary size. <br/> - 这显然不能提供固定的词汇表大小。")

    text("New words we haven't seen during training get a special UNK token, which is ugly and can mess up perplexity calculations. <br/> 训练期间没见过的新词会得到一个特殊的 UNK token，这很丑陋，可能会搞乱困惑度计算。")

    vocabulary_size = "Number of distinct segments in the training data"
    compression_ratio = get_compression_ratio(string, segments)  # @inspect compression_ratio


def bpe_tokenizer():
    text("## Byte Pair Encoding (BPE) <br/> 字节对编码（BPE）")
    link(title="[Wikipedia]", url="https://en.wikipedia.org/wiki/Byte_pair_encoding")
    text("The BPE algorithm was introduced by Philip Gage in 1994 for data compression. <br/> BPE 算法由 Philip Gage 于 1994 年为数据压缩而引入。"), article_link("http://www.pennelynn.com/Documents/CUJ/HTML/94HTML/19940045.HTM")
    text("It was adapted to NLP for neural machine translation. <br/> 它被改编用于神经机器翻译的 NLP。"), link(sennrich_2016)
    text("(Previously, papers had been using word-based tokenization.) <br/> （此前，论文一直使用基于单词的分词。）")
    text("BPE was then used by GPT-2. <br/> BPE 随后被 GPT-2 使用。"), link(gpt2)

    text("Basic idea: *train* the tokenizer on raw text to automatically determine the vocabulary. <br/> 基本思想：在原始文本上*训练*分词器以自动确定词汇表。")
    text("Intuition: common sequences of characters are represented by a single token, rare sequences are represented by many tokens. <br/> 直觉：常见的字符序列由单个 token 表示，罕见的序列由多个 token 表示。")

    text("The GPT-2 paper used word-based tokenization to break up the text into inital segments and run the original BPE algorithm on each segment. <br/> GPT-2 论文使用基于单词的分词将文本分解为初始片段，并在每个片段上运行原始 BPE 算法。")
    text("Sketch: start with each byte as a token, and successively merge the most common pair of adjacent tokens. <br/> 草图：从每个字节作为一个 token 开始，然后依次合并最常见的相邻 token 对。")

    text("## Training the tokenizer <br/> 训练分词器")
    string = "the cat in the hat"  # @inspect string
    params = train_bpe(string, num_merges=3)

    text("## Using the tokenizer <br/> 使用分词器")
    text("Now, given a new text, we can encode it. <br/> 现在，给定新文本，我们可以对其进行编码。")
    tokenizer = BPETokenizer(params)
    string = "the quick brown fox"  # @inspect string
    indices = tokenizer.encode(string)  # @inspect indices
    reconstructed_string = tokenizer.decode(indices)  # @inspect reconstructed_string
    assert string == reconstructed_string

    text("In Assignment 1, you will go beyond this in the following ways: <br/> 在作业 1 中，你将通过以下方式超越这些：")
    text("- encode() currently loops over all merges. Only loop over merges that matter. <br/> - encode() 目前遍历所有合并。只遍历重要的合并。")
    text("- Detect and preserve special tokens (e.g., <|endoftext|>). <br/> - 检测并保留特殊 token（例如，<|endoftext|>）。")
    text("- Use pre-tokenization (e.g., the GPT-2 tokenizer regex). <br/> - 使用预分词（例如，GPT-2 分词器正则表达式）。")
    text("- Try to make the implementation as fast as possible. <br/> - 尽量使实现尽可能快。")


def train_bpe(string: str, num_merges: int) -> BPETokenizerParams:  # @inspect string, @inspect num_merges
    text("Start with the list of bytes of `string`. <br/> 从 `string` 的字节列表开始。")
    indices = list(map(int, string.encode("utf-8")))  # @inspect indices
    merges: dict[tuple[int, int], int] = {}  # index1, index2 => merged index
    vocab: dict[int, bytes] = {x: bytes([x]) for x in range(256)}  # index -> bytes

    for i in range(num_merges):
        text("Count the number of occurrences of each pair of tokens <br/> 计算每对 token 的出现次数")
        counts = defaultdict(int)
        for index1, index2 in zip(indices, indices[1:]):  # For each adjacent pair
            counts[(index1, index2)] += 1  # @inspect counts

        text("Find the most common pair. <br/> 找到最常见的对。")
        pair = max(counts, key=counts.get)  # @inspect pair
        index1, index2 = pair

        text("Merge that pair. <br/> 合并该对。")
        new_index = 256 + i  # @inspect new_index
        merges[pair] = new_index  # @inspect merges
        vocab[new_index] = vocab[index1] + vocab[index2]  # @inspect vocab
        indices = merge(indices, pair, new_index)  # @inspect indices

    return BPETokenizerParams(vocab=vocab, merges=merges)


if __name__ == "__main__":
    main()
