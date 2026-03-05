from execute_util import text, link, image
from lecture_util import x_link, blog_link
from references import deepseek_r1, llama4, olmo2_32b, mmlu

def main():
    text("**Evaluation**: given a **fixed model**, how "**good**" is it? <br/> **评估**：给定一个**固定模型**，它有多"**好**"？")

    what_you_see()
    how_to_think_about_evaluation()

    perplexity()

    knowledge_benchmarks()
    instruction_following_benchmarks()
    agent_benchmarks()
    pure_reasoning_benchmarks()
    safety_benchmarks()

    realism()
    validity()
    what_are_we_evaluating()

    text("Takeaways <br/> 要点")
    text("- There is no one true evaluation; choose the evaluation depending on what you're trying to measure. <br/> - 没有唯一的真实评估；根据你想测量的内容选择评估方法。")
    text("- Always look at the individual instances and the predictions. <br/> - 始终查看单个实例和预测。")
    text("- There are many aspects to consider: capabilities, safety, costs, realism. <br/> - 有许多方面需要考虑：能力、安全性、成本、真实性。")
    text("- Clearly state the rules of the game (methods versus models/systems). <br/> - 明确说明游戏规则（方法与模型/系统）。")


def what_you_see():
    text("## Benchmark scores <br/> ## 基准测试分数")
    image("images/deepseek-r1-benchmarks.png", width=800), link(deepseek_r1)
    image("images/llama4-benchmarks.png", width=800), link(llama4)
    image("https://www.datocms-assets.com/64837/1741887109-instruct-1.png", width=800), link(olmo2_32b)

    text("Recent language models are evaluated on similar, but not entirely identical, benchmarks (MMLU, MATH, etc.). <br/> 最近的语言模型在相似但不完全相同的基准测试上进行评估（MMLU、MATH 等）。")
    text("What are these benchmarks? <br/> 这些基准测试是什么？")
    text("What do these numbers mean? <br/> 这些数字意味着什么？")

    image("images/helm-capabilities-leaderboard.png", width=1000)
    link(title="[HELM capabilities]", url="https://crfm.stanford.edu/helm/capabilities/latest/#/leaderboard")

    text("Pay close attention to the costs! <br/> 密切关注成本！")
    image("images/artificial-analysis.png", width=800), link(title="[Artificial Analysis]", url="https://artificialanalysis.ai/")

    text("Maybe a model is good if people choose to use it (and pay for it)... <br/> 也许如果人们选择使用它（并为之付费），这个模型就是好的...")
    image("images/openrouter.png", width=600), link(title="[OpenRouter]", url="https://openrouter.ai/rankings")

    image("images/chatbot-arena-leaderboard.png", width=800)
    link(title="[Chatbot Arena]", url="https://huggingface.co/spaces/lmarena-ai/chatbot-arena-leaderboard")

    text("## Vibes <br/> ## 氛围")
    x_link("https://x.com/demishassabis/status/1919779362980692364")
    image("images/demis-gemini-2.5.png", width=500)

    text("A crisis... <br/> 一场危机...")
    image("images/karpathy-crisis.png", width=600)


def how_to_think_about_evaluation():
    text("You might think evaluation is a mechanical process (take existing model, throw prompts at it, average some numbers)... <br/> 你可能认为评估是一个机械过程（拿现有模型，向它抛出提示，平均一些数字）...")
    text("Actually, evaluation is a profound and rich topic... <br/> 实际上，评估是一个深刻而丰富的话题...")
    text("...and it determines the future of language models. <br/> ...而且它决定了语言模型的未来。")

    text("What's the point of evaluation? <br/> 评估的意义是什么？")
    text("There is no one true evaluation; it depends on what question you're trying to answer. <br/> 没有唯一的真实评估；这取决于你想回答什么问题。")
    text("1. User or company wants to make a purchase decision (model A or model B) for their use case (e.g., customer service chatbots). <br/> 1. 用户或公司想为其用例做出购买决策（模型 A 或模型 B）（如客户服务聊天机器人）。")
    text("2. Researchers want to measure the raw capabilities of a model (e.g., intelligence). <br/> 2. 研究人员想测量模型的原始能力（如智能）。")
    text("3. We want to understand the benefits + harms of a model (for business and policy reasons). <br/> 3. 我们想了解模型的益处 + 危害（出于商业和政策原因）。")
    text("4. Model developers want to get feedback to improve the model. <br/> 4. 模型开发者想获得反馈以改进模型。")
    text("In each case, there is an abstract **goal** that needs to be translated into a concrete evaluation. <br/> 在每种情况下，都有一个抽象的**目标**需要转化为具体的评估。")

    text("Framework <br/> 框架")
    text("1. What are the **inputs**? <br/> 1. **输入**是什么？")
    text("2. How do **call** the language model? <br/> 2. 如何**调用**语言模型？")
    text("3. How do you evaluate the **outputs**? <br/> 3. 如何评估**输出**？")
    text("4. How to **interpret** the results? <br/> 4. 如何**解释**结果？")

    text("What are the inputs? <br/> 输入是什么？")
    text("1. What use cases are **covered**? <br/> 1. **覆盖**哪些用例？")
    text("2. Do we have representation of **difficult** inputs in the tail? <br/> 2. 我们在尾部是否有**困难**输入的代表？")
    text("3. Are the inputs **adapted** to the model (e.g., multi-turn)? <br/> 3. 输入是否**适应**模型（如多轮）？")

    text("How do you call the language model? <br/> 如何调用语言模型？")
    text("1. How do you prompt the language model? <br/> 1. 如何提示语言模型？")
    text("2. Does the language model use chain-of-thought, tools, RAG, etc.? <br/> 2. 语言模型是否使用思维链、工具、RAG 等？")
    text("3. Are we evaluating the language model or an agentic system (model developer wants former, user wants latter)? <br/> 3. 我们是在评估语言模型还是智能体系统（模型开发者想要前者，用户想要后者）？")

    text("How do you evaluate the outputs? <br/> 如何评估输出？")
    text("1. Are the reference outputs used for evaluation error-free? <br/> 1. 用于评估的参考输出是否无错误？")
    text("2. What metrics do you use (e.g., pass@k)? <br/> 2. 使用什么指标（如 pass@k）？")
    text("3. How do you factor in cost (e.g., inference + training)? <br/> 3. 如何考虑成本（如推理 + 训练）？")
    text("4. How do you factor in asymmetric errors (e.g., hallucinations in a medical setting)? <br/> 4. 如何考虑不对称错误（如医疗环境中的幻觉）？")
    text("5. How do you handle open-ended generation (no ground truth)? <br/> 5. 如何处理开放式生成（无真实标签）？")

    text("How do you inteprret the metrics? <br/> 如何解释指标？")
    text("1. How do you interpret a number (e.g., 91%) - is it ready for deployment? <br/> 1. 如何解释一个数字（如 91%）- 它准备好部署了吗？")
    text("2. How do we assess generalization in the face of train-test overlap? <br/> 2. 面对训练-测试重叠时，我们如何评估泛化能力？")
    text("3. Are we evaluating the final model or the method? <br/> 3. 我们是在评估最终模型还是方法？")

    text("Summary: lots of questions to think through when doing evaluation <br/> 总结：进行评估时有很多问题需要考虑")

def perplexity():
    text("Recall: that a language model is a probability distribution **p(x)** over sequences of tokens. <br/> 回想：语言模型是 token 序列上的概率分布 **p(x)**。")
    text("Perplexity (1/p(D))^(1/|D|) measures whether p assigns high probability to some dataset D. <br/> 困惑度 (1/p(D))^(1/|D|) 测量 p 是否给某个数据集 D 分配高概率。")

    text("In pre-training, you minimize perplexity on the training set. <br/> 在预训练中，你在训练集上最小化困惑度。")
    text("The obvious thing is to measure perplexity on the test set. <br/> 显然的事情是在测试集上测量困惑度。")

    text("Standard datasets: Penn Treebank (WSJ), WikiText-103 (Wikipedia), One Billion Word Benchmark (from machine translation WMT11 - EuroParl, UN, news) <br/> 标准数据集：Penn Treebank (WSJ)、WikiText-103 (Wikipedia)、One Billion Word Benchmark（来自机器翻译 WMT11 - EuroParl、UN、新闻）")
    text("Papers trained on a dataset (training split) and evaluated on the same dataset (test split) <br/> 论文在数据集（训练集）上训练，在同一数据集（测试集）上评估")
    text("Pure CNNs+LSTMs on the One Billion Word Benchmark (perplexity 51.3 -> 30.0) "), link("https://arxiv.org/abs/1602.02410")

    text("GPT-2 trained on WebText (40GB text, websites linked from Reddit), zero-shot on standard datasets <br/> GPT-2 在 WebText（40GB 文本，来自 Reddit 链接的网站）上训练，在标准数据集上零样本测试")
    text("This is out-of-distribution evaluation (but idea is that training covers a lot) <br/> 这是分布外评估（但想法是训练覆盖很多）")
    image("images/gpt2-perplexity.png", width=800)
    text("Works better on small datasets (transfer is helpful), but not larger datasets (1BW) <br/> 在小数据集上效果更好（迁移有帮助），但在更大数据集上不行（1BW）")

    text("Since GPT-2 and GPT-3, language modeling papers have shifted more towards downstream task accuracy. <br/> 自 GPT-2 和 GPT-3 以来，语言建模论文已更多转向下游任务准确性。")
    text("But reasons why perplexity is still useful: <br/> 但困惑度仍然有用的原因：")
    text("- Smoother than downstream task accuracy (for fitting scaling laws) <br/> - 比下游任务准确性更平滑（用于拟合缩放定律）")
    text("- Is universal (why we use it for training) whereas task accuracy might miss some nuances <br/> - 是通用的（为什么我们用它训练）而任务准确性可能遗漏一些细微差别")
    text("- Note: can measure conditional perplexity on downstream task too (used for scaling laws) <br/> - 注意：也可以在下游任务上测量条件困惑度（用于缩放定律）"), link("https://arxiv.org/abs/2412.04403")

    text("Warning (if you're running a leaderboard): evaluator needs to trust the language model <br/> 警告（如果你运行排行榜）：评估者需要信任语言模型")
    text("For task accuracy, can just take output generated from a blackbox model and compute the desired metrics <br/> 对于任务准确性，可以直接从黑盒模型生成的输出并计算所需指标")
    text("For perplexity, need LM to generate probabilities and trust that they sum to 1 (even worse with UNKs back in the day) <br/> 对于困惑度，需要 LM 生成概率并信任它们总和为 1（当年有 UNK 时更糟）")

    text("The perplexity maximalist view: <br/> 困惑度最大化观点：")
    text("- Your true distribution is t, model is p <br/> - 你的真实分布是 t，模型是 p")
    text("- Best possible perplexity is H(t) obtained iff p = t <br/> - 最佳可能困惑度是 H(t)，当且仅当 p = t 时获得")
    text("- If have t, then solve all the tasks <br/> - 如果有 t，则解决所有任务")
    text("- So by pushing down on perplexity, will eventually reach AGI <br/> - 所以通过降低困惑度，最终会达到 AGI")
    text("- Caveat: this might not be the most efficient way to get there (pushing down on parts of the distribution that don't matter) <br/> - 警告：这可能不是达到目标的最有效方式（在不重要的分布部分上降低）")

    text("Things that are spiritually perplexity: <br/> 精神上类似困惑度的东西：")
    text("Similar idea: cloze tasks like LAMBADA <br/> 类似想法：完形填空任务如 LAMBADA"), link("https://arxiv.org/abs/1606.06031")
    image("images/lambada.png", width=800)
    text("HellaSwag <br/> HellaSwag"), link("https://arxiv.org/pdf/1905.07830")
    image("images/hellaswag.png", width=600)


def knowledge_benchmarks():
    text("### Massive Multitask Language Understanding (MMLU) <br/> ### 大规模多任务语言理解 (MMLU)")
    link(mmlu)
    text("- 57 subjects (e.g., math, US history, law, morality), multiple-choice <br/> - 57 个科目（如数学、美国历史、法律、道德），多选题")
    text("- \"collected by graduate and undergraduate students from freely available sources online\" <br/> - \"由研究生和本科生从在线免费资源收集\"")
    text("- Really about testing knowledge, not language understanding <br/> - 真正是关于测试知识，而不是语言理解")
    text("- Evaluated on GPT-3 using few-shot prompting <br/> - 使用少样本提示在 GPT-3 上评估")
    image("images/mmlu.png", width=800)
    link(title="[HELM MMLU for visualizing predictions]", url="https://crfm.stanford.edu/helm/mmlu/latest/")

    text("### MMLU-Pro <br/> ### MMLU-Pro")
    link("https://arxiv.org/abs/2406.01574")
    text("- Removed noisy/trivial questions from MMLU <br/> - 从 MMLU 中移除嘈杂/琐碎问题")
    text("- Expanded 4 choices to 10 choices <br/> - 将 4 个选项扩展到 10 个选项")
    text("- Evaluated using chain of thought (gives model more of a chance) <br/> - 使用思维链评估（给模型更多机会）")
    text("- Accuracy of models drop by 16% to 33% (not as saturated) <br/> - 模型准确率下降 16% 到 33%（不那么饱和）")
    image("images/mmlu-pro.png", width=800)
    link(title="[HELM MMLU-Pro for visualizing predictions]", url="https://crfm.stanford.edu/helm/capabilities/latest/#/leaderboard/mmlu_pro")

    text("### Graduate-Level Google-Proof Q&A (GPQA) <br/> ### 研究生级别防谷歌问答 (GPQA)")
    link("https://arxiv.org/abs/2311.12022")
    text("- Questions written by 61 PhD contractors from Upwork <br/> - 由来自 Upwork 的 61 位博士承包商编写的问题")
    image("images/gpqa.png", width=800)
    text("- PhD experts achieve 65% accuracy <br/> - 博士专家达到 65% 准确率")
    text("- Non-experts achieve 34% over 30 minutes with access to Google <br/> - 非专家在 30 分钟内使用谷歌达到 34%")
    text("- GPT-4 achieves 39% <br/> - GPT-4 达到 39%")
    link(title="[HELM GPQA for visualizing predictions]", url="https://crfm.stanford.edu/helm/capabilities/latest/#/leaderboard/gpqa")

    text("### Humanity's Last Exam <br/> ### 人类的最后考试")
    link("https://arxiv.org/abs/2501.14249")
    text("- 2500 questions: multimodal, many subjects, multiple-choice + short-answer <br/> - 2500 个问题：多模态、多科目、多选题 + 简答题")
    image("images/hle-examples.png", width=800)
    text("- Awarded $500K prize pool + co-authorship to question creators <br/> - 授予 50 万美元奖金池 + 问题创建者共同作者身份")
    text("- Filtered by frontier LLMs, multiple stages of review <br/> - 由前沿 LLM 过滤，多阶段审查")
    image("images/hle-pipeline.png", width=800)
    image("images/hle-results.png", width=800)
    link(title="[latest leaderboard]", url="https://agi.safe.ai/")


def instruction_following_benchmarks():
    text("So far, we've been evaluating on fairly structured tasks. <br/> 到目前为止，我们一直在评估相当结构化的任务。")
    text("Instruction following (as popularized by ChatGPT): just follow the instructions. <br/> 指令遵循（由 ChatGPT 推广）：只需遵循指令。")
    text("Challenge: how to evaluate an open-ended response? <br/> 挑战：如何评估开放式响应？")

    text("### Chatbot Arena <br/> ### 聊天机器人竞技场")
    link("https://arxiv.org/abs/2403.04132")
    text("How it works: <br/> 工作原理：")
    text("- Random person from the Internet types in prompt <br/> - 来自互联网的随机人输入提示")
    text("- They get response from two random (anonymized) models <br/> - 他们从两个随机（匿名）模型获得响应")
    text("- They rate which one is better <br/> - 他们评价哪个更好")
    text("- ELO scores are computed based on the pairwise comparisons <br/> - ELO 分数基于成对比较计算")
    text("- Features: live (not static) inputs, can accomodate new models <br/> - 特点：实时（非静态）输入，可容纳新模型")
    image("images/chatbot-arena-leaderboard.png", width=800)
    link(title="[Chatbot Arena]", url="https://huggingface.co/spaces/lmarena-ai/chatbot-arena-leaderboard")

    text("### Instruction-Following Eval (IFEval) <br/> ### 指令遵循评估 (IFEval)")
    link("https://arxiv.org/abs/2311.07911")
    image("images/ifeval-categories.png", width=600)
    text("- Add simple synthetic constraints to instructions <br/> - 添加简单的合成约束到指令")
    text("- Constraints can be automatically verified, but not the semantics of the response <br/> - 约束可以自动验证，但响应的语义不能")
    text("- Fairly simple instructions, constraints are a bit artificial <br/> - 相当简单的指令，约束有点人为")
    link(title="[HELM IFEval for visualizing predictions]", url="https://crfm.stanford.edu/helm/capabilities/latest/#/leaderboard/ifeval")

    text("### AlpacaEval <br/> ### AlpacaEval")
    link("https://tatsu-lab.github.io/alpaca_eval/")
    text("- 805 instructions from various sources <br/> - 来自各种来源的 805 条指令")
    text("- Metric: win rate against GPT-4 preview as judged by GPT-4 preview (potential bias) <br/> - 指标：由 GPT-4 preview 评判的相对于 GPT-4 preview 的胜率（潜在偏见）")
    image("images/alpacaeval-leaderboard.png", width=600)

    text("### WildBench <br/> ### WildBench")
    link("https://arxiv.org/pdf/2406.04770")
    text("- Sourced 1024 examples from 1M human-chatbot conversations <br/> - 从 100 万人机对话中采集 1024 个示例")
    text("- Uses GPT-4 turbo as a judge with a checklist (like CoT for judging) + GPT-4 as a judge <br/> - 使用 GPT-4 turbo 作为评判者，带有检查清单（类似用于评判的 CoT）+ GPT-4 作为评判者")
    text("- Well-correlated (0.95) with Chatbot Arena (seems to be the de facto sanity check for benchmarks) <br/> - 与 Chatbot Arena 高度相关 (0.95)（似乎是基准测试的事实上的合理性检查）")
    image("images/wildbench.png", width=800)
    link(title="[HELM WildBench for visualizing predictions]", url="https://crfm.stanford.edu/helm/capabilities/latest/#/leaderboard/wildbench")


def agent_benchmarks():
    text("Consider tasks that require tool use (e.g., running code) and iterating over a period of time <br/> 考虑需要工具使用（如运行代码）并在一定时间内迭代的任务")
    text("Agent = language model + agent scaffolding (logic for deciding how to use the LM) <br/> 智能体 = 语言模型 + 智能体脚手架（决定如何使用 LM 的逻辑）")

    text("### SWEBench <br/> ### SWEBench")
    link("https://arxiv.org/abs/2310.06770")
    text("- 2294 tasks across 12 Python repositories <br/> - 跨 12 个 Python 仓库的 2294 个任务")
    text("- Given codebase + issue description, submit a PR <br/> - 给定代码库 + 问题描述，提交 PR")
    text("- Evaluation metric: unit tests <br/> - 评估指标：单元测试")
    image("images/swebench.png", width=800)

    text("### CyBench <br/> ### CyBench")
    link("https://arxiv.org/abs/2408.08926")
    text("- 40 Capture the Flag (CTF) tasks <br/> - 40 个夺旗 (CTF) 任务")
    text("- Use first-solve time as a measure of difficulty <br/> - 使用首次解决时间作为难度度量")
    image("images/cybench.png", width=800)
    image("images/cybench-agent.png", width=800)
    image("images/cybench-results.png", width=800)

    text("### MLEBench <br/> ### MLEBench")
    link("https://arxiv.org/abs/2410.07095")
    text("- 75 Kaggle competitions (require training models, processing data, etc.) <br/> - 75 个 Kaggle 竞赛（需要训练模型、处理数据等）")
    image("images/mlebench.png", width=800)
    image("images/mlebench-results.png", width=800)


def pure_reasoning_benchmarks():
    text("All of the tasks so far require linguistic and world knowledge <br/> 到目前为止，所有任务都需要语言和世界知识")
    text("Can we isolate reasoning from knowledge? <br/> 我们能否将推理与知识分离？")
    text("Arguably, reasoning captures a more pure form of intelligence (isn't just about memorizing facts) <br/> 可以说，推理捕捉了更纯粹的智能形式（不仅仅是记忆事实）")

    link(title="ARC-AGI", url="https://arcprize.org/arc-agi")
    text("Introduced in 2019 by Francois Chollet <br/> 由 Francois Chollet 于 2019 年提出")

    text("ARC-AGI-1 <br/> ARC-AGI-1")
    image("https://arcprize.org/media/images/arc-task-grids.jpg", width=800)
    image("https://arcprize.org/media/images/oseriesleaderboard.png", width=800)

    text("ARC-AGI-2: harder <br/> ARC-AGI-2：更难")
    image("https://arcprize.org/media/images/blog/arc-agi-2-unsolved-1.png", width=800)


def safety_benchmarks():
    image("https://www.team-bhp.com/forum/attachments/road-safety/2173645d1625144681-will-crash-test-rating-change-if-higher-variant-chosen-images-30.jpeg", width=500)
    text("What does safety mean for AI? <br/> 安全对 AI 意味着什么？")

    link(title="[HELM safety: curated set of benchmarks]", url="https://crfm.stanford.edu/helm/safety/latest/#/leaderboard")

    text("### HarmBench <br/> ### HarmBench")
    link("https://arxiv.org/abs/2402.04249")
    text("- Based on 510 harmful behaviors that violate laws or norms <br/> - 基于 510 种违反法律或规范的有害行为")
    link(title="[HarmBench on HELM]", url="https://crfm.stanford.edu/helm/safety/latest/#/leaderboard/harm_bench")
    link(title="[Example of safety failure]", url="https://crfm.stanford.edu/helm/safety/latest/#/runs/harm_bench:model=anthropic_claude-3-7-sonnet-20250219?instancesPage=4")

    text("### AIR-Bench <br/> ### AIR-Bench")
    link("https://arxiv.org/abs/2407.17436")
    text("- Based on regulatory frameworks and company policies <br/> - 基于监管框架和公司政策")
    text("- Taxonomized into 314 risk categories, 5694 prompts <br/> - 分类为 314 个风险类别，5694 个提示")
    image("https://crfm.stanford.edu/helm/assets/air-overview-d2e6c49f.png", width=800)
    link(title="[HELM AIR-Bench]", url="https://crfm.stanford.edu/helm/air-bench/latest/#/leaderboard")

    text("### Jailbreaking <br/> ### 越狱")
    text("- Language models are trained to refuse harmful instructions <br/> - 语言模型被训练拒绝有害指令")
    text("- Greedy Coordinate Gradient (GCG) automatically optimizes prompts to bypass safety <br/> - 贪婪坐标梯度 (GCG) 自动优化提示以绕过安全"), link("https://arxiv.org/pdf/2307.15043")
    text("- Transfers from open-weight models (Llama) to closed models (GPT-4) <br/> - 从开源权重模型 (Llama) 迁移到闭源模型 (GPT-4)")
    image("images/gcg-examples.png", width=800)

    text("### Pre-deployment testing <br/> ### 部署前测试")
    text("- US Safety Institute + UK AI Safety Institute working together <br/> - 美国安全研究所 + 英国 AI 安全研究所合作")
    text("- Company gives safety institutes access to model before release (currently voluntary) <br/> - 公司在发布前给安全研究所访问模型的权限（目前是自愿的）")
    text("- Safety institutes run evaluations and produce a report to company <br/> - 安全研究所运行评估并向公司生成报告")
    link(title="[report]", url="https://www.nist.gov/system/files/documents/2024/12/18/US_UK_AI%20Safety%20Institute_%20December_Publication-OpenAIo1.pdf")

    text("### But what is safety? <br/> ### 但安全是什么？")
    text("- Many aspects of safety are strongly contextual (politics, law, social norms - which vary across countries) <br/> - 安全的许多方面都是强上下文相关的（政治、法律、社会规范——因国家而异）")
    text("- Naively, one might think safety is about refusal and is at odds with capability, but there's more... <br/> - 天真地，人们可能认为安全是关于拒绝的，与能力相矛盾，但还有更多...")
    text("- Hallucinations in a medical setting makes systems more capable and more safe <br/> - 医疗环境中的幻觉使系统更有能力也更安全")

    text("Two aspects of a model that reduce safety: capabilities + propensity <br/> 降低模型安全性的两个方面：能力 + 倾向")
    text("- A system could be capable of doing something, but refuse to do it <br/> - 系统可能有能力做某事，但拒绝做")
    text("- For API models, propensity matters <br/> - 对于 API 模型，倾向很重要")
    text("- For open weight models, capability matters (since can easily fine-tune safety away) <br/> - 对于开源权重模型，能力很重要（因为可以轻易通过微调消除安全限制）")

    text("**Dual-use**: capable cybersecurity agents (do well on CyBench) can be used to hack into a system or to do penetration testing <br/> **双重用途**：有能力的网络安全智能体（在 CyBench 上表现好）可以用来入侵系统或进行渗透测试")
    text("CyBench is used by the safety institute as a safety evaluation, but is it really a capability evaluation? <br/> CyBench 被安全研究所用作安全评估，但它真的是能力评估吗？")


def realism():
    text("Language models are used heavily in practice: <br/> 语言模型在实践中被大量使用：")
    image("images/openai-100b-tokens.png", width=600); link(title=" [tweet]", url="https://x.com/sama/status/1756089361609981993")
    image("images/cursor-1b-lines.png", width=600); link(title=" [tweet]", url="https://x.com/amanrsanger/status/1916968123535880684")

    text("However, most existing benchmarks (e.g., MMLU) are far away from real-world use. <br/> 然而，大多数现有基准测试（如 MMLU）远离实际使用。")
    text("Live traffic from real people contain garbage, that's not always what we want either. <br/> 来自真实用户的实时流量包含垃圾，这也不总是我们想要的。")

    text("Two types of prompts: <br/> 两种类型的提示：")
    text("1. Quizzing: User knows the answer and trying to test the system (think standardized exams). <br/> 1. 测验：用户知道答案并试图测试系统（想想标准化考试）。")
    text("2. Asking: User doesn't know the answer is trying to use the system to get it. <br/> 2. 询问：用户不知道答案，试图使用系统来获取它。")
    text("Asking is more realistic and produces value for the user. <br/> 询问更真实，为用户创造价值。")

    text("### Clio (Anthropic) <br/> ### Clio (Anthropic)")
    link("https://arxiv.org/abs/2412.13678")
    text("- Use language models to analyze real user data <br/> - 使用语言模型分析真实用户数据")
    text("- Share general patterns of what people are asking <br/> - 分享人们询问的一般模式")
    image("images/clio-table4.png", width=700)

    text("### MedHELM <br/> ### MedHELM")
    link("https://arxiv.org/abs/2412.13678")
    text("- Previous medical benchmarks were based on standardized exams <br/> - 之前的医疗基准测试基于标准化考试")
    text("- 121 clinical tasks sourced from 29 clinicians, mixture of private and public datasets <br/> - 121 个临床任务来自 29 名临床医生，混合私人和公共数据集")
    image("https://crfm.stanford.edu/helm/assets/medhelm-overview-3ddfcd65.png", width=700)
    link(title="[MedHELM]", url="https://crfm.stanford.edu/helm/medhelm/latest/#/leaderboard")

    text("Unfortunately, realism and privacy are sometimes at odds with each other. <br/> 不幸的是，真实性和隐私有时相互矛盾。")


def validity():
    text("How do we know our evaluations are valid? <br/> 我们如何知道我们的评估是有效的？")

    text("### Train-test overlap <br/> ### 训练-测试重叠")
    text("- Machine learning 101: don't train on your test set <br/> - 机器学习 101：不要在测试集上训练")
    text("- Pre-foundation models (ImageNet, SQuAD): well-defined train-test splits <br/> - 基础模型之前 (ImageNet、SQuAD)：定义明确的训练-测试分割")
    text("- Nowadays: train on the Internet and don't tell people about your data <br/> - 如今：在互联网上训练，不告诉人们你的数据")

    text("Route 1: try to infer train-test overlap from model <br/> 路线 1：尝试从模型推断训练-测试重叠")
    text("- Exploit exchangeability of data points <br/> - 利用数据点的可交换性"), link("https://arxiv.org/pdf/2310.17623")
    image("images/contamination-exchangeability.png", width=600)

    text("Route 2: encourage reporting norms (e.g., people report confidence intervals) <br/> 路线 2：鼓励报告规范（如人们报告置信区间）")
    text("- Model providers should report train-test overlap <br/> - 模型提供者应报告训练-测试重叠"), link("https://arxiv.org/abs/2410.08385")

    text("### Dataset quality <br/> ### 数据集质量")
    text("- Fixed up SWE-Bench to produce SWE-Bench Verified <br/> - 修复 SWE-Bench 以产生 SWE-Bench Verified"), blog_link("https://openai.com/index/introducing-swe-bench-verified/")
    text("- Create Platinum versions of benchmarks <br/> - 创建基准的 Platinum 版本"), link("https://arxiv.org/abs/2502.03461")
    image("https://pbs.twimg.com/media/GjICXQlWkAAYnDS?format=jpg&name=4096x4096", width=700)
    image("https://pbs.twimg.com/media/GjICcGQXYAAM4o1?format=jpg&name=4096x4096", width=800)


def what_are_we_evaluating():
    text("What are we even evaluating? <br/> 我们到底在评估什么？")
    text("In other words, what are the rules of a game? <br/> 换句话说，游戏规则是什么？")

    text("Pre-foundation models, we evaluated **methods** (standardized train-test splits). <br/> 基础模型之前，我们评估**方法**（标准化训练-测试分割）。")
    text("Today, we're evaluating **models/systems** (anything goes). <br/> 今天，我们评估**模型/系统**（任何东西都可以）。")

    text("There are some exceptions... <br/> 有一些例外...")
    text("nanogpt speedrun: fixed data, compute time to get to a particular validation loss <br/> nanogpt 速跑：固定数据，计算时间以达到特定验证损失")
    image("images/karpathy-nanogpt-speedrun.png", width=600), x_link("https://x.com/karpathy/status/1846790537262571739")

    text("DataComp-LM: given a raw dataset, get the best accuracy using standard training pipeline <br/> DataComp-LM：给定原始数据集，使用标准训练管道获得最佳准确率"), link("https://arxiv.org/abs/2406.11794")

    text("Evaluating methods encourage algorithmic innovation from researchers. <br/> 评估方法鼓励研究人员进行算法创新。")
    text("Evaluating models/systems is useful for downstream users. <br/> 评估模型/系统对下游用户有用。")

    text("Either way, we need to define the rules of the game! <br/> 无论如何，我们需要定义游戏规则！")


if __name__ == "__main__":
    main()
