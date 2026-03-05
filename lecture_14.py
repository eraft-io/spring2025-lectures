from dataclasses import dataclass
import math
import torch
import torch.nn as nn
from torch.nn.functional import softmax
import numpy as np
import kenlm
import fasttext
import itertools
import mmh3
from bitarray import bitarray
from basic_util import count, repeat
from file_util import download_file
from execute_util import text, image, link
from lecture_util import article_link, named_link
from references import dolma

def main():
    text("Last lecture: overview of datasets used for training language models <br/> 上一讲：用于训练语言模型的数据集概述")
    text("- Live service (GitHub) → dump/crawl (GH Archive) → processed data (The Stack) <br/> - 实时服务（GitHub）→ 转储/爬取（GH Archive）→ 处理数据（The Stack）")
    text("- Processing: HTML to text, language/quality/toxicity filtering, deduplication <br/> - 处理：HTML 转文本、语言/质量/毒性过滤、去重")

    text("This lecture: deep dive into the mechanics <br/> 本讲：深入探讨机制")
    text("- Algorithms for filtering (e.g., classifiers) <br/> - 过滤算法（如分类器）")
    text("- Applications of filtering (e.g., language, quality, toxicity) <br/> - 过滤应用（如语言、质量、毒性）")
    text("- Deduplication (e.g., Bloom filters, MinHash, LSH) <br/> - 去重（如 Bloom 过滤器、MinHash、LSH）")

    filtering_algorithms()
    filtering_applications()
    deduplication()

    text("### Summary <br/> ### 总结")
    text("- Algorithmic tools: n-gram models (KenLM), classifiers (fastText), importance resampling (DSIR) <br/> - 算法工具：n-gram 模型（KenLM）、分类器（fastText）、重要性重采样（DSIR）")
    text("- Applications: language identification, quality filtering, toxicity filtering <br/> - 应用：语言识别、质量过滤、毒性过滤")
    text("- Deduplication: hashing scales to large datasets for fuzzy matching <br/> - 去重：哈希扩展到大型数据集进行模糊匹配")
    text("- Now you have the tools (mechanics), just have to spend time with data (intuitions) <br/> - 现在你有工具（机制），只需要花时间处理数据（直觉）")


def filtering_algorithms():
    text("Algorithmic building block: <br/> 算法构建块：")
    text("- Given some **target data** T and lots of **raw data** R, find subset T' of R similar to T. <br/> - 给定一些**目标数据** T 和大量**原始数据** R，找到 R 中与 T 相似的子集 T'。")
    image("images/raw-target-schema.png", width=600)

    text("Desiderata for filtering algorithm: <br/> 过滤算法的期望特性：")
    text("- Generalize from the target data (want T and T' to be different) <br/> - 从目标数据泛化（希望 T 和 T' 不同）")
    text("- Extremely fast (have to run it on R, which is huge) <br/> - 极快（必须在 R 上运行，R 很大）")

    kenlm_main()         # Train n-gram model
    fasttext_main()      # Train a classifier
    dsir_main()          # Train bag of n-grams model, do importance resampling
    filtering_summary()

    text("Survey paper on data selection <br/> 数据选择综述论文"), link("https://arxiv.org/abs/2402.16827")


def kenlm_main():
    text("**n-gram model with Kneser-Ney smoothing** <br/> **带有 Kneser-Ney 平滑的 n-gram 模型**"), article_link("https://en.wikipedia.org/wiki/Kneser%E2%80%93Ney_smoothing")
    text("- KenLM: fast implementation originally for machine translation <br/> - KenLM：最初用于机器翻译的快速实现"), named_link("code", "https://kheafield.com/code/kenlm/")
    text("- Common language model used for data filtering <br/> - 用于数据过滤的通用语言模型")
    text("- Extremely simple / fast - just count and normalize <br/> - 极其简单/快速 - 只需计数和归一化")

    text("### Concepts <br/> ### 概念")
    text("Maximum likelihood estimation of n-gram language model: <br/> n-gram 语言模型的最大似然估计：")
    text("- n = 3: p(in | the cat) = count(the cat in) / count(the cat) <br/> - n = 3：p(in | the cat) = count(the cat in) / count(the cat)")
    text("Problem: sparse counts (count of many n-grams is 0 for large n) <br/> 问题：稀疏计数（对于大 n，许多 n-gram 的计数为 0）")
    text("Solution: Use Kneser-Ney smoothing to handle unseen n-grams <br/> 解决方案：使用 Kneser-Ney 平滑处理未见过的 n-gram"), article_link("https://en.wikipedia.org/wiki/Kneser%E2%80%93Ney_smoothing")
    text("- p(in | the cat) depends on p(in | cat) too <br/> - p(in | the cat) 也依赖于 p(in | cat)")

    # Download a KenLM language model
    model_url = "https://huggingface.co/edugp/kenlm/resolve/main/wikipedia/en.arpa.bin"
    model_path = "var/en.arpa.bin"
    download_file(model_url, model_path)
    model = kenlm.Model(model_path)

    # Use the language model
    def compute(content: str):
        # Hacky preprocessing
        content = "<s> " + content.replace(",", " ,").replace(".", " .") + " </s>"

        # log p(content)
        score = model.score(content)

        # Perplexity normalizes by number of tokens to avoid favoring short documents
        num_tokens = len(list(model.full_scores(content)))
        perplexity = math.exp(-score / num_tokens)

        return score, perplexity

    score, perplexity = compute("Stanford University was founded in 1885 by Leland and Jane Stanford as a tribute to the memory of their only child, Leland Stanford Jr.")  # @inspect score, @inspect perplexity
    score, perplexity = compute("If you believe that the course staff made an objective error in grading, you may submit a regrade request on Gradescope within 3 days after the grades are released.")  # @inspect score, @inspect perplexity
    score, perplexity = compute("asdf asdf asdf asdf asdf")  # @inspect score, @inspect perplexity
    score, perplexity = compute("the the the the the the the the the the the the the the the the")  # @inspect score, @inspect perplexity

    text("### CCNet <br/> ### CCNet")
    link("https://arxiv.org/pdf/1911.00359")
    text("- Items are paragraphs of text <br/> - 项目是文本段落")
    text("- Sort paragraphs by increasing perplexity <br/> - 按困惑度递增排序段落")
    text("- Keep the top 1/3 <br/> - 保留前 1/3")
    text("- Was used in LLaMA <br/> - 用于 LLaMA")

    text("Summary: Kneser-Ney n-gram language models (with KenLM implementation) is fast but crude")


def fasttext_main():
    text("fastText classifier "), link("https://arxiv.org/pdf/1607.01759")
    text("- Task: text classification (e.g., sentiment classification)")
    text("- Goal was to train a fast classifier for text classification")
    text("- They found it was as good as much slower neural network classifiers")

    text("### Baseline: bag of words (not what they did)")
    L = 32                              # Length of input
    V = 8192                            # Vocabulary size
    K = 64                              # Number of classes
    W = nn.Embedding(V, K)              # Embedding parameters (V x K)
    x = torch.randint(V, (L,))          # Input tokens (L) - e.g., ["the", "cat", "in", "the", "hat"]
    y = softmax(W(x).mean(dim=0))       # Output probabilities (K)
    text("Problem: V*K parameters (could be huge)")

    text("### fastText classifier: bag of word embeddings")
    H = 16                              # Hidden dimension
    W = nn.Embedding(V, H)              # Embedding parameters (V x H)
    U = nn.Linear(H, K)                 # Head parameters (H x K)
    y = softmax(U(W(x).mean(dim=0)))    # Output probabilities (K)
    text("Only H*(V + K) parameters")

    text("Implementation:")
    text("- Parallelized, asynchronous SGD")
    text("- Learning rate: linear interpolation from [some number] to 0 "), article_link("https://github.com/facebookresearch/fastText/blob/main/src/fasttext.cc#L653")

    text("### Bag of n-grams")
    x = ["the cat", "cat in", "in the", "the hat"]  # @inspect x
    text("Problem: number of bigrams can get large (and also be unbounded)")
    text("Solution: hashing trick")
    num_bins = 8  # In practice, 10M bins
    hashed_x = [mmh3.hash(bigram) % num_bins for bigram in x]  # @inspect hashed_x

    text("- For quality filtering, we have K = 2 classes (good versus bad)")
    text("- In that case, fastText is just a linear classifier (H = K = 2)")

    text("In general, can use any classifier (e.g., BERT, Llama), it's just slower")


def dsir_main():
    text("Data Selection for Language Models via Importance Resampling (DSIR) "), link("https://arxiv.org/abs/2302.03169")
    image("https://www.jinghong-chen.net/content/images/size/w1200/2023/12/Screenshot-2023-12-24-at-17.41.38.png", width=600)

    importance_sampling()

    text("Setup:")
    text("- Target dataset D_p (small)")
    text("- Proposal (raw) dataset D_q (large)")

    text("Take 1:")
    text("- Fit target distribution p to D_p")
    text("- Fit proposal distribution q to D_q")
    text("- Do importance resampling with p, q, and raw samples D_q")
    text("Problem: target data D_p is too small to estimate a good model")

    text("Take 2: use hashed n-grams")
    training_text = "the cat in the hat"

    # Hash the n-grams
    num_bins = 4
    def get_hashed_ngrams(text: str):
        ngrams = text.split(" ")  # Unigram for now
        return [mmh3.hash(ngram) % num_bins for ngram in ngrams]

    training_hashed_ngrams = get_hashed_ngrams(training_text)  # @inspect training_hashed_ngrams

    # Learn unigram model
    probs = [count(training_hashed_ngrams, x) / len(training_hashed_ngrams) for x in range(num_bins)]  # @inspect probs

    # Evaluate probability of any sentence
    hashed_ngrams = get_hashed_ngrams("the text")  # @inspect hashed_ngrams
    prob = np.prod([probs[x] for x in hashed_ngrams])  # @inspect prob
    text("Result: DSIR slightly better than heuristic classification (fastText) on the [GLUE](https://gluebenchmark.com/) benchmark")
    image("images/dsir-results.png", width=700)
    
    text("Comparison with fastText:")
    text("- Modeling distributions is a more principled approach capturing diversity")
    text("- Similar computation complexity")
    text("- Both can be improved by better modeling")


def importance_sampling():
    text("Setup:")
    text("- Target distribution p (want samples from here)")
    text("- Proposal distribution q (have samples from here)")

    vocabulary = [0, 1, 2, 3]
    p = [0.1, 0.2, 0.3, 0.4]
    q = [0.4, 0.3, 0.2, 0.1]

    # 1. Sample from q
    n = 100
    samples = np.random.choice(vocabulary, p=q, size = n)  # @inspect samples
    text(f"Samples (q): {samples}")

    # 2. Compute weights over samples (w \propto p/q)
    w = [p[x] / q[x] for x in samples]  # @inspect w
    z = sum(w)  # @inspect z
    w = [w_i / z for w_i in w]  # @inspect w

    # 3. Resample
    samples = np.random.choice(samples, p=w, size=n)  # @inspect samples
    text(f"Resampled (p): {samples}")


def filtering_summary():
    text("Implementations: KenLM, fastText, DSIR")

    text("### General framework")
    text("Given target T and raw R, find subset of R similar to T")
    text("1. Estimate some model based on R and T and derive a scoring function")
    text("2. Keep examples in R based on their score")

    text("### Instantiations of the framework")

    text("Generative model of T (KenLM):")
    text("1. score(x) = p_T(x)")
    text("2. Keep examples x with score(x) >= threshold (stochastically)")

    text("Discriminative classifier (fastText):")
    text("1. score(x) = p(T | x)")
    text("2. Keep examples x with score(x) >= threshold (stochastically)")

    text("Importance resampling (DSIR):")
    text("1. score(x) = p_T(x) / p_R(x)")
    text("2. Resample examples x with probability proportional to score(x)")


def filtering_applications():
    text("The same data filtering machinery can be used for different filtering tasks.")
    language_identification()
    quality_filtering()
    toxicity_filtering()


def language_identification():
    text("Language identification: find text of a specific language (e.g., English)")

    text("Why not just go multilingual?")
    text("- Data: difficult to do curation / processing of high-quality data in any given language")
    text("- Compute: in computed-limited regime, less compute/tokens dedicated to any given language")
    text("Models differ on multilinguality:")
    text("- English was only 30% of BLOOM (was undertrained), English performance suffered "), link("https://arxiv.org/pdf/2303.03915")
    text("- Most frontier models (GPT-4, Claude, Gemini, Llama, Qwen) are heavily multilingual (sufficiently trained)")

    text("fastText language identification "), article_link("https://fasttext.cc/docs/en/language-identification.html")
    text("- Off-the-shelf classifier")
    text("- Supports 176 languages")
    text("- Trained on multilingual sites: Wikipedia, Tatoeba (translation site) and SETimes (Southeast European news)")

    text("Example: Dolma keeps pages with p(English) >= 0.5 "), link(dolma)
    
    # Download the model
    model_url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
    model_path = "var/lid.176.bin"
    download_file(model_url, model_path)
    model = fasttext.load_model(model_path)

    # Make predictions
    predictions = model.predict(["The quick brown fox jumps over the lazy dog."])  # English @inspect predictions
    predictions = model.predict(["The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog."])  # Duplicate @inspect predictions
    predictions = model.predict(["OMG that movie was 🔥🔥! So dope 😎🤘!"])  # Informal English @inspect predictions
    predictions = model.predict(["Auf dem Wasser zu singen"])  # German @inspect predictions
    predictions = model.predict(["The quadratic formula is $x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}$."])  # Latex @inspect predictions
    predictions = model.predict(["for (int i = 0; i < 10; i++)"])  # C++ @inspect predictions
    predictions = model.predict(["Hello!"])  # English @inspect predictions
    predictions = model.predict(["Bonjour!"])  # French @inspect predictions
    predictions = model.predict(["Feliz Navidad / Próspero año y felicidad / I wanna wish you a Merry Christmas"])  # Spanish + English @inspect predictions

    text("Caveats:")
    text("- Difficult for short sequences")
    text("- Difficult for low-resource languages")
    text("- Could accidentally filter out dialects of English")
    text("- Hard for similar languages (Malay and Indonesian)")
    text("- Ill-defined for code-switching (e.g., Spanish + English)")

    text("OpenMathText "), link("https://arxiv.org/pdf/2310.06786")
    text("- Goal: curate large corpus of mathematical text from CommonCrawl")
    text("- Use rules to filter (e.g., contains latex commands)")
    text("- KenLM trained on ProofPile, keep if perplexity < 15000")
    text("- Trained fastText classifier to predict mathematical writing, threshold is 0.17 if math, 0.8 if no math")
    text("Result: produced 14.7B tokens, used to train 1.4B models that do better than models trained on 20x data")


def quality_filtering():
    text("- Some deliberately do not use model-based filtering (C4, Gopher, RefinedWeb, FineWeb, Dolma)")
    text("- Some use model-based filtering (GPT-3, LLaMA, DCLM) [becoming the norm]")

    text("**GPT-3** "), link("https://arxiv.org/pdf/2005.14165")  # Appendix A
    text("- Positives: samples from {Wikipedia, WebText2, Books1, Books2}")
    text("- Negatives: samples from CommonCrawl <br/> - 负例：来自 CommonCrawl 的样本")
    image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/11/Probability_density_function_of_Pareto_distribution.svg/325px-Probability_density_function_of_Pareto_distribution.svg.png", width=0.5)
    text("Train linear classifier based on word features "), article_link("https://spark.apache.org/docs/latest/ml-features#tokenizer")
    text("Keep documents stochastically based on score")
    def keep_document(score: float) -> bool:
        return np.random.pareto(9) > 1 - score

    text("** LLaMA/RedPajama** <br/> ** LLaMA/RedPajama**"), link("https://arxiv.org/pdf/2302.13971")
    text("- Positives: samples from pages **referenced** by Wikipedia <br/> - 正例：来自维基百科**引用**页面的样本")
    text("- Negatives: samples from CommonCrawl <br/> - 负例：来自 CommonCrawl 的样本")
    text("- Keep documents that are classified positive <br/> - 保留被分类为正例的文档")

    text("**phi-1** <br/> **phi-1**"), link("https://arxiv.org/pdf/2306.11644")
    text("Philosophy: really high quality data (textbooks) to train a small model (1.5B) <br/> 理念：用真正高质量的数据（教科书）训练小模型（1.5B）")
    text("Includes synthetic data from GPT 3.5 (later: GPT-4) and filtered data <br/> 包括来自 GPT 3.5（后来：GPT-4）的合成数据和过滤数据")

    R = "Python subset of the Stack"   # Raw data
    prompt = "determine its educational value for a student whose goal is to learn basic coding concepts"
    T = "Use GPT-4 with this prompt to classify 100K subset of R to get positive examples"
    text("Train random forest classifier on T using output embedding from pretrained codegen model <br/> 使用预训练 codegen 模型的输出嵌入在 T 上训练随机森林分类器")
    text("Select data from R that is classified positive by the classifier <br/> 从 R 中选择被分类器分类为正例的数据")

    text("Result on [HumanEval](https://huggingface.co/datasets/openai_humaneval): <br/> [HumanEval](https://huggingface.co/datasets/openai_humaneval) 的结果：")
    text("- Train 1.3B LM on Python subset of The Stack (performance: 12.19% after 96K steps) <br/> - 在 The Stack 的 Python 子集上训练 1.3B LM（性能：96K 步后 12.19%）")
    text("- Train 1.3B LM on new filtered subset (performance: 17.68% after 36K steps) - better! <br/> - 在新的过滤子集上训练 1.3B LM（性能：36K 步后 17.68%）- 更好！")


@dataclass
class Example:
    text: str
    label: int


def toxicity_filtering():
    # WARNING: potentially offensive content below
    text("Toxicity filtering in Dolma <br/> Dolma 中的毒性过滤"), link(dolma)
    
    text("Dataset: Jigsaw Toxic Comments dataset (2018) <br/> 数据集：Jigsaw 毒性评论数据集（2018）"), named_link("dataset", "https://www.kaggle.com/datasets/julian3833/jigsaw-toxic-comment-classification-challenge")
    text("- Project goal: help people have better discussions online <br/> - 项目目标：帮助人们在网上进行更好的讨论"), article_link("https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/discussion/46064")
    text("- Data: comments on Wikipedia talk page annotated with {toxic, severe_toxic, obscene, threat, insult, identity_hate} <br/> - 数据：维基百科讨论页上的评论，标记为 {toxic, severe_toxic, obscene, threat, insult, identity_hate}")

    text("Trained 2 fastText classifiers <br/> 训练了 2 个 fastText 分类器")
    text("- hate: positive = {unlabeled, obscene}, negative = all else <br/> - hate：正例 = {unlabeled, obscene}，负例 = 其他所有")
    text("- NSFW: positive = {obscene}, negative = all else <br/> - NSFW：正例 = {obscene}，负例 = 其他所有")

    # Examples from the dataset: (obscene, text)
    train_examples = [
        Example(label=0, text="Are you threatening me for disputing neutrality? I know in your country it's quite common to bully your way through a discussion and push outcomes you want. But this is not Russia."),
        Example(label=1, text="Stupid peace of shit stop deleting my stuff asshole go die and fall in a hole go to hell!"),
    ]

    # Download model
    model_url = "https://dolma-artifacts.org/fasttext_models/jigsaw_fasttext_bigrams_20230515/jigsaw_fasttext_bigrams_nsfw_final.bin"
    model_path = "var/jigsaw_fasttext_bigrams_nsfw_final.bin"
    download_file(model_url, model_path)
    model = fasttext.load_model(model_path)

    # Make predictions
    predictions = model.predict([train_examples[0].text])  # @inspect predictions
    predictions = model.predict([train_examples[1].text])  # @inspect predictions
    predictions = model.predict(["I love strawberries"])  # @inspect predictions
    predictions = model.predict(["I hate strawberries"])  # @inspect predictions


def print_predict(model, content):
    """Run classifier `model` on `content` and print out the results."""
    predictions = model.predict([content])
    print(predictions)
    #labels, prob =
    #labels = ", ".join(labels)
    #text(f"{content} => {labels} {prob}")


def deduplication():
    text("Two types of duplicates: <br/> 两种重复类型：")
    text("- Exact duplicates (mirror sites, GitHub forks) <br/> - 精确重复（镜像站点、GitHub fork）"), named_link("Gutenberg mirrors", "https://www.gutenberg.org/MIRRORS.ALL")
    text("- Near duplicates: same text differing by a few tokens <br/> - 近似重复：相同文本仅相差几个 token")

    text("Examples of near duplicates: <br/> 近似重复的示例：")
    text("- Terms of service and licenses <br/> - 服务条款和许可证"), named_link("MIT license", "https://opensource.org/license/mit")
    text("- Formulaic writing (copy/pasted or generated from a template) <br/> - 公式化写作（复制/粘贴或从模板生成）"), image("https://d3i71xaburhd42.cloudfront.net/4566c0d22ebf3c31180066ab23b6c445aeec78d5/5-Table1-1.png", width=600)
    text("- Minor formatting differences in copy/pasting <br/> - 复制/粘贴中的轻微格式差异")

    text("Product description repeated 61,036 times in C4 <br/> C4 中重复了 61,036 次的产品描述")
    text("'“by combining fantastic ideas, interesting arrangements, and follow the current trends in the field of that make you more inspired and give artistic touches. We’d be honored if you can apply some or all of these design in your wedding.  believe me, brilliant ideas would be perfect if it can be applied in real and make the people around you amazed!")
    named_link("example page", "https://www.amazon.co.uk/suryagede-100-Graffiti-Gas-Mask/dp/B07CRHT3RG")

    text("Deduplication training data makes language models better <br/> 去重训练数据使语言模型更好"), link("https://arxiv.org/pdf/2107.06499")
    text("- Train more efficiently (because have fewer tokens) <br/> - 更高效地训练（因为 token 更少）")
    text("- Avoid memorization (can mitigate copyright, privacy concerns) <br/> - 避免记忆（可以缓解版权、隐私问题）")

    text("Design space: <br/> 设计空间：")
    text("1. What is an item (sentence, paragraph, document)? <br/> 1. 什么是项目（句子、段落、文档）？")
    text("2. How to match (exact match, existence of common subitem, fraction of common subitems)? <br/> 2. 如何匹配（精确匹配、公共子项的存在、公共子项的比例）？")
    text("3. What action to take (remove all, remove all but one)? <br/> 3. 采取什么行动（全部移除、只保留一个）？")

    text("Key challenge: <br/> 关键挑战：")
    text("- Deduplication is fundamentally about comparing items to other items <br/> - 去重本质上是关于将项目与其他项目进行比较")
    text("- Need linear time algorithms to scale <br/> - 需要线性时间算法来扩展")

    hash_functions()

    exact_deduplication()
    bloom_filter()

    jaccard_minhash()
    locality_sensitive_hashing()


def hash_functions():
    text("- Hash function h maps item to a hash value (integer or string) <br/> - 哈希函数 h 将项目映射到哈希值（整数或字符串）")
    text("- Hash value much smaller than item <br/> - 哈希值远小于项目")
    text("- Hash collision: h(x) = h(y) for x ≠ y <br/> - 哈希碰撞：h(x) = h(y) 当 x ≠ y")

    text("Tradeoff between efficiency and collision resistance <br/> 效率与抗碰撞性之间的权衡"),  article_link("https://softwareengineering.stackexchange.com/questions/49550/which-hashing-algorithm-is-best-for-uniqueness-and-speed")
    text("- Cryptographic hash functions (SHA-256): collision resistant, slow (used in bitcoin) <br/> - 加密哈希函数（SHA-256）：抗碰撞，慢（用于比特币）")
    text("- DJB2, MurmurHash, CityHash: not collision resistant, fast (used for hash tables) <br/> - DJB2、MurmurHash、CityHash：不抗碰撞，快（用于哈希表）")

    text("We will use MurmurHash: <br/> 我们将使用 MurmurHash：")
    h = mmh3.hash("hello")  # @inspect h


def exact_deduplication():
    text("**Simple example** <br/> **简单示例**")
    text("1. Item: string <br/> 1. 项目：字符串")
    text("2. How to match: exact match <br/> 2. 如何匹配：精确匹配")
    text("3. Action: remove all but one <br/> 3. 操作：只保留一个")

    # Original items
    items = ["Hello!", "hello", "hello there", "hello", "hi", "bye"]  # @inspect items

    # Compute hash -> list of items with that hash
    hash_items = itertools.groupby(sorted(items, key=mmh3.hash), key=mmh3.hash)

    # Keep one item from each group
    deduped_items = [next(group) for h, group in hash_items]  # @inspect deduped_items

    text("- Pro: simple, clear semantics, high precision <br/> - 优点：简单、语义清晰、高精度")
    text("- Con: does not deduplicate near duplicates <br/> - 缺点：不去重近似重复")
    text("- This code is written in a MapReduce way, can easily parallelize and scale <br/> - 这段代码以 MapReduce 方式编写，可以轻松并行化和扩展")

    text("**C4** <br/> **C4**"), link("https://arxiv.org/pdf/1910.10683v4")
    text("1. Item: 3-sentence spans <br/> 1. 项目：3 句跨度")
    text("2. How to match: use exact match <br/> 2. 如何匹配：使用精确匹配")
    text("3. Action: remove all but one <br/> 3. 操作：只保留一个")
    text("Warning: when a 3-sentence span is removed from the middle of a document, the resulting document might not be coherent <br/> 警告：当从文档中间移除 3 句跨度时，结果文档可能不连贯")


def bloom_filter():
    text("Goal: efficient, approximate data structure for testing set membership <br/> 目标：用于测试集合成员的高效近似数据结构")

    text("Features of Bloom filters <br/> Bloom 过滤器的特性")
    text("- Memory efficient <br/> - 内存高效")
    text("- Can update, but can't delete <br/> - 可以更新，但不能删除")
    text("- If return 'no', definitely 'no' <br/> - 如果返回'否'，肯定是'否'")
    text("- If return 'yes', most likely 'yes', but small probability of 'no' <br/> - 如果返回'是'，很可能是'是'，但有小概率是'否'")
    text("- Can drive the false positive rate down exponentially with more time/compute <br/> - 可以用更多时间/计算将误报率指数级降低")

    items = ["the", "cat", "in", "the", "hat"]
    non_items = ["what", "who", "why", "when", "where", "which", "how"]

    text("First, make the range of hash function small (small number of bins). <br/> 首先，使哈希函数的范围变小（少量桶）。")
    m = 8  # Number of bins
    table = build_table(items, m)
    for item in items:
        assert query_table(table, item, m) == 1
    result = {item: query_table(table, item, m) for item in non_items}  # @inspect result
    num_mistakes = count(result.values(), True)  # @inspect num_mistakes
    false_positive_rate = num_mistakes / (len(items) + num_mistakes)  # @inspect false_positive_rate
    text("Problem: false positives for small bins <br/> 问题：小桶的误报")

    text("Naive solution: increase the number of bins <br/> 朴素解决方案：增加桶的数量")
    text("Error probability is O(1/num_bins), decreases polynomially with memory <br/> 错误概率为 O(1/num_bins)，随内存多项式下降")

    text("Better solution: use more hash functions <br/> 更好的解决方案：使用更多哈希函数")
    k = 2  # Number of hash functions
    table = build_table_k(items, m, k)
    for item in items:
        assert query_table_k(table, item, m, k) == 1
    result = {item: query_table_k(table, item, m, k) for item in non_items}  # @inspect result
    num_mistakes = count(result.values(), 1)  # @inspect num_mistakes
    false_positive_rate = num_mistakes / (len(items) + num_mistakes)  # @inspect false_positive_rate
    text("Reduced the false positive rate! <br/> 降低了误报率！")

    false_positive_rate_analysis()


def false_positive_rate_analysis():
    text("Assume independence of hash functions and items <br/> 假设哈希函数和项目独立"), article_link("https://en.wikipedia.org/wiki/Bloom_filter")
    m = 1000   # Number of bins
    k = 10     # Number of hash functions
    n = 100    # Number of items we're inserting

    text("Consider a test input (not in the set) that would hash into a given test bin (say, i). <br/> 考虑一个测试输入（不在集合中），它将哈希到给定的测试桶（比如 i）。")
    text("Now consider putting items into the Bloom filter and seeing if it hits i. <br/> 现在考虑将项目放入 Bloom 过滤器，看看是否命中 i。")

    # Insert one item, ask if the test bin B(i) = 1?
    # B: [0 0 1 0 0 0 0 0 0 0] - have to miss 1 time
    f = 1 / m                              # P[B(i) = 1 after 1 insertion with 1 hash function]  # @inspect f
    # B: [0 0 1 0 0 1 0 1 0 0] - have to miss k times
    f = 1 - (1 - 1 / m) ** k               # P[B(i) = 1 after 1 insertion with k hash functions]  # @inspect f

    # Insert n items, ask if the test bin B(i) = 1?
    # Have to miss k*n times
    f = 1 - (1 - 1 / m) ** (k * n)         # P[B(i) = 1 after n insertions for 1 hash function]  # @inspect f
    # Get k chances to miss (since test input is hashed k times too)
    f = f ** k                             # P[B(i) = 1 after n insertions for k hash functions]  # @inspect f

    text("Optimal value of k (given fixed m / n ratio) [results in f ~ 0.5] <br/> k 的最优值（给定固定的 m/n 比率）[结果 f ~ 0.5]")
    k = math.log(2) * m / n  # @inspect k
    text("Resulting false positive rate (improved) <br/> 结果误报率（改进）")
    f = 0.5 ** k  # @inspect f

    text("Tradeoff between compute (k), memory (m), and false positive rate (f) <br/> 计算（k）、内存（m）和误报率（f）之间的权衡"), named_link("lecture notes", "https://people.eecs.berkeley.edu/~daw/teaching/cs170-s03/Notes/lecture10.pdf")

    text("Example: Dolma <br/> 示例：Dolma")
    text("- Set false positive rate to 1e-15 <br/> - 设置误报率为 1e-15")
    text("- Perform on items = paragraphs <br/> - 在项目 = 段落上执行")


def build_table(items: list[str], num_bins: int):
    """Build a Bloom filter table of size `num_bins`, inserting `items` into it."""
    table = bitarray(num_bins)  # @inspect table
    for item in items:
        h = mmh3.hash(item) % num_bins  # @inspect item, @inspect h
        table[h] = 1  # @inspect table
    return table


def build_table_k(items: list[str], num_bins: int, k: int):
    """Build a Bloom filter table of size `num_bins`, inserting `items` into it.
    Use `k` hash functions."""
    table = bitarray(num_bins)  # @inspect table
    for item in items:
        # For each of the k functions
        for seed in range(k):
            h = mmh3.hash(item, seed) % num_bins  # @inspect item, @inspect h, @inspect seed
            table[h] = 1  # @inspect table
    return table


def query_table(table: bitarray, item: str, num_bins: int, seed: int = 0):
    """Return whether `item` is in the `table`."""
    h = mmh3.hash(item, seed) % num_bins
    return table[h]


def query_table_k(table: bitarray, item: str, num_bins: int, k: int):
    """Return 1 if table set to 1 for all `k` hash functions."""
    return int(all(
        query_table(table, item, num_bins, seed)
        for seed in range(k)
    ))


def jaccard_minhash():
    text("Let's now look at approximate set membership. <br/> 现在让我们看看近似集合成员。")
    text("First we need a similarity measure. <br/> 首先我们需要一个相似度度量。")

    text("### Jaccard similarity <br/> ### Jaccard 相似度")
    text("Definition: Jaccard(A, B) = |A intersect B| / |A union B| <br/> 定义：Jaccard(A, B) = |A 交 B| / |A 并 B|")
    A = {"1", "2", "3", "4"}
    B = {"1", "2", "3", "5"}

    def compute_jaccard(A, B):
        intersection = len(A & B)  # @inspect intersection
        union = len(A | B)  # @inspect union
        return intersection / union
    jaccard = compute_jaccard(A, B)  # @inspect jaccard

    text("Definition: two documents are **near duplicates** if their Jaccard similarity >= threshold <br/> 定义：如果两个文档的 Jaccard 相似度 >= 阈值，则它们是**近似重复**")

    text("Algorithmic challenge: find near duplicates in linear time <br/> 算法挑战：在线性时间内找到近似重复")

    text("### MinHash <br/> ### MinHash")
    text("MinHash: a random hash function h so that Pr[h(A) = h(B)] = Jaccard(A, B) <br/> MinHash：一个随机哈希函数 h，使得 Pr[h(A) = h(B)] = Jaccard(A, B)")

    text("Normally, you want different items to hash to different hashes <br/> 通常，你希望不同项目哈希到不同哈希值")
    text("...but here, you want collision概率 to depend on similarity <br/> ...但在这里，你希望碰撞概率依赖于相似度")

    def minhash(S: set[str], seed: int):
        return min(mmh3.hash(x, seed) for x in S)

    text("Characteristic matrix representation: <br/> 特征矩阵表示：")
    text("item | A | B", verbatim=True)
    text("1    | 1 | 1", verbatim=True)
    text("2    | 1 | 1", verbatim=True)
    text("3    | 1 | 1", verbatim=True)
    text("4    | 1 | 0", verbatim=True)
    text("5    | 0 | 1", verbatim=True)

    text("Random hash function induces a permutation over items <br/> 随机哈希函数在项目上诱导排列")
    text("Look at which item is first in A and which item is first in B. <br/> 看哪个项目在 A 中第一个，哪个项目在 B 中第一个。")
    text("Each item has the same probability as being first (min) <br/> 每个项目有相同的概率成为第一个（最小）")
    text("- If 1, 2, 3 is first, then first in A = first in B. <br/> - 如果 1、2、3 是第一个，那么 A 中的第一个 = B 中的第一个。")
    text("- If 4, 5 is first, then first in A ≠ first in B. <br/> - 如果 4、5 是第一个，那么 A 中的第一个 ≠ B 中的第一个。")

    # Verify MinHash approximates Jaccard as advertised
    n = 100  # Generate this many random hash functions
    matches = [minhash(A, seed) == minhash(B, seed) for seed in range(n)]
    estimated_jaccard = count(matches, True) / len(matches)  # @inspect estimated_jaccard
    assert abs(estimated_jaccard - jaccard) < 0.01

    text("Now we can hash our items, but a collision doesn't tell us Jaccard(A, B) > threshold. <br/> 现在我们可以哈希我们的项目，但碰撞并不能告诉我们 Jaccard(A, B) > 阈值。")


def locality_sensitive_hashing():
    text("Locality sensitive hashing (LSH) <br/> 局部敏感哈希 (LSH)"), named_link("book chapter", "http://infolab.stanford.edu/~ullman/mmds/ch3n.pdf")

    text("Suppose we hash examples just one MinHash function <br/> 假设我们只用一个 MinHash 函数哈希示例")
    text("P[A and B collide] = Jaccard(A, B) <br/> P[A 和 B 碰撞] = Jaccard(A, B)")
    text("On average, more similar items will collide, but very stochastic... <br/> 平均而言，更相似的项目会碰撞，但非常随机...")

    text("Goal: have A and B collide if Jaccard(A, B) > threshold <br/> 目标：如果 Jaccard(A, B) > 阈值，让 A 和 B 碰撞")
    text("We have to somehow sharpen the probabilities... <br/> 我们必须以某种方式锐化概率...")

    text("Solution: use n hash functions <br/> 解决方案：使用 n 个哈希函数")
    text("Break up into b bands of r hash functions each (n = b * r) <br/> 分成 b 个带，每个带 r 个哈希函数（n = b * r）")

    n = 12      # Number of hash functions
    b = 3       # Number of bands
    r = 4       # Number of hash functions per band
    text("Hash functions: <br/> 哈希函数：")
    text("h1 h2 h3 h4  |  h5 h6 h7 h8  |  h9 h10 h11 h12", verbatim=True)

    text("Key: A and B collide if for *some* band, *all* its hash functions return same value <br/> 关键：如果对于*某个*带，*所有*其哈希函数返回相同值，则 A 和 B 碰撞")
    text("As we will see, the and-or structure of the bands sharpens the threshold <br/> 正如我们将看到的，带的与或结构锐化了阈值")

    text("Given Jaccard(A, B), what is the probability that A and B collide? <br/> 给定 Jaccard(A, B)，A 和 B 碰撞的概率是多少？")

    def get_prob_collision(sim, b, r):  # @inspect sim, @inspect b, @inspect r
        prob_match = sim ** r                        # Probability that a fixed band matches  @inspect prob_match
        prob_collision = 1 - (1 - prob_match) ** b   # Probability that some band matches  @inspect prob_collision
        return prob_collision

    text("**Example** <br/> **示例**")
    prob_collision = get_prob_collision(sim=0.8, b=5, r=10)  # @inspect prob_collision
    image("https://cdn.sanity.io/images/vr8gru94/production/b470799575b8e77911bacb8500977afef06d6c85-1280x720.png", width=600)


    sims = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.98]
    probs = {sim: get_prob_collision(sim=sim, b=10, r=10) for sim in sims}  # @inspect probs

    text("Increasing r sharpens the threshold and moves the curve to the right (harder to match) <br/> 增加 r 锐化阈值并将曲线向右移动（更难匹配）")
    probs = {sim: get_prob_collision(sim=sim, b=10, r=20) for sim in sims}  # @inspect probs

    text("Increasing b moves the curve to the left (easier to match) <br/> 增加 b 将曲线向左移动（更容易匹配）")
    probs = {sim: get_prob_collision(sim=sim, b=20, r=20) for sim in sims}  # @inspect probs
    image("https://cdn.sanity.io/images/vr8gru94/production/aace49fa240778e8ecf6e85ad08a2de7f5385566-1280x720.png", width=600)

    text("Example setting <br/> 示例设置"), link("https://arxiv.org/pdf/2107.06499"), text(": n = 9000, b = 20, r = 450")
    b = 20
    r = 450
    text("What is the threshold (where the phase transition happens)? <br/> 阈值是多少（相变发生的地方）？")
    threshold = (1 / b) ** (1 / r)  # @inspect threshold
    text("Probability that a fixed band matches: <br/> 固定带匹配的概率：")
    prob_match = (1 / b)  # @inspect prob_match
    text("Probability that A and B collide (≈ 1-1/e): <br/> A 和 B 碰撞的概率（≈ 1-1/e）：")
    prob_collision = 1 - (1 - 1 / b) ** b  #  @inspect prob_collision


if __name__ == "__main__":
    main()
