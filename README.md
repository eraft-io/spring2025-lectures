# Spring 2025 CS336 lectures

This repo contains the lecture materials for "Stanford CS336: Language modeling from scratch".

## Non-executable (ppt/pdf) lectures

Located in `nonexecutable/` as PDFs

| 文件 | 主题 | 内容概要 |
|------|------|----------|
| `2025 Lecture 3 - architecture.pdf` | 模型架构详解 | Transformer架构深入分析，注意力变体，位置编码方案，归一化技术 |
| `2025 Lecture 4 - MoEs.pdf` | 混合专家模型 (MoE) | 稀疏激活机制，专家路由算法，Switch Transformer，MoE训练策略 |
| `2025 Lecture 5 - GPUs.pdf` | GPU硬件架构 | NVIDIA GPU架构详解，Tensor Core，内存带宽与计算瓶颈分析 |
| `2025 Lecture 7 - Parallelism basics.pdf` | 并行计算基础 | 数据并行、模型并行、流水线并行原理与实现 |
| `2025 Lecture 9 - Scaling laws basics.pdf` | 扩展定律基础 | Chinchilla扩展定律，计算最优训练，模型大小与数据量的权衡 |
| `2025 Lecture 11 - Scaling details.pdf` | 扩展细节深入 | 训练不稳定性，学习率调度，批量大小扩展，精度训练技术 |
| `2025 Lecture 15 - RLHF Alignment.pdf` | RLHF对齐 | 人类反馈强化学习，奖励模型训练，PPO算法，安全性考虑 |
| `2025 Lecture 16 - RLVR.pdf` | 可验证奖励的强化学习 | 基于规则/验证器的奖励，数学推理训练，代码生成优化 |

## Executable lectures

Located as `lecture_*.py` in the root directory

### Lecture 内容简介

| 文件 | 主题 | 内容概要 |
|------|------|----------|
| `lecture_01.py` | 基础Transformer实现 | 从零开始实现Transformer架构，包括注意力机制、位置编码、层归一化等核心组件 |
| `lecture_02.py` | 分词与嵌入 | 文本分词算法（BPE、WordPiece）、词嵌入、位置编码的实现与可视化 |
| `lecture_06.py` | MLP与CUDA优化 | 多层感知机实现，CUDA内核编程基础，矩阵乘法优化 |
| `lecture_08.py` | GPU加速原理 | GPU架构详解，内存层次结构，核函数优化技巧 |
| `lecture_10.py` | 推理优化 | 推理工作负载分析，KV缓存优化，量化、剪枝、投机采样等加速技术 |
| `lecture_12.py` | 模型评估 | 困惑度计算，各类基准测试（MMLU、HellaSwag等），评估方法论 |
| `lecture_13.py` | 训练数据 | 预训练数据来源（Common Crawl、Wikipedia等），数据清洗与过滤，版权法律问题 |
| `lecture_14.py` | 数据过滤与去重 | KenLM、fastText分类器，Bloom过滤器，MinHash/LSH去重算法 |
| `lecture_17.py` | 强化学习 | 策略梯度算法，GRPO优化，基线函数，优势函数，RL训练流程 |

### 运行方式

You can compile a lecture by running:

        python execute.py -m lecture_01

which generates a `var/traces/lecture_01.json` and caches any images as
appropriate.

However, if you want to run it on the cluster, you can do:

        ./remote_execute.sh lecture_01

which copies the files to our slurm cluster, runs it there, and copies the
results back.  You have to setup the appropriate environment and tweak some
configs to make this work (these instructions are not complete).

### Frontend

If you need to tweak the Javascript:

Install (one-time):

        npm create vite@latest trace-viewer -- --template react
        cd trace-viewer
        npm install

Load a local server to view at `http://localhost:5173?trace=var/traces/sample.json`:

        npm run dev

Deploy to the main website:

        cd trace-viewer
        npm run build
        git add dist/assets
        # then commit to the repo and it should show up on the website

## 本地预览与部署

### 快速启动本地开发服务器

一键启动本地开发服务器（自动安装依赖）：

        ./start_server.sh

访问地址: `http://localhost:5173/?trace=../var/traces/lecture_01.json`

### 构建静态网站

构建用于 GitHub Pages 部署的静态网站：

        ./build_static.sh

静态文件将输出到 `githubpagestatic/` 目录。

### 本地预览静态网站

构建完成后，可以在本地预览静态网站效果：

        ./preview_static.sh

访问地址: `http://localhost:8080/spring2025-lectures/?trace=/spring2025-lectures/var/traces/lecture_01.json`

### 部署到 GitHub Pages

#### 方式一：使用 gh-pages 分支

1. 构建静态网站：

        ./build_static.sh

2. 将 `githubpagestatic/` 目录的内容推送到 `gh-pages` 分支：

        cd githubpagestatic
        git init
        git checkout -b gh-pages
        git add -A
        git commit -m "Deploy to GitHub Pages"
        git remote add origin git@github.com:YOUR_USERNAME/spring2025-lectures.git
        git push -f origin gh-pages

3. 在仓库 Settings > Pages 中，选择 `gh-pages` 分支作为 Source

#### 方式二：使用 GitHub Actions 自动部署

1. 在仓库中创建 `.github/workflows/deploy.yml` 文件
2. 配置工作流在 push 时自动构建并部署
3. 在 Settings > Pages 中选择 "GitHub Actions" 作为 Source

部署成功后，访问地址为：`https://YOUR_USERNAME.github.io/spring2025-lectures/?trace=/spring2025-lectures/var/traces/lecture_01.json`
