# Spring 2025 CS336 lectures

This repo contains the lecture materials for "Stanford CS336: Language modeling from scratch".

## Non-executable (ppt/pdf) lectures

Located in `nonexecutable/`as PDFs

## Executable lectures

Located as `lecture_*.py` in the root directory

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
