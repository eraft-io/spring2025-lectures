#!/bin/bash

# 构建 trace-viewer 静态网站，输出到 githubpagestatic 目录
# 用于 GitHub Pages 部署

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/trace-viewer"

# 检查是否需要安装依赖
if [ ! -d "node_modules" ] || [ ! -f "node_modules/.package-lock.json" ]; then
    echo "正在安装依赖..."
    npm install
    echo ""
fi

# 检查 rollup 原生模块是否存在（修复 npm 可选依赖 bug）
if [ ! -d "node_modules/@rollup/rollup-darwin-arm64" ]; then
    echo "检测到 rollup 平台依赖缺失，正在清理并重新安装..."
    rm -rf node_modules package-lock.json
    npm install
    echo ""
fi

# 创建输出目录
OUTPUT_DIR="$SCRIPT_DIR/githubpagestatic"
mkdir -p "$OUTPUT_DIR"

echo "正在构建静态网站..."
echo "输出目录: $OUTPUT_DIR"
echo ""

# 使用 vite 构建静态网站
npm run build -- --outDir "$OUTPUT_DIR"

echo ""
echo "构建完成！"
echo "静态文件已输出到: $OUTPUT_DIR"
echo ""
echo "GitHub Pages 部署说明:"
echo "1. 将 $OUTPUT_DIR 目录的内容推送到仓库的 gh-pages 分支"
echo "2. 或在仓库设置中启用 GitHub Pages，选择分支和目录"
