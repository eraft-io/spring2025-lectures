#!/bin/bash

# 启动 trace-viewer 本地开发服务器
# 用于访问课程轨迹可视化页面

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

echo "正在启动课程轨迹可视化服务..."
echo "服务启动后，请在浏览器中访问: http://localhost:5173"
echo ""

npm run dev
