#!/bin/bash

# 本地预览 githubpagestatic 静态网站
# 使用 Python HTTP 服务器模拟 GitHub Pages 环境

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATIC_DIR="$SCRIPT_DIR/githubpagestatic"

# 检查静态目录是否存在
if [ ! -d "$STATIC_DIR" ]; then
    echo "错误: 目录 $STATIC_DIR 不存在"
    echo "请先运行 ./build_static.sh 构建静态网站"
    exit 1
fi

# 创建符号链接以匹配 GitHub Pages 路径 /spring2025-lectures/
LINK_PATH="$STATIC_DIR/spring2025-lectures"
if [ ! -L "$LINK_PATH" ]; then
    ln -s "$STATIC_DIR" "$LINK_PATH"
fi

# 查找可用端口
PORT=8080
while lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1; do
    PORT=$((PORT + 1))
done

echo "正在启动本地 HTTP 服务器预览静态网站..."
echo "预览地址: http://localhost:$PORT/spring2025-lectures/?trace=/spring2025-lectures/var/traces/lecture_01.json"
echo "静态目录: $STATIC_DIR"
echo ""
echo "按 Ctrl+C 停止服务器"
echo ""

# 进入静态目录并启动服务器
cd "$STATIC_DIR"

# 尝试使用 Python 3
if command -v python3 >/dev/null 2>&1; then
    python3 -m http.server $PORT
# 尝试使用 Python 2
elif command -v python >/dev/null 2>&1; then
    python -m SimpleHTTPServer $PORT
else
    echo "错误: 未找到 Python，无法启动 HTTP 服务器"
    exit 1
fi
