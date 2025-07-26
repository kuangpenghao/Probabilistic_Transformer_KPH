#!/bin/bash

# 启动自动更新模型状态的脚本
# 这个脚本会在后台运行，每10秒更新一次models_summary.json

echo "启动模型状态自动更新服务..."

# 切换到项目目录
cd /home/kuangph/hf-starter/Web_Display

# 检查是否已有进程在运行
if pgrep -f "auto_update.py" > /dev/null; then
    echo "检测到auto_update.py已在运行中"
    echo "如要重启，请先运行: pkill -f auto_update.py"
    exit 1
fi

# 启动自动更新脚本
echo "正在启动自动更新脚本..."
nohup python3 auto_update.py > auto_update.log 2>&1 &

# 获取进程ID
PID=$!
echo "自动更新脚本已启动，进程ID: $PID"
echo "日志文件: auto_update.log"
echo ""
echo "常用命令："
echo "  查看日志: tail -f auto_update.log"
echo "  停止服务: pkill -f auto_update.py"
echo "  查看进程: ps aux | grep auto_update.py"
