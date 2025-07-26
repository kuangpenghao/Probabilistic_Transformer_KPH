#!/bin/bash

# 停止自动更新模型状态的脚本

echo "正在停止模型状态自动更新服务..."

# 查找并停止auto_update.py进程
if pgrep -f "auto_update.py" > /dev/null; then
    pkill -f "auto_update.py"
    echo "✓ 自动更新服务已停止"
else
    echo "! 没有找到正在运行的自动更新服务"
fi

# 显示进程状态
echo ""
echo "当前进程状态："
ps aux | grep auto_update.py | grep -v grep || echo "没有相关进程在运行"
