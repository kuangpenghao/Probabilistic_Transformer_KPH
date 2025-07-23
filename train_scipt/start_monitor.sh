#!/bin/bash

echo "启动自动化训练任务监控脚本"
echo "========================================"
echo ""

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到 python3 命令"
    exit 1
fi

# 检查tmux是否安装
if ! command -v tmux &> /dev/null; then
    echo "错误: 未找到 tmux 命令"
    exit 1
fi

# 检查必要文件是否存在
required_files=("train_script.py" "run_clm.py" "run_clm.sh")
for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        echo "错误: 缺少必要文件 $file"
        exit 1
    fi
done

echo "环境检查通过"
echo ""

# 显示当前tmux会话
echo "当前tmux会话:"
tmux ls 2>/dev/null || echo "   没有活跃的tmux会话"
echo ""

# 询问用户是否要继续
read -p "是否要启动监控脚本? (y/N): " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "用户取消操作"
    exit 0
fi

echo ""
echo "启动监控脚本..."
echo "提示: 使用 Ctrl+C 可以安全停止脚本"
echo ""

# 启动监控脚本
python3 train_script.py
