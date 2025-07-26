#!/bin/bash

# 停止模型监控Web服务器

echo "🛑 正在停止模型监控Web服务器..."

# 查找并停止所有web_server相关进程
echo "查找web_server进程..."
web_processes=$(pgrep -f "web_server" | wc -l)

if [ $web_processes -gt 0 ]; then
    echo "找到 $web_processes 个web_server进程，正在停止..."
    
    # 优雅停止
    pkill -f "web_server"
    echo "✓ 发送停止信号"
    
    # 等待进程停止
    sleep 3
    
    # 检查是否还有残留进程
    remaining_processes=$(pgrep -f "web_server" | wc -l)
    if [ $remaining_processes -gt 0 ]; then
        echo "⚠️  发现 $remaining_processes 个残留进程，强制结束..."
        pkill -9 -f "web_server"
        sleep 1
    fi
    
    echo "✅ Web服务器已停止"
else
    echo "ℹ️  没有找到正在运行的Web服务器"
fi

# 显示当前状态
echo ""
echo "📊 当前状态检查："

# 检查进程状态
remaining=$(pgrep -f "web_server" | wc -l)
if [ $remaining -eq 0 ]; then
    echo "✅ 没有web_server进程在运行"
else
    echo "❌ 仍有 $remaining 个web_server进程在运行:"
    ps aux | grep web_server | grep -v grep
fi

# 检查端口状态
echo ""
echo "🌐 端口状态检查："
ports_in_use=$(ss -tlnp | grep -E ":808[0-9]" | wc -l)
if [ $ports_in_use -eq 0 ]; then
    echo "✅ 端口8080-8089都已释放"
else
    echo "⚠️  以下端口仍在使用:"
    ss -tlnp | grep -E ":808[0-9]"
fi

echo ""
echo "🎯 服务器已完全停止!"
