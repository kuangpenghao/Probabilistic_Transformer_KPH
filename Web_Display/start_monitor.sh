#!/bin/bash

# 模型监控系统启动脚本
# 解决连接超时和稳定性问题

echo "🚀 启动模型监控系统"
echo "===================="

# 检查并停止现有服务器
echo "检查现有服务器进程..."
if pgrep -f "web_server" > /dev/null; then
    echo "停止现有服务器..."
    pkill -f "web_server"
    sleep 2
fi

# 切换到正确目录
cd /home/kuangph/hf-starter/Web_Display

# 检查必要文件
if [ ! -f "HTMLProcessing.html" ]; then
    echo "❌ 错误: HTMLProcessing.html 文件不存在"
    exit 1
fi

if [ ! -f "JsonConclusion.py" ]; then
    echo "❌ 错误: JsonConclusion.py 文件不存在"
    exit 1
fi

if [ ! -f "web_server_fixed.py" ]; then
    echo "❌ 错误: web_server_fixed.py 文件不存在"
    exit 1
fi

# 启动服务器
echo "启动改进版服务器..."
nohup python3 web_server_fixed.py > web_server.log 2>&1 &
SERVER_PID=$!

# 等待服务器启动
sleep 3

# 检查服务器是否启动成功
if ss -tlnp | grep -q ":8084"; then
    echo "✅ 服务器启动成功!"
    echo ""
    echo "📊 访问地址:"
    echo "   本机访问: http://localhost:8084"
    echo "   局域网访问: http://10.15.89.226:8084"
    echo ""
    echo "🔧 控制命令:"
    echo "   查看日志: tail -f /home/kuangph/hf-starter/Web_Display/web_server.log"
    echo "   停止服务器: pkill -f 'web_server_fixed.py'"
    echo "   测试系统: python3 test_system.py"
    echo ""
    echo "🎯 服务器进程ID: $SERVER_PID"
    echo "📝 日志文件: /home/kuangph/hf-starter/Web_Display/web_server.log"
else
    echo "❌ 服务器启动失败"
    echo "检查日志文件: /home/kuangph/hf-starter/Web_Display/web_server.log"
    exit 1
fi
