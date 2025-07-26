#!/bin/bash

# 启动模型监控Web服务器
# 这个脚本会启动HTTP服务器并在后台持续更新数据

echo "启动模型训练状态监控Web服务器..."

# 切换到项目目录
cd /home/kuangph/hf-starter/Web_Display

# 检查是否已有Web服务器在运行
if pgrep -f "web_server.py" > /dev/null; then
    echo "检测到web_server.py已在运行中"
    echo "如要重启，请先运行: ./stop_web_server.sh"
    exit 1
fi

# 自动检测可用端口
find_available_port() {
    for port in 8080 8081 8082 8083 8084; do
        if ! ss -tln | grep -q ":$port "; then
            echo $port
            return
        fi
    done
    echo "无法找到可用端口"
    exit 1
}

PORT=$(find_available_port)
if [ "$PORT" = "无法找到可用端口" ]; then
    echo "错误: 端口8080-8084都被占用"
    echo "请手动释放端口或使用其他端口"
    exit 1
fi

echo "使用端口: $PORT"

# 启动Web服务器
echo "正在启动Web服务器..."
nohup python3 web_server.py --port $PORT > web_server.log 2>&1 &

# 获取进程ID
PID=$!
echo "Web服务器已启动，进程ID: $PID"
echo "服务器端口: $PORT"
echo "日志文件: web_server.log"
echo ""
echo "访问地址："
echo "  本机访问: http://localhost:$PORT"
echo "  局域网访问: http://10.15.89.226:$PORT"
echo ""
echo "常用命令："
echo "  查看日志: tail -f web_server.log"
echo "  停止服务: ./stop_web_server.sh"
echo "  查看进程: ps aux | grep web_server.py"
