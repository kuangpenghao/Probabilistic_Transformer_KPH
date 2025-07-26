#!/bin/bash

# 快速启动模型监控系统
# 适合双击执行或添加到桌面快捷方式

SCRIPT_DIR="/home/kuangph/hf-starter/Web_Display"
cd "$SCRIPT_DIR"

# 检查是否已经有服务器在运行
if pgrep -f "web_server" > /dev/null; then
    echo "⚠️  检测到服务器已在运行，是否重启？"
    echo "1) 重启服务器"
    echo "2) 查看状态"
    echo "3) 退出"
    read -p "请选择 (1-3): " choice
    
    case $choice in
        1)
            ./manage_server.sh restart
            ;;
        2)
            ./manage_server.sh status
            ;;
        3)
            exit 0
            ;;
        *)
            echo "无效选择，退出"
            exit 1
            ;;
    esac
else
    echo "🚀 启动模型监控系统..."
    ./manage_server.sh start
fi

# 询问是否打开浏览器
echo ""
read -p "是否打开浏览器访问监控页面？(y/n): " open_browser

if [[ $open_browser =~ ^[Yy]$ ]]; then
    # 尝试打开浏览器
    if command -v xdg-open > /dev/null; then
        xdg-open "http://localhost:8085"
    elif command -v firefox > /dev/null; then
        firefox "http://localhost:8085" &
    elif command -v google-chrome > /dev/null; then
        google-chrome "http://localhost:8085" &
    else
        echo "请手动打开浏览器访问: http://localhost:8085"
    fi
fi

echo ""
echo "🎯 管理命令:"
echo "   查看状态: ./manage_server.sh status"
echo "   停止服务: ./manage_server.sh stop"
echo "   查看日志: ./manage_server.sh logs"
