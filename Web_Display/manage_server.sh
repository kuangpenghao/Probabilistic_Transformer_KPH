#!/bin/bash

# 模型监控系统管理脚本
# 用法: ./manage_server.sh [start|stop|restart|status|logs|test]

SCRIPT_DIR="/home/kuangph/hf-starter/Web_Display"
LOG_FILE="$SCRIPT_DIR/web_server.log"
PID_FILE="$SCRIPT_DIR/web_server.pid"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# 检查进程是否运行
is_server_running() {
    pgrep -f "web_server_fixed.py" > /dev/null
    return $?
}

# 获取服务器PID
get_server_pid() {
    pgrep -f "web_server_fixed.py"
}

# 启动服务器
start_server() {
    echo "🚀 启动模型监控系统"
    echo "===================="
    
    # 检查是否已经运行
    if is_server_running; then
        print_warning "服务器已经在运行中"
        show_status
        return 1
    fi
    
    # 切换到正确目录
    cd "$SCRIPT_DIR" || {
        print_error "无法切换到目录: $SCRIPT_DIR"
        return 1
    }
    
    # 检查必要文件
    required_files=("HTMLProcessing.html" "JsonConclusion.py" "web_server_fixed.py")
    for file in "${required_files[@]}"; do
        if [ ! -f "$file" ]; then
            print_error "缺少必要文件: $file"
            return 1
        fi
    done
    
    # 启动服务器
    print_info "启动改进版服务器..."
    nohup python3 web_server_fixed.py > "$LOG_FILE" 2>&1 &
    server_pid=$!
    
    # 保存PID
    echo $server_pid > "$PID_FILE"
    
    # 等待启动
    sleep 3
    
    # 检查启动状态
    if is_server_running; then
        print_status "服务器启动成功! (PID: $(get_server_pid))"
        echo ""
        echo "📊 访问地址:"
        echo "   本机访问: http://localhost:8085"
        echo "   局域网访问: http://10.15.89.226:8085"
        echo ""
        echo "🔧 管理命令:"
        echo "   查看状态: ./manage_server.sh status"
        echo "   查看日志: ./manage_server.sh logs"
        echo "   停止服务: ./manage_server.sh stop"
        echo "   运行测试: ./manage_server.sh test"
    else
        print_error "服务器启动失败"
        if [ -f "$LOG_FILE" ]; then
            echo "错误日志:"
            tail -10 "$LOG_FILE"
        fi
        return 1
    fi
}

# 停止服务器
stop_server() {
    echo "🛑 停止模型监控系统"
    echo "===================="
    
    if ! is_server_running; then
        print_info "服务器没有运行"
        return 0
    fi
    
    print_info "正在停止服务器..."
    
    # 优雅停止
    pkill -f "web_server_fixed.py"
    sleep 2
    
    # 检查是否仍在运行
    if is_server_running; then
        print_warning "强制停止服务器..."
        pkill -9 -f "web_server_fixed.py"
        sleep 1
    fi
    
    # 清理PID文件
    [ -f "$PID_FILE" ] && rm "$PID_FILE"
    
    if ! is_server_running; then
        print_status "服务器已成功停止"
    else
        print_error "无法停止服务器"
        return 1
    fi
}

# 重启服务器
restart_server() {
    echo "🔄 重启模型监控系统"
    echo "===================="
    
    stop_server
    sleep 2
    start_server
}

# 显示状态
show_status() {
    echo "📊 模型监控系统状态"
    echo "===================="
    
    # 检查进程
    if is_server_running; then
        pid=$(get_server_pid)
        print_status "服务器正在运行 (PID: $pid)"
        
        # 检查端口
        if ss -tlnp | grep -q ":8085"; then
            print_status "端口8085正在监听"
        else
            print_warning "端口8085没有监听"
        fi
        
        # 测试连接
        if curl -s --max-time 3 http://localhost:8085/ > /dev/null; then
            print_status "HTTP服务响应正常"
        else
            print_warning "HTTP服务无响应"
        fi
        
        # 显示访问地址
        echo ""
        echo "🌐 访问地址:"
        echo "   本机: http://localhost:8085"
        echo "   局域网: http://10.15.89.226:8085"
        
    else
        print_error "服务器没有运行"
    fi
    
    # 显示端口使用情况
    echo ""
    echo "🔌 端口使用情况:"
    if ss -tlnp | grep -E ":808[0-9]" > /dev/null; then
        ss -tlnp | grep -E ":808[0-9]"
    else
        echo "   没有8080-8089端口在使用"
    fi
}

# 显示日志
show_logs() {
    echo "📝 查看服务器日志"
    echo "=================="
    
    if [ -f "$LOG_FILE" ]; then
        echo "日志文件: $LOG_FILE"
        echo "最近20行:"
        echo "----------------------------------------"
        tail -20 "$LOG_FILE"
        echo "----------------------------------------"
        echo ""
        echo "💡 实时查看日志: tail -f $LOG_FILE"
    else
        print_warning "日志文件不存在: $LOG_FILE"
    fi
}

# 运行测试
run_test() {
    echo "🧪 运行系统测试"
    echo "================"
    
    if ! is_server_running; then
        print_error "服务器没有运行，请先启动服务器"
        return 1
    fi
    
    cd "$SCRIPT_DIR" || {
        print_error "无法切换到目录: $SCRIPT_DIR"
        return 1
    }
    
    if [ -f "check_status.py" ]; then
        python3 check_status.py
    elif [ -f "test_system.py" ]; then
        python3 test_system.py
    else
        print_error "找不到测试脚本"
        return 1
    fi
}

# 显示帮助
show_help() {
    echo "模型监控系统管理脚本"
    echo "===================="
    echo ""
    echo "用法: $0 [命令]"
    echo ""
    echo "可用命令:"
    echo "  start    - 启动服务器"
    echo "  stop     - 停止服务器"
    echo "  restart  - 重启服务器"
    echo "  status   - 显示状态"
    echo "  logs     - 查看日志"
    echo "  test     - 运行测试"
    echo "  help     - 显示帮助"
    echo ""
    echo "示例:"
    echo "  $0 start"
    echo "  $0 status"
    echo "  $0 logs"
}

# 主程序
main() {
    case "$1" in
        start)
            start_server
            ;;
        stop)
            stop_server
            ;;
        restart)
            restart_server
            ;;
        status)
            show_status
            ;;
        logs)
            show_logs
            ;;
        test)
            run_test
            ;;
        help|--help|-h)
            show_help
            ;;
        "")
            show_help
            ;;
        *)
            print_error "未知命令: $1"
            echo ""
            show_help
            exit 1
            ;;
    esac
}

# 运行主程序
main "$@"
