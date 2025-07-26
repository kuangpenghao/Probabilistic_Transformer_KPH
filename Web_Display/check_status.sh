#!/bin/bash

# 检查模型监控系统状态

echo "🔍 模型训练监控系统状态检查"
echo "=" * 50

# 检查Web服务器进程
echo "1. 检查Web服务器进程..."
if pgrep -f "web_server.py" > /dev/null; then
    PID=$(pgrep -f "web_server.py")
    echo "   ✓ Web服务器正在运行 (PID: $PID)"
else
    echo "   ✗ Web服务器未运行"
fi

# 检查端口占用
echo ""
echo "2. 检查端口占用情况..."
for port in 8080 8081 8082 8083 8084; do
    if ss -tln | grep -q ":$port "; then
        echo "   端口 $port: 已占用"
    else
        echo "   端口 $port: 可用"
    fi
done

# 检查文件存在性
echo ""
echo "3. 检查关键文件..."
FILES=("JsonConclusion.py" "web_server.py" "HTMLProcessing.html" "models_summary.json")
for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "   ✓ $file 存在"
    else
        echo "   ✗ $file 缺失"
    fi
done

# 检查JSON数据
echo ""
echo "4. 检查JSON数据..."
if [ -f "models_summary.json" ]; then
    TIMESTAMP=$(python3 -c "import json; data=json.load(open('models_summary.json')); print(data['timestamp'])" 2>/dev/null)
    TOTAL=$(python3 -c "import json; data=json.load(open('models_summary.json')); print(data['total_models'])" 2>/dev/null)
    if [ ! -z "$TIMESTAMP" ]; then
        echo "   ✓ JSON数据有效"
        echo "   ✓ 最后更新: $TIMESTAMP"
        echo "   ✓ 总模型数: $TOTAL"
    else
        echo "   ✗ JSON数据格式错误"
    fi
else
    echo "   ✗ JSON文件不存在"
fi

# 运行快速测试
echo ""
echo "5. 运行连接测试..."
python3 quick_test.py

echo ""
echo "状态检查完成!"
