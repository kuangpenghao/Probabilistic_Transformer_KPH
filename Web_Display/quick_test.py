#!/usr/bin/env python3
"""
快速测试脚本，验证Web服务器是否正常工作
"""

import urllib.request
import json
import socket
import time
import sys

def test_server(port):
    """测试指定端口的服务器"""
    base_url = f"http://localhost:{port}"
    socket.setdefaulttimeout(5)
    
    print(f"测试服务器: {base_url}")
    
    try:
        # 测试HTML页面
        print("1. 测试HTML页面...")
        response = urllib.request.urlopen(f"{base_url}/")
        if response.getcode() == 200:
            print("   ✓ HTML页面正常")
        else:
            print(f"   ✗ HTML页面错误: {response.getcode()}")
            return False
    except Exception as e:
        print(f"   ✗ HTML页面访问失败: {e}")
        return False
    
    try:
        # 测试JSON API
        print("2. 测试JSON API...")
        response = urllib.request.urlopen(f"{base_url}/models_summary.json")
        if response.getcode() == 200:
            data = json.loads(response.read().decode())
            print("   ✓ JSON API正常")
            print(f"   ✓ 总模型数: {data.get('total_models', 0)}")
            completed = sum(1 for m in data.get('models', []) if m.get('model_state') == 'Completed')
            training = data.get('total_models', 0) - completed
            print(f"   ✓ 已完成: {completed}, 训练中: {training}")
        else:
            print(f"   ✗ JSON API错误: {response.getcode()}")
            return False
    except Exception as e:
        print(f"   ✗ JSON API访问失败: {e}")
        return False
    
    return True

def find_running_server():
    """查找正在运行的服务器"""
    for port in [8080, 8081, 8082, 8083, 8084]:
        try:
            socket.setdefaulttimeout(1)
            response = urllib.request.urlopen(f"http://localhost:{port}/models_summary.json")
            if response.getcode() == 200:
                return port
        except:
            continue
    return None

def main():
    print("🔍 正在查找运行中的Web服务器...")
    
    port = find_running_server()
    if port:
        print(f"✓ 发现服务器运行在端口 {port}")
        if test_server(port):
            print(f"\n🎉 服务器测试通过!")
            print(f"📱 访问地址:")
            print(f"   本机: http://localhost:{port}")
            print(f"   局域网: http://10.15.89.226:{port}")
        else:
            print(f"\n❌ 服务器测试失败")
            sys.exit(1)
    else:
        print("❌ 没有发现运行中的Web服务器")
        print("请先启动服务器:")
        print("  ./start_web_server.sh")
        sys.exit(1)

if __name__ == "__main__":
    main()
