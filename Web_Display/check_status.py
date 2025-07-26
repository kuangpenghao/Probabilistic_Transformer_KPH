#!/usr/bin/env python3
"""
检查模型监控系统状态的脚本
"""

import subprocess
import requests
import json
import sys

def check_server_process():
    """检查服务器进程"""
    try:
        result = subprocess.run(['pgrep', '-f', 'web_server'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            pids = result.stdout.strip().split('\n')
            return pids
        return []
    except Exception as e:
        print(f"检查进程时出错: {e}")
        return []

def check_port_listening():
    """检查端口是否在监听"""
    try:
        result = subprocess.run(['ss', '-tlnp'], capture_output=True, text=True)
        return ':8085' in result.stdout
    except Exception as e:
        print(f"检查端口时出错: {e}")
        return False

def test_connectivity():
    """测试连接性"""
    endpoints = [
        'http://localhost:8085/',
        'http://localhost:8085/models_summary.json',
        'http://10.15.89.226:8085/',
        'http://10.15.89.226:8085/models_summary.json'
    ]
    
    results = {}
    for endpoint in endpoints:
        try:
            response = requests.get(endpoint, timeout=5)
            results[endpoint] = {
                'status': response.status_code,
                'success': response.status_code == 200,
                'size': len(response.content)
            }
        except Exception as e:
            results[endpoint] = {
                'status': 'ERROR',
                'success': False,
                'error': str(e)
            }
    
    return results

def main():
    print("🔍 模型监控系统状态检查")
    print("=" * 40)
    
    # 检查进程
    pids = check_server_process()
    if pids:
        print(f"✅ 服务器进程运行中 (PID: {', '.join(pids)})")
    else:
        print("❌ 没有找到服务器进程")
        return
    
    # 检查端口
    if check_port_listening():
        print("✅ 端口8085正在监听")
    else:
        print("❌ 端口8085未在监听")
        return
    
    # 测试连接
    print("\n🌐 连接性测试:")
    results = test_connectivity()
    
    for endpoint, result in results.items():
        if result['success']:
            size_info = f" ({result['size']} bytes)" if 'size' in result else ""
            print(f"✅ {endpoint}{size_info}")
        else:
            error_info = result.get('error', f"HTTP {result['status']}")
            print(f"❌ {endpoint} - {error_info}")
    
    # 检查JSON数据
    try:
        response = requests.get('http://localhost:8085/models_summary.json', timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"\n📊 当前数据状态:")
            print(f"   时间戳: {data['timestamp']}")
            print(f"   总模型数: {data['total_models']}")
            completed = sum(1 for m in data['models'] if m['model_state'] == 'Completed')
            print(f"   已完成: {completed}")
            print(f"   训练中: {data['total_models'] - completed}")
    except Exception as e:
        print(f"\n❌ 无法获取JSON数据: {e}")
    
    print(f"\n🎯 访问地址:")
    print(f"   本机: http://localhost:8085")
    print(f"   局域网: http://10.15.89.226:8085")

if __name__ == "__main__":
    main()
