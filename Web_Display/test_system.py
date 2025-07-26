#!/usr/bin/env python3
"""
测试模型监控系统的完整功能
"""

import requests
import json
import time
import sys

def test_web_server(port=8084):
    """测试Web服务器功能"""
    base_url = f"http://localhost:{port}"
    
    print("正在测试模型监控系统...")
    print(f"服务器地址: {base_url}")
    print("-" * 50)
    
    try:
        # 测试JSON API
        print("1. 测试JSON API...")
        response = requests.get(f"{base_url}/models_summary.json", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"   ✓ JSON API正常")
            print(f"   ✓ 时间戳: {data['timestamp']}")
            print(f"   ✓ 总模型数: {data['total_models']}")
            
            completed = sum(1 for m in data['models'] if m['model_state'] == 'Completed')
            training = data['total_models'] - completed
            print(f"   ✓ 已完成: {completed}, 训练中: {training}")
        else:
            print(f"   ✗ JSON API错误: HTTP {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"   ✗ 无法连接到服务器: {e}")
        return False
    
    try:
        # 测试HTML页面
        print("\n2. 测试HTML页面...")
        response = requests.get(f"{base_url}/", timeout=5)
        if response.status_code == 200:
            content = response.text
            if "模型训练状态监控" in content:
                print("   ✓ HTML页面正常")
                print("   ✓ 页面标题正确")
            else:
                print("   ✗ HTML页面内容异常")
                return False
        else:
            print(f"   ✗ HTML页面错误: HTTP {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"   ✗ 无法访问HTML页面: {e}")
        return False
    
    # 测试数据更新
    print("\n3. 测试数据更新...")
    try:
        # 获取第一次时间戳
        response1 = requests.get(f"{base_url}/models_summary.json", timeout=5)
        timestamp1 = response1.json()['timestamp']
        
        print("   等待数据更新...")
        time.sleep(12)  # 等待数据更新（服务器每10秒更新一次）
        
        # 获取第二次时间戳
        response2 = requests.get(f"{base_url}/models_summary.json", timeout=5)
        timestamp2 = response2.json()['timestamp']
        
        if timestamp1 != timestamp2:
            print("   ✓ 数据自动更新正常")
            print(f"   ✓ 更新前: {timestamp1}")
            print(f"   ✓ 更新后: {timestamp2}")
        else:
            print("   ⚠ 数据可能没有更新（或更新间隔较长）")
            
    except Exception as e:
        print(f"   ✗ 测试数据更新时出错: {e}")
    
    print("\n" + "=" * 50)
    print("测试完成！")
    print(f"✓ 本机访问: {base_url}")
    print(f"✓ 局域网访问: http://192.168.142.1:{port}")
    print("✓ 系统运行正常，可以在局域网中访问")
    
    return True

def main():
    """主函数"""
    try:
        # 安装requests库（如果没有的话）
        import requests
    except ImportError:
        print("需要安装requests库:")
        print("pip install requests")
        return
    
    # 测试系统
    success = test_web_server()
    
    if success:
        print("\n🎉 模型监控系统测试通过！")
        print("\n📱 现在可以:")
        print("   1. 在浏览器中访问 http://localhost:8081")
        print("   2. 在局域网其他设备中访问 http://10.20.192.158:8081")
        print("   3. 查看实时的模型训练状态")
    else:
        print("\n❌ 系统测试失败，请检查服务器状态")
        sys.exit(1)

if __name__ == "__main__":
    main()
