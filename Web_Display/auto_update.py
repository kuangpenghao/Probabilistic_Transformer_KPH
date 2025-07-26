#!/usr/bin/env python3
"""
自动更新模型训练状态的脚本
每隔10秒钟更新一次models_summary.json文件
"""

import time
import signal
import sys
from JsonConclusion import save_json_summary


def signal_handler(sig, frame):
    """优雅地处理Ctrl+C中断"""
    print('\n正在停止自动更新...')
    sys.exit(0)


def main():
    """主函数，定期更新模型状态"""
    print("开始自动更新模型训练状态...")
    print("每10秒更新一次，按Ctrl+C停止")
    print("-" * 50)
    
    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    
    update_count = 0
    
    while True:
        try:
            update_count += 1
            print(f"\n[更新 #{update_count}] {time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # 生成并保存JSON文件
            result = save_json_summary()
            
            if result:
                completed_count = sum(1 for model in result['models'] if model["model_state"] == "Completed")
                training_count = result['total_models'] - completed_count
                print(f"✓ 已完成: {completed_count} 个模型, 训练中: {training_count} 个模型")
            else:
                print("✗ 更新失败")
            
            print("等待下次更新...")
            time.sleep(10)  # 等待10秒
            
        except Exception as e:
            print(f"更新过程中出错: {e}")
            print("5秒后重试...")
            time.sleep(5)


if __name__ == "__main__":
    main()
