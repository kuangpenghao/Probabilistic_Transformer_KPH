#!/usr/bin/env python3
"""
快速验证run_clm.sh修改功能
"""

import sys
import os
import shutil

sys.path.append('.')
import train_script

def quick_test():
    """快速测试一次修改"""
    print("🧪 快速测试run_clm.sh修改功能")
    
    # 读取当前状态
    print("\n📖 当前脚本状态:")
    with open("run_clm.sh", 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if '--config_name' in line or '--output_dir' in line:
                print(f"   第{i+1}行: {line.strip()}")
    
    # 创建备份
    shutil.copy2("run_clm.sh", "run_clm.sh.quick_backup")
    
    # 执行一次修改
    test_config = "Version3_Method2"
    test_output = "v3m2"
    
    print(f"\n🔧 测试修改为: {test_config} -> {test_output}")
    success = train_script.modify_training_script(test_config, test_output)
    
    if success:
        print("\n📖 修改后的脚本状态:")
        with open("run_clm.sh", 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if '--config_name' in line or '--output_dir' in line:
                    print(f"   第{i+1}行: {line.strip()}")
    
    # 恢复备份
    shutil.copy2("run_clm.sh.quick_backup", "run_clm.sh")
    os.remove("run_clm.sh.quick_backup")
    
    print(f"\n✅ 测试完成，文件已恢复")
    return success

if __name__ == "__main__":
    quick_test()
