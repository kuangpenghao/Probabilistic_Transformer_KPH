#!/usr/bin/env python3
"""
测试就地修改功能
"""

import sys
import os
import shutil

# 添加当前目录到Python路径
sys.path.append('.')
import train_script

def test_in_place_modification():
    """测试就地修改run_clm.sh脚本的功能"""
    print("🧪 测试就地修改run_clm.sh功能...")
    
    # 创建原始脚本的备份用于测试后恢复
    original_backup = "run_clm.sh.original_test_backup"
    if not os.path.exists(original_backup):
        shutil.copy2("run_clm.sh", original_backup)
        print(f"📋 已创建原始备份: {original_backup}")
    
    # 测试配置
    test_config = "Version3_Method3_1"
    test_output = "v3m3_1"
    
    print(f"\n📝 测试修改参数:")
    print(f"   配置文件: {test_config}")
    print(f"   输出目录: {test_output}")
    
    # 读取修改前的内容
    print(f"\n📖 修改前的脚本内容:")
    with open("run_clm.sh", 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if '--config_name' in line or '--output_dir' in line:
                print(f"   第{i+1}行: {line.strip()}")
    
    # 执行修改
    print(f"\n🔧 开始执行就地修改...")
    success = train_script.modify_training_script(test_config, test_output)
    
    if success:
        print(f"\n✅ 修改成功!")
        
        # 读取修改后的内容
        print(f"\n📖 修改后的脚本内容:")
        with open("run_clm.sh", 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if '--config_name' in line or '--output_dir' in line:
                    print(f"   第{i+1}行: {line.strip()}")
    else:
        print(f"\n❌ 修改失败!")
    
    # 恢复原始文件
    print(f"\n🔄 恢复原始文件...")
    shutil.copy2(original_backup, "run_clm.sh")
    print(f"✅ 已恢复原始run_clm.sh")
    
    # 清理临时文件
    backup_files = ["run_clm.sh.backup", original_backup]
    for backup_file in backup_files:
        if os.path.exists(backup_file):
            os.remove(backup_file)
            print(f"🧹 已清理临时文件: {backup_file}")
    
    return success

def test_verification_function():
    """单独测试验证函数"""
    print(f"\n🧪 测试验证函数...")
    
    # 创建测试文件
    test_content = """#!/bin/bash
echo "Start running..."
export HF_ENDPOINT=https://hf-mirror.com

accelerate launch run_clm.py \\
    --config_name configs/Version3_Method3_1.json \\
    --tokenizer_name TinyLlama/TinyLlama-1.1B-intermediate-step-1195k-token-2.5T \\
    --output_dir outputs/v3m3_1 \\
"""
    
    test_file = "test_script.sh"
    with open(test_file, 'w') as f:
        f.write(test_content)
    
    # 测试验证
    result = train_script.verify_script_modification(test_file, "Version3_Method3_1", "v3m3_1")
    
    # 清理测试文件
    os.remove(test_file)
    
    if result:
        print(f"✅ 验证函数测试通过")
    else:
        print(f"❌ 验证函数测试失败")
    
    return result

if __name__ == "__main__":
    print("🚀 开始测试就地修改功能")
    
    try:
        # 测试验证函数
        verify_result = test_verification_function()
        
        # 测试就地修改功能
        modify_result = test_in_place_modification()
        
        print(f"\n📊 测试结果总结:")
        print(f"   验证函数: {'✅ 通过' if verify_result else '❌ 失败'}")
        print(f"   就地修改: {'✅ 通过' if modify_result else '❌ 失败'}")
        
        if verify_result and modify_result:
            print(f"\n🎉 所有测试通过！就地修改功能可以正常使用。")
        else:
            print(f"\n⚠️ 部分测试失败，需要检查代码。")
            
    except Exception as e:
        print(f"\n❌ 测试过程中出错: {e}")
        import traceback
        traceback.print_exc()
