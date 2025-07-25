#!/usr/bin/env python3
"""
实际测试run_clm.sh脚本内容更新功能
"""

import sys
import os
import shutil
import time

# 添加当前目录到Python路径
sys.path.append('.')
import train_script

def read_script_key_lines(script_path="run_clm.sh"):
    """读取脚本中的关键行"""
    config_line = None
    output_line = None
    
    with open(script_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if line.strip().startswith('--config_name'):
                config_line = (i+1, line.strip())
            elif line.strip().startswith('--output_dir'):
                output_line = (i+1, line.strip())
    
    return config_line, output_line

def test_script_modification():
    """测试脚本修改功能"""
    print("🧪 开始实际测试run_clm.sh脚本内容更新功能")
    print("="*60)
    
    # 创建原始文件的安全备份
    original_backup = "run_clm.sh.original_backup"
    shutil.copy2("run_clm.sh", original_backup)
    print(f"📋 已创建原始文件备份: {original_backup}")
    
    # 读取初始状态
    print(f"\n📖 读取初始脚本状态:")
    initial_config, initial_output = read_script_key_lines()
    print(f"   配置行 (第{initial_config[0]}行): {initial_config[1]}")
    print(f"   输出行 (第{initial_output[0]}行): {initial_output[1]}")
    
    # 测试用例列表
    test_cases = [
        ("Version3_Method1", "v3m1"),
        ("Version3_Method2", "v3m2"),
        ("Version3_Method3_1", "v3m3_1"),
        ("Version3_Method3_2", "v3m3_2"),
        ("Version3_Method4_2", "v3m4_2"),
    ]
    
    all_tests_passed = True
    
    for i, (test_config, test_output) in enumerate(test_cases, 1):
        print(f"\n{'='*40}")
        print(f"🧪 测试用例 {i}/{len(test_cases)}")
        print(f"📝 目标配置: {test_config}")
        print(f"📁 目标输出: {test_output}")
        print(f"{'='*40}")
        
        # 执行修改
        print(f"🔧 开始修改脚本...")
        success = train_script.modify_training_script(test_config, test_output)
        
        if not success:
            print(f"❌ 修改失败!")
            all_tests_passed = False
            continue
        
        # 读取修改后的状态
        print(f"📖 读取修改后的脚本状态:")
        modified_config, modified_output = read_script_key_lines()
        print(f"   配置行 (第{modified_config[0]}行): {modified_config[1]}")
        print(f"   输出行 (第{modified_output[0]}行): {modified_output[1]}")
        
        # 验证修改结果
        expected_config_line = f"--config_name configs/{test_config}.json \\"
        expected_output_line = f"--output_dir outputs/{test_output} \\"
        
        config_correct = modified_config[1] == expected_config_line
        output_correct = modified_output[1] == expected_output_line
        
        print(f"🔍 验证结果:")
        print(f"   配置行正确: {'✅' if config_correct else '❌'}")
        print(f"   输出行正确: {'✅' if output_correct else '❌'}")
        
        if config_correct and output_correct:
            print(f"✅ 测试用例 {i} 通过!")
        else:
            print(f"❌ 测试用例 {i} 失败!")
            print(f"   期望配置行: {expected_config_line}")
            print(f"   实际配置行: {modified_config[1]}")
            print(f"   期望输出行: {expected_output_line}")
            print(f"   实际输出行: {modified_output[1]}")
            all_tests_passed = False
        
        # 暂停一下，让用户看清结果
        time.sleep(1)
    
    # 恢复原始文件
    print(f"\n🔄 测试完成，恢复原始文件...")
    shutil.copy2(original_backup, "run_clm.sh")
    
    # 验证恢复是否成功
    restored_config, restored_output = read_script_key_lines()
    restore_success = (restored_config == initial_config and restored_output == initial_output)
    
    if restore_success:
        print(f"✅ 原始文件已成功恢复")
        print(f"   配置行: {restored_config[1]}")
        print(f"   输出行: {restored_output[1]}")
    else:
        print(f"❌ 原始文件恢复失败!")
    
    # 清理备份文件
    backup_files = [original_backup, "run_clm.sh.backup"]
    for backup_file in backup_files:
        if os.path.exists(backup_file):
            os.remove(backup_file)
            print(f"🧹 已清理备份文件: {backup_file}")
    
    # 输出最终结果
    print(f"\n{'='*60}")
    print(f"📊 测试结果总结:")
    print(f"   总测试用例: {len(test_cases)}")
    print(f"   文件恢复: {'✅ 成功' if restore_success else '❌ 失败'}")
    print(f"   整体结果: {'🎉 全部通过' if all_tests_passed and restore_success else '❌ 存在问题'}")
    
    return all_tests_passed and restore_success

def test_edge_cases():
    """测试边界情况"""
    print(f"\n🧪 测试边界情况...")
    
    # 创建安全备份
    edge_backup = "run_clm.sh.edge_test_backup"
    shutil.copy2("run_clm.sh", edge_backup)
    
    try:
        # 测试不存在的配置文件
        print(f"\n测试不存在的脚本文件:")
        fake_script = "fake_script.sh"
        result = train_script.verify_script_modification(fake_script, "test", "test")
        print(f"不存在文件的验证结果: {'❌ 正确失败' if not result else '⚠️ 意外成功'}")
        
        # 测试空配置
        print(f"\n测试空配置:")
        try:
            result = train_script.modify_training_script("", "")
            print(f"空配置修改结果: {'⚠️ 意外成功' if result else '❌ 正确失败'}")
        except Exception as e:
            print(f"空配置修改异常: {e}")
    
    finally:
        # 恢复文件
        shutil.copy2(edge_backup, "run_clm.sh")
        os.remove(edge_backup)
        print(f"🔄 已恢复原始文件状态")

if __name__ == "__main__":
    print("🚀 开始实际测试run_clm.sh脚本内容更新功能")
    
    try:
        # 检查必要文件是否存在
        if not os.path.exists("run_clm.sh"):
            print("❌ 错误: run_clm.sh 文件不存在!")
            sys.exit(1)
        
        # 执行主要测试
        main_test_success = test_script_modification()
        
        # 执行边界测试
        test_edge_cases()
        
        print(f"\n🏁 所有测试完成!")
        if main_test_success:
            print("✅ run_clm.sh脚本内容更新功能工作正常!")
        else:
            print("❌ 发现问题，需要检查代码!")
            
    except Exception as e:
        print(f"\n❌ 测试过程中发生异常: {e}")
        import traceback
        traceback.print_exc()
