#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试所有Method测试文件的导入路径是否正确
"""

import os
import sys
import importlib.util

def test_import_for_file(file_path):
    """测试单个文件的导入是否正确"""
    print(f"\n测试文件: {file_path}")
    
    try:
        # 读取文件内容
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 查找导入语句
        lines = content.split('\n')
        imports = []
        for line in lines:
            line = line.strip()
            if line.startswith('from models.') and 'import' in line:
                imports.append(line)
        
        print(f"  发现导入语句: {len(imports)}")
        for imp in imports:
            print(f"    {imp}")
        
        # 尝试执行文件来检查导入是否成功
        spec = importlib.util.spec_from_file_location("test_module", file_path)
        if spec and spec.loader:
            try:
                module = importlib.util.module_from_spec(spec)
                # 只测试导入，不执行主函数
                sys.modules["test_module"] = module
                spec.loader.exec_module(module)
                print("  ✅ 导入成功")
                return True
            except Exception as e:
                print(f"  ❌ 导入失败: {str(e)}")
                return False
        else:
            print("  ❌ 无法加载模块")
            return False
            
    except Exception as e:
        print(f"  ❌ 读取文件失败: {str(e)}")
        return False

def main():
    """测试所有测试文件的导入路径"""
    print("测试所有Method测试文件的导入路径...")
    
    # 测试文件列表
    test_files = [
        "/home/kuangph/hf-starter/tests_for_methods/test_for_Method1/test_new_residual.py",
        "/home/kuangph/hf-starter/tests_for_methods/test_for_Method1/test_residual_logic.py",
        "/home/kuangph/hf-starter/tests_for_methods/test_for_Method2/test_method2.py",
        "/home/kuangph/hf-starter/tests_for_methods/test_for_Method3/test_method3.py",
        "/home/kuangph/hf-starter/tests_for_methods/test_for_Method5/test_method5.py",
        "/home/kuangph/hf-starter/tests_for_methods/test_for_Method6/test_method6.py",
        "/home/kuangph/hf-starter/tests_for_methods/test_for_Method7/test_method7.py",
    ]
    
    success_count = 0
    total_count = len(test_files)
    
    for test_file in test_files:
        if os.path.exists(test_file):
            if test_import_for_file(test_file):
                success_count += 1
        else:
            print(f"\n❌ 文件不存在: {test_file}")
    
    print(f"\n{'='*60}")
    print(f"导入路径测试结果: {success_count}/{total_count} 成功")
    
    if success_count == total_count:
        print("✅ 所有测试文件的导入路径都正确!")
    else:
        print("❌ 部分测试文件的导入路径有问题，需要进一步修复")
    print("="*60)

if __name__ == "__main__":
    main()
