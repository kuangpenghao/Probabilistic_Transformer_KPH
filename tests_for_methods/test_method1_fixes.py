#!/usr/bin/env python3
"""
综合测试所有Method1相关文件的修复情况
"""
import os
import sys
import subprocess
import traceback

def test_import_and_run(file_path, description):
    """测试单个文件的导入和运行"""
    print(f"\n{'='*60}")
    print(f"测试: {description}")
    print(f"文件: {file_path}")
    print(f"{'='*60}")
    
    if not os.path.exists(file_path):
        print(f"❌ 文件不存在: {file_path}")
        return False
    
    try:
        # 尝试运行文件
        result = subprocess.run(
            [sys.executable, file_path],
            cwd="/home/kuangph/hf-starter",
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode == 0:
            print(f"✅ 成功运行")
            if result.stdout:
                print("输出:")
                print(result.stdout[:500] + "..." if len(result.stdout) > 500 else result.stdout)
        else:
            print(f"❌ 运行失败 (返回码: {result.returncode})")
            if result.stderr:
                print("错误信息:")
                print(result.stderr)
        
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print(f"⏰ 运行超时 (60秒)")
        return False
    except Exception as e:
        print(f"❌ 运行异常: {e}")
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("Method1文件修复验证测试")
    print("=" * 60)
    
    # 要测试的文件列表
    test_files = [
        ("tests_for_methods/test_for_Method1/test_new_residual.py", "Method1基础测试"),
        ("tests_for_methods/test_for_Method1/test_residual_logic.py", "Method1残差逻辑测试"),
        ("tests_for_methods/test_for_Method1/debug_attention.py", "Method1注意力调试"),
        ("tests_for_methods/test_for_Method1/demo_new_residual.py", "Method1演示脚本"),
    ]
    
    results = []
    
    for file_path, description in test_files:
        full_path = os.path.join("/home/kuangph/hf-starter", file_path)
        success = test_import_and_run(full_path, description)
        results.append((description, success))
    
    # 打印总结
    print(f"\n{'='*60}")
    print("测试总结")
    print(f"{'='*60}")
    
    success_count = sum(1 for _, success in results if success)
    total_count = len(results)
    
    for description, success in results:
        status = "✅ 通过" if success else "❌ 失败"
        print(f"{status} {description}")
    
    print(f"\n总计: {success_count}/{total_count} 个测试通过")
    
    if success_count == total_count:
        print("\n🎉 所有Method1测试文件修复成功！")
    else:
        print(f"\n⚠️  还有 {total_count - success_count} 个文件需要修复")
    
    return success_count == total_count

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
