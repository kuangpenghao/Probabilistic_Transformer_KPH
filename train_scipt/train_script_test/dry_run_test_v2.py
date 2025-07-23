#!/usr/bin/env python3
"""
干运行测试 - 测试监控逻辑但不实际提交任务（使用就地修改方式）
"""

import sys
import os

# 添加当前目录到Python路径
sys.path.append('.')
import train_script

def test_dry_run():
    """执行一次干运行测试"""
    print("🧪 开始干运行测试（使用就地修改方式）...")
    
    # 使用实际的配置
    configs_name = ["Version3_Method3_1", "Version3_Method3_2", "Version3_Method4_1", "Version3_Method4_2"]
    output_dir_name = ["v3m3_1", "v3m3_2", "v3m4_1", "v3m4_2"]
    sessions_name = ["v3m3-1", "v3m3-2", "v3m4-1", "v3m4-2"]
    
    print(f"📋 配置的会话: {sessions_name}")
    
    # 获取当前实际的tmux会话
    current_sessions = train_script.list_tmux_sessions()
    print(f"📋 当前tmux会话: {current_sessions}")
    
    # 创建会话配置映射
    session_config_map = {}
    for i, session in enumerate(sessions_name):
        if i < len(configs_name) and i < len(output_dir_name):
            session_config_map[session] = (configs_name[i], output_dir_name[i])
    
    print(f"🗺️ 会话配置映射: {session_config_map}")
    
    # 逐个检查会话状态
    for session_name in sessions_name:
        print(f"\n{'='*30}")
        print(f"🔍 检查会话: {session_name}")
        
        if session_name not in current_sessions:
            print(f"⚠️ 会话 {session_name} 不存在")
            continue
        
        if session_name not in session_config_map:
            print(f"⚠️ 会话 {session_name} 没有配置映射")
            continue
        
        config_name, output_dir_name_single = session_config_map[session_name]
        print(f"📄 配置文件: {config_name}")
        print(f"📁 输出目录: {output_dir_name_single}")
        
        # 检查会话是否空闲
        is_idle = train_script.check_tmux_session_idle(session_name)
        print(f"💤 会话空闲状态: {is_idle}")
        
        if not is_idle:
            print(f"✅ 会话 {session_name} 正在运行中，无需操作")
            continue
        
        # 检查训练是否完成
        output_path = f"outputs/{output_dir_name_single}"
        is_completed = train_script.check_training_completed(output_path)
        print(f"🏁 训练完成状态: {is_completed}")
        
        if is_completed:
            print(f"✅ 会话 {session_name} 的训练已完成")
        else:
            print(f"🔄 会话 {session_name} 需要重新启动训练")
            # 在干运行模式下，我们只显示会执行的操作，不实际执行
            print(f"📝 将就地修改run_clm.sh脚本:")
            print(f"   - 配置文件: configs/{config_name}.json")
            print(f"   - 输出目录: outputs/{output_dir_name_single}")
            print(f"🚀 将提交命令: srun -N 1 -n 1 -X -u -p normal --gres=gpu:1 -c 2 --mem=1M -t 0-96:00:00 bash run_clm.sh")

if __name__ == "__main__":
    print("🔍 开始干运行测试 - 不会实际提交任何任务")
    test_dry_run()
    print("\n✅ 干运行测试完成")
