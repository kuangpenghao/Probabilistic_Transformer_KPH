import subprocess
import time
import os
import shutil
from typing import List, Tuple, Optional
from datetime import datetime

def log_training_submission(session_name: str, config_name: str, output_dir_name: str, success: bool, log_path: str = "log.txt") -> None:
    """
    记录训练任务提交日志
    :param session_name: tmux会话名
    :param config_name: 配置文件名
    :param output_dir_name: 输出目录名
    :param success: 提交是否成功
    :param log_path: 日志文件路径
    """
    try:
        # 获取当前时间
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 构建日志条目
        status = "成功" if success else "失败"
        log_entry = f"[{current_time}] 重新提交任务 - 会话: {session_name}, 配置: {config_name}, 输出目录: {output_dir_name}, 状态: {status}\n"
        
        # 追加写入日志文件
        script_dir = os.path.dirname(os.path.abspath(__file__))
        full_log_path = os.path.join(script_dir, log_path)
        
        with open(full_log_path, 'a', encoding='utf-8') as f:
            f.write(log_entry)
        
        print(f" 日志已记录到: {full_log_path}")
        
    except Exception as e:
        print(f" 记录日志时出错: {e}")


def list_tmux_sessions():
    """
    列出所有当前的 tmux 会话
    :return: 会话名称列表
    """
    result = subprocess.run(
        ["tmux", "ls"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    if result.returncode != 0:
        error_msg = result.stderr.decode().strip()
        print(f" 无法获取会话列表: {error_msg}")
        return []

    # 解析每一行，只保留冒号前的会话名
    lines = result.stdout.decode().strip().split('\n')
    sessions = [line.split(':')[0] for line in lines if line]
    return sessions


def check_tmux_session_idle(session_name: str) -> bool:
    """
    检查tmux会话是否处于空闲状态
    :param session_name: tmux会话名称
    :return: True表示空闲，False表示正在运行
    """
    try:
        # 检查会话中是否有正在运行的进程
        result = subprocess.run(
            ["tmux", "list-panes", "-t", session_name, "-F", "#{pane_current_command}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        if result.returncode != 0:
            print(f" 无法检查会话 {session_name} 状态")
            return False
            
        commands = result.stdout.strip().split('\n')
        # 如果只有bash/sh等shell命令，说明空闲
        for cmd in commands:
            if cmd not in ['bash', 'sh', 'zsh', 'fish', '']:
                return False
        return True
        
    except Exception as e:
        print(f" 检查会话 {session_name} 时出错: {e}")
        return False


def check_training_completed(output_dir: str) -> bool:
    """
    检查训练是否已完成
    通过检查输出目录中是否存在标志性文件来判断
    :param output_dir: 输出目录路径
    :return: True表示训练完成，False表示未完成
    """
    # 检查是否存在训练完成的标志文件
    completion_file = os.path.join(output_dir, "training_completed.flag")
    if os.path.exists(completion_file):
        return True
    
    # 检查是否存在all_results.json文件（通常表示训练完成）
    results_file = os.path.join(output_dir, "all_results.json")
    if os.path.exists(results_file):
        return True
        
    return False


def modify_training_script(config_name: str, output_dir_name: str) -> bool:
    """
    就地修改run_clm.sh脚本的配置参数
    :param config_name: 配置文件名
    :param output_dir_name: 输出目录名
    :return: True表示修改成功，False表示失败
    """
    script_path = "run_clm.sh"
    backup_path = "run_clm.sh.backup"
    
    try:
        # 1. 创建备份
        shutil.copy2(script_path, backup_path)
        print(f" 已创建备份文件: {backup_path}")
        
        # 2. 读取原始脚本
        with open(script_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 3. 替换配置文件和输出目录
        lines = content.split('\n')
        config_modified = False
        output_modified = False
        
        for i, line in enumerate(lines):
            if line.strip().startswith('--config_name'):
                lines[i] = f"    --config_name configs/{config_name}.json \\"
                config_modified = True
                print(f"  修改config_name: configs/{config_name}.json")
            elif line.strip().startswith('--output_dir'):
                lines[i] = f"    --output_dir outputs/{output_dir_name} \\"
                output_modified = True
                print(f"  修改output_dir: outputs/{output_dir_name}")
        
        # 4. 写入修改后的内容
        modified_content = '\n'.join(lines)
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(modified_content)
        
        # 5. 二次检查验证修改是否正确
        print(f"  开始二次检查验证...")
        verification_success = verify_script_modification(script_path, config_name, output_dir_name)
        
        if verification_success and config_modified and output_modified:
            print(f"  脚本修改成功并通过验证")
            return True
        else:
            # 如果验证失败，恢复备份
            shutil.copy2(backup_path, script_path)
            print(f"   脚本修改验证失败，已恢复备份")
            return False
            
    except Exception as e:
        # 如果出现异常，尝试恢复备份
        try:
            if os.path.exists(backup_path):
                shutil.copy2(backup_path, script_path)
                print(f"  异常发生，已恢复备份文件")
        except:
            pass
        print(f"   修改脚本时出错: {e}")
        return False


def verify_script_modification(script_path: str, expected_config: str, expected_output: str) -> bool:
    """
    验证脚本修改是否正确
    :param script_path: 脚本路径
    :param expected_config: 期望的配置文件名
    :param expected_output: 期望的输出目录名
    :return: True表示验证通过，False表示验证失败
    """
    try:
        with open(script_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        lines = content.split('\n')
        config_found = False
        output_found = False
        
        expected_config_line = f"    --config_name configs/{expected_config}.json \\"
        expected_output_line = f"    --output_dir outputs/{expected_output} \\"
        
        for line in lines:
            if line.strip().startswith('--config_name'):
                if line == expected_config_line:
                    config_found = True
                    print(f"  config_name验证通过: {line.strip()}")
                else:
                    print(f"  config_name验证失败，期望: {expected_config_line.strip()}, 实际: {line.strip()}")
                    return False
            elif line.strip().startswith('--output_dir'):
                if line == expected_output_line:
                    output_found = True
                    print(f"  output_dir验证通过: {line.strip()}")
                else:
                    print(f"  output_dir验证失败，期望: {expected_output_line.strip()}, 实际: {line.strip()}")
                    return False
        
        if config_found and output_found:
            print(f"    所有关键参数验证通过")
            return True
        else:
            print(f"     关键参数缺失 - config_found: {config_found}, output_found: {output_found}")
            return False
            
    except Exception as e:
        print(f"     验证过程出错: {e}")
        return False


def submit_training_job(session_name: str, script_path: str, config_name: str, output_dir_name: str) -> bool:
    """
    在指定的tmux会话中提交训练任务
    :param session_name: tmux会话名
    :param script_path: 训练脚本路径
    :param config_name: 配置文件名
    :param output_dir_name: 输出目录名
    :return: True表示提交成功，False表示失败
    """
    try:
        # 构建srun命令
        srun_command = f"srun -N 1 -n 1 -X -u -p normal --gres=gpu:1 -c 2 --mem=1M -t 0-96:00:00 bash {script_path}"
        
        # 向tmux会话发送命令
        result = subprocess.run(
            ["tmux", "send-keys", "-t", session_name, srun_command, "Enter"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        success = result.returncode == 0
        
        # 记录日志
        log_training_submission(session_name, config_name, output_dir_name, success)
        
        if success:
            print(f" 成功向会话 {session_name} 提交训练任务")
            return True
        else:
            print(f" 向会话 {session_name} 提交任务失败")
            return False
            
    except Exception as e:
        print(f" 提交任务时出错: {e}")
        # 记录失败日志
        log_training_submission(session_name, config_name, output_dir_name, False)
        return False


def process_single_session(session_name: str, config_name: str, output_dir_name: str) -> bool:
    """
    处理单个tmux会话
    :param session_name: tmux会话名
    :param config_name: 配置文件名
    :param output_dir_name: 输出目录名
    :return: True表示处理成功，False表示失败
    """

    time.sleep(2)  # 确保会话列表更新

    print(f"检查会话: {session_name}")
    
    # 检查会话是否空闲
    if not check_tmux_session_idle(session_name):
        print(f" 会话 {session_name} 正在运行中，跳过")
        return True
    
    print(f"会话 {session_name} 处于空闲状态")
    
    # 检查训练是否完成
    output_path = f"outputs/{output_dir_name}"
    if check_training_completed(output_path):
        print(f" 会话 {session_name} 的训练已完成")
        return True
    
    print(f"会话 {session_name} 需要重新启动训练")
    
    # 修改训练脚本
    if modify_training_script(config_name, output_dir_name):
        print(f" 已成功修改训练脚本配置")
        
        # 提交训练任务
        script_path = "run_clm.sh"  # 使用原始脚本文件
        success = submit_training_job(session_name, script_path, config_name, output_dir_name)
        
        if success:
            print(f" 会话 {session_name} 训练任务已重新提交")
        else:
            print(f" 会话 {session_name} 训练任务提交失败")
        
        return success
    else:
        print(f" 修改训练脚本失败，跳过任务提交")
        return False


def run_monitoring_cycle(configs_name: List[str], output_dir_name: List[str], sessions_name: List[str]) -> None:
    """
    执行一轮监控循环
    :param configs_name: 配置文件名列表
    :param output_dir_name: 输出目录名列表
    :param sessions_name: 会话名列表
    """
    print(f"\n{'='*50}")
    print(f"开始新的监控循环 - {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*50}")
    
    # 获取当前所有tmux会话
    current_sessions = list_tmux_sessions()
    if not current_sessions:
        print(" 没有找到任何tmux会话")
        return
    
    print(f"\n当前tmux会话: {current_sessions}")
    
    # 创建配置映射
    session_config_map = {}
    for i, session in enumerate(sessions_name):
        if i < len(configs_name) and i < len(output_dir_name):
            session_config_map[session] = (configs_name[i], output_dir_name[i])
    
    # 处理每个目标会话
    for session_name in sessions_name:
        print(f"\n")
        if session_name not in current_sessions:
            print(f" 目标会话 {session_name} 不存在，跳过")
            continue
        
        if session_name not in session_config_map:
            print(f" 会话 {session_name} 没有对应的配置，跳过")
            continue
        
        config_name, output_dir_name_single = session_config_map[session_name]
        
        try:
            process_single_session(session_name, config_name, output_dir_name_single)
        except Exception as e:
            print(f" 处理会话 {session_name} 时出错: {e}")
    
    print(f" 监控循环完成 - {time.strftime('%Y-%m-%d %H:%M:%S')}")


def main():
    """
    主函数 - 自动化训练任务监控脚本
    """
    # 配置三个列表：配置文件名、输出目录名、会话名
    # 根据测试结果调整为实际的会话名
    configs_name = ["Original_llama_llamatiny","Version2_Method1","Version2_Method3","Version3_Method1","Version3_Method3_1","Version3_Method3_2","Method1","Method2","Method3","Method4","Method5"]
    output_dir_name = ["base","v2m1","v2m3","v3m1","v3m3_1","v3m3_2","test1","test2","test3","test4","test5"]
    sessions_name = ["base","v2m1","v2m3","v3m1","v3m3-1","v3m3-2","test1","test2","test3","test4","test5"]
    
    print("自动化训练任务监控脚本启动")
    print(f"配置数量: {len(configs_name)}")
    print(f"监控的配置: {configs_name}")
    print(f"对应输出目录: {output_dir_name}")
    print(f"对应tmux会话: {sessions_name}")
    
    # 验证列表长度一致性
    if not (len(configs_name) == len(output_dir_name) == len(sessions_name)):
        print(" 错误：三个列表长度不一致！")
        return
    
    try:
        while True:
            run_monitoring_cycle(configs_name, output_dir_name, sessions_name)
            print(f"\n等待60s后进行下一轮检查...")
            time.sleep(60)  # 等待
            
    except KeyboardInterrupt:
        print(f"\n脚本被用户中断")
    except Exception as e:
        print(f"\n脚本运行出错: {e}")


if __name__ == "__main__":
    main()