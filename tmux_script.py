import subprocess

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
        print(f"❌ 无法获取会话列表: {error_msg}")
        return []

    # 解析每一行，只保留冒号前的会话名
    lines = result.stdout.decode().strip().split('\n')
    sessions = [line.split(':')[0] for line in lines if line]
    return sessions

def print_tmux_sessions():
    sessions=list_tmux_sessions()
    for session in sessions:
        print(session)

def close_tmux_session(session_name):
    """
    关闭指定名称的 tmux 会话（包括所有窗口）
    :param session_name: 会话名称
    """
    print(f"正在尝试关闭会话 '{session_name}' ...")
    result = subprocess.run(
        ["tmux", "kill-session", "-t", session_name],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    if result.returncode == 0:
        print(f"✅ 成功关闭会话 '{session_name}'")
    else:
        error_msg = result.stderr.decode().strip()
        print(f"❌ 无法关闭会话 '{session_name}': {error_msg}")

def create_tmux_session(session_name):
    """
    创建指定名称的 tmux 会话（包括所有窗口）
    :param session_name: 会话名称
    """
    print(f"正在尝试打开会话 '{session_name}' ...")
    result = subprocess.run(
        ["tmux", "new", "-s", session_name],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    if result.returncode == 0:
        print(f"✅ 成功打开会话 '{session_name}'")
    else:
        error_msg = result.stderr.decode().strip()
        print(f"❌ 无法打开会话 '{session_name}': {error_msg}")

def attach_tmux_session(session_name):
    result = subprocess.run(
        ["tmux", "attach", "-t", session_name],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    if result.returncode == 0:
        print(f"✅ 成功进入会话 '{session_name}'")
    else:
        error_msg = result.stderr.decode().strip()
        print(f"❌ 无法进入会话 '{session_name}': {error_msg}")

if __name__ == "__main__":

    '''
    sessions = list_tmux_sessions()
    
    print(sessions)
    '''

    '''
  
    '''

    ''''''
    sessions=[
        "v3m3-2",
        "v3m4-1",
        "v3m4-2"
    ]
    
    for session in sessions:
        attach_tmux_session(session)
    