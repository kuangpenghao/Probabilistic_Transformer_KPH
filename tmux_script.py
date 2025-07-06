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

    sessions = result.stdout.decode().strip()
    print(sessions)

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

if __name__ == "__main__":
    sessions_to_close = [
        "m0_1_100M",
        "m0_2_100M",
        "m0_3_100M",
        "m3_1_100M",
        "m3_2_100M",
        "m3_3_100M"
    ]

    for session in sessions_to_close:
        close_tmux_session(session)