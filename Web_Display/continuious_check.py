import time
import subprocess

def main():

    print("正在检查模型监控服务器状态...")

    while True:
        result=subprocess.run(
            [ "./manage_server.sh","status"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
# "cd", "/home/kuangph/hf-starter/Web_Display", "&&",
        if result.returncode == 0:
            print("模型监控服务器正在运行")
            output = result.stdout.decode().strip()
            print(f"输出: {output}")
        else:
            print("模型监控服务器未运行或发生错误")
            error_msg = result.stderr.decode().strip()
            print(f"错误信息: {error_msg}")

        print("60秒后再次检查...")
        time.sleep(60)  # 每60秒检查一次

if __name__ == "__main__":
    main()