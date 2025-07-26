#!/usr/bin/env python3
"""
HTTP服务器用于在局域网中提供模型监控网页服务
支持自动更新JSON数据和网页访问
"""

import http.server
import socketserver
import threading
import time
import os
import signal
import sys
from JsonConclusion import save_json_summary


class ModelMonitorHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """自定义HTTP请求处理器"""
    
    def __init__(self, *args, **kwargs):
        # 设置服务器根目录为当前目录
        super().__init__(*args, directory="/home/kuangph/hf-starter/Web_Display", **kwargs)
    
    def end_headers(self):
        # 添加CORS头部，允许跨域访问
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'X-Requested-With, Content-type')
        super().end_headers()
    
    def do_GET(self):
        # 如果请求的是根路径，重定向到HTMLProcessing.html
        if self.path == '/' or self.path == '':
            self.path = '/HTMLProcessing.html'
        
        # 对于JSON文件请求，添加无缓存头部
        if self.path.endswith('.json') or 'models_summary.json' in self.path:
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Expires', '0')
            self.end_headers()
            
            # 读取并返回JSON文件内容
            json_file_path = "/home/kuangph/hf-starter/Web_Display/models_summary.json"
            try:
                with open(json_file_path, 'rb') as f:
                    self.wfile.write(f.read())
            except FileNotFoundError:
                self.wfile.write(b'{"error": "JSON file not found"}')
            return
        
        # 其他请求使用默认处理
        super().do_GET()
    
    def log_message(self, format, *args):
        # 自定义日志格式
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {format % args}")


class ModelMonitorServer:
    """模型监控服务器"""
    
    def __init__(self, host='0.0.0.0', port=8080):
        self.host = host
        self.port = port
        self.httpd = None
        self.update_thread = None
        self.running = False
        
        # 注册信号处理器
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def signal_handler(self, sig, frame):
        """处理中断信号"""
        print('\n正在停止服务器...')
        self.stop()
        sys.exit(0)
    
    def update_data_continuously(self):
        """持续更新JSON数据"""
        print("数据更新线程已启动，每10秒更新一次...")
        
        while self.running:
            try:
                result = save_json_summary()
                if result:
                    completed_count = sum(1 for model in result['models'] if model["model_state"] == "Completed")
                    training_count = result['total_models'] - completed_count
                    print(f"[{time.strftime('%H:%M:%S')}] 数据已更新 - 完成: {completed_count}, 训练中: {training_count}")
                else:
                    print(f"[{time.strftime('%H:%M:%S')}] 数据更新失败")
                    
            except Exception as e:
                print(f"[{time.strftime('%H:%M:%S')}] 更新数据时出错: {e}")
            
            # 等待10秒
            for _ in range(100):  # 分成100次检查，每次0.1秒，便于快速响应停止信号
                if not self.running:
                    break
                time.sleep(0.1)
    
    def start(self):
        """启动服务器"""
        try:
            # 确保端口可用
            self.httpd = socketserver.TCPServer((self.host, self.port), ModelMonitorHTTPRequestHandler)
            self.httpd.allow_reuse_address = True
            
            print("=" * 60)
            print("模型训练状态监控服务器")
            print("=" * 60)
            print(f"服务器地址: http://{self.host}:{self.port}")
            print(f"本机访问: http://localhost:{self.port}")
            print(f"局域网访问: http://10.15.89.226:{self.port}")
            print("=" * 60)
            print("按 Ctrl+C 停止服务器")
            print()
            
            # 首次生成数据
            print("正在生成初始数据...")
            save_json_summary()
            
            # 启动后台数据更新线程
            self.running = True
            self.update_thread = threading.Thread(target=self.update_data_continuously, daemon=True)
            self.update_thread.start()
            
            # 启动HTTP服务器
            print("HTTP服务器已启动，等待连接...")
            self.httpd.serve_forever()
            
        except OSError as e:
            if e.errno == 98:  # Address already in use
                print(f"错误: 端口 {self.port} 已被占用")
                print("请尝试使用其他端口或停止占用该端口的程序")
            else:
                print(f"启动服务器时出错: {e}")
        except Exception as e:
            print(f"服务器运行出错: {e}")
    
    def stop(self):
        """停止服务器"""
        self.running = False
        
        if self.update_thread and self.update_thread.is_alive():
            print("正在停止数据更新线程...")
            self.update_thread.join(timeout=2)
        
        if self.httpd:
            print("正在停止HTTP服务器...")
            self.httpd.shutdown()
            self.httpd.server_close()
        
        print("服务器已停止")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='模型训练状态监控服务器')
    parser.add_argument('--host', default='0.0.0.0', help='服务器主机地址 (默认: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=8080, help='服务器端口 (默认: 8080)')
    
    args = parser.parse_args()
    
    # 检查是否在正确的目录
    if not os.path.exists('/home/kuangph/hf-starter/Web_Display/HTMLProcessing.html'):
        print("错误: 请确保在正确的目录中运行此脚本")
        print("当前目录应包含 HTMLProcessing.html 文件")
        return
    
    # 创建并启动服务器
    server = ModelMonitorServer(host=args.host, port=args.port)
    server.start()


if __name__ == "__main__":
    main()
