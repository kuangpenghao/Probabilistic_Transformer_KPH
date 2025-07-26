#!/usr/bin/env python3
"""
简化版HTTP服务器，解决连接超时问题
"""

import http.server
import socketserver
import threading
import time
import os
import signal
import sys
import json
from JsonConclusion import save_json_summary


class FixedHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """修复的HTTP请求处理器"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory="/home/kuangph/hf-starter/Web_Display", **kwargs)
    
    def end_headers(self):
        # 添加CORS头部
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'X-Requested-With, Content-type')
        super().end_headers()
    
    def do_GET(self):
        try:
            # 处理根路径请求
            if self.path == '/' or self.path == '':
                self.path = '/HTMLProcessing.html'
            
            # 处理JSON API请求
            if self.path.endswith('models_summary.json') or 'models_summary.json' in self.path:
                json_file_path = "/home/kuangph/hf-starter/Web_Display/models_summary.json"
                
                try:
                    with open(json_file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json; charset=utf-8')
                    self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
                    self.send_header('Pragma', 'no-cache')
                    self.send_header('Expires', '0')
                    self.send_header('Content-Length', str(len(content.encode('utf-8'))))
                    self.end_headers()
                    
                    self.wfile.write(content.encode('utf-8'))
                    return
                    
                except Exception as e:
                    error_msg = json.dumps({"error": f"Failed to read JSON: {str(e)}"})
                    self.send_response(500)
                    self.send_header('Content-type', 'application/json')
                    self.send_header('Content-Length', str(len(error_msg.encode('utf-8'))))
                    self.end_headers()
                    self.wfile.write(error_msg.encode('utf-8'))
                    return
            
            # 其他请求使用默认处理，但确保正确编码
            if self.path.endswith('.html'):
                # 处理HTML文件编码
                file_path = self.translate_path(self.path)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    self.send_response(200)
                    self.send_header('Content-type', 'text/html; charset=utf-8')
                    self.send_header('Content-Length', str(len(content.encode('utf-8'))))
                    self.end_headers()
                    self.wfile.write(content.encode('utf-8'))
                    return
                except Exception as e:
                    print(f"读取HTML文件错误: {e}")
            
            # 其他请求使用默认处理
            super().do_GET()
            
        except Exception as e:
            print(f"处理请求时出错: {e}")
            try:
                self.send_error(500, f"Internal server error: {str(e)}")
            except:
                pass
    
    def log_message(self, format, *args):
        # 简化日志格式
        print(f"[{time.strftime('%H:%M:%S')}] {format % args}")


class SimpleModelServer:
    """简化的模型监控服务器"""
    
    def __init__(self, host='0.0.0.0', port=8085):
        self.host = host
        self.port = port
        self.httpd = None
        self.update_thread = None
        self.running = False
    
    def update_data_loop(self):
        """后台数据更新循环"""
        print("数据更新线程启动...")
        
        while self.running:
            try:
                result = save_json_summary()
                if result:
                    completed = sum(1 for model in result['models'] if model["model_state"] == "Completed")
                    training = result['total_models'] - completed
                    print(f"[{time.strftime('%H:%M:%S')}] 数据更新完成 - 完成:{completed} 训练:{training}")
                
                # 分段睡眠以便快速响应停止信号
                for _ in range(100):
                    if not self.running:
                        break
                    time.sleep(0.1)
                    
            except Exception as e:
                print(f"数据更新错误: {e}")
                time.sleep(1)
    
    def start(self):
        """启动服务器"""
        try:
            print("正在启动服务器...")
            
            # 创建服务器，设置重用地址
            self.httpd = socketserver.TCPServer((self.host, self.port), FixedHTTPRequestHandler)
            self.httpd.allow_reuse_address = True
            self.httpd.timeout = 10  # 设置超时
            
            print("=" * 50)
            print("模型监控服务器 v2.0")
            print("=" * 50)
            print(f"本机访问: http://localhost:{self.port}")
            print(f"局域网访问: http://10.15.89.226:{self.port}")
            print("按 Ctrl+C 停止")
            print("=" * 50)
            
            # 生成初始数据
            save_json_summary()
            print("初始数据已生成")
            
            # 启动数据更新线程
            self.running = True
            self.update_thread = threading.Thread(target=self.update_data_loop, daemon=True)
            self.update_thread.start()
            
            # 启动HTTP服务器
            print("HTTP服务器已启动")
            self.httpd.serve_forever()
            
        except KeyboardInterrupt:
            print("\n收到停止信号")
        except Exception as e:
            print(f"服务器错误: {e}")
        finally:
            self.stop()
    
    def stop(self):
        """停止服务器"""
        print("正在停止服务器...")
        self.running = False
        
        if self.httpd:
            self.httpd.shutdown()
            self.httpd.server_close()
        
        print("服务器已停止")


def main():
    # 检查文件
    if not os.path.exists('/home/kuangph/hf-starter/Web_Display/HTMLProcessing.html'):
        print("错误: HTMLProcessing.html 文件不存在")
        return
    
    # 创建并启动服务器
    server = SimpleModelServer()
    
    try:
        server.start()
    except KeyboardInterrupt:
        print("\n用户中断")
    except Exception as e:
        print(f"启动失败: {e}")


if __name__ == "__main__":
    main()
