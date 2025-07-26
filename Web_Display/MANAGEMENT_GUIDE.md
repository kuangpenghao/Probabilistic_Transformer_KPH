# 模型监控系统管理指南

## 📋 概述

这是一套完整的模型训练状态监控系统，包含Web界面、API接口和完整的管理工具。

## 🚀 快速开始

### 方法1：使用统一管理脚本（推荐）

```bash
cd /home/kuangph/hf-starter/Web_Display

# 启动服务器
./manage_server.sh start

# 查看状态
./manage_server.sh status

# 停止服务器
./manage_server.sh stop
```

### 方法2：使用快速启动脚本

```bash
cd /home/kuangph/hf-starter/Web_Display
./quick_start.sh
```

### 方法3：直接启动（不推荐）

```bash
cd /home/kuangph/hf-starter/Web_Display
python3 web_server_fixed.py
```

## 🛠️ 管理命令

### 统一管理脚本 (`manage_server.sh`)

| 命令 | 功能 | 示例 |
|------|------|------|
| `start` | 启动服务器 | `./manage_server.sh start` |
| `stop` | 停止服务器 | `./manage_server.sh stop` |
| `restart` | 重启服务器 | `./manage_server.sh restart` |
| `status` | 查看状态 | `./manage_server.sh status` |
| `logs` | 查看日志 | `./manage_server.sh logs` |
| `test` | 运行测试 | `./manage_server.sh test` |
| `help` | 显示帮助 | `./manage_server.sh help` |

### 独立脚本

- `quick_start.sh` - 交互式快速启动
- `stop_web_server.sh` - 停止服务器（旧版本兼容）
- `check_status.py` - 详细状态检查
- `test_system.py` - 系统功能测试

## 🌐 访问地址

启动成功后，可以通过以下地址访问：

- **本机访问**: http://localhost:8085
- **局域网访问**: http://10.15.89.226:8085
- **JSON API**: http://localhost:8085/models_summary.json

## 📁 文件结构

```
Web_Display/
├── manage_server.sh        # 统一管理脚本（推荐使用）
├── quick_start.sh          # 快速启动脚本
├── stop_web_server.sh      # 停止服务器脚本
├── web_server_fixed.py     # 改进版服务器（主程序）
├── web_server.py          # 原版服务器（备用）
├── HTMLProcessing.html     # Web界面
├── JsonConclusion.py       # 数据处理模块
├── check_status.py         # 状态检查脚本
├── test_system.py          # 系统测试脚本
├── web_server.log          # 服务器日志文件
└── models_summary.json     # 数据文件（自动生成）
```

## 🔧 故障排除

### 1. 端口被占用错误

```bash
# 查看端口使用情况
ss -tlnp | grep :8085

# 强制释放端口
fuser -k 8085/tcp

# 重新启动
./manage_server.sh start
```

### 2. 连接超时问题

```bash
# 检查服务器状态
./manage_server.sh status

# 查看详细日志
./manage_server.sh logs

# 重启服务器
./manage_server.sh restart
```

### 3. 进程管理问题

```bash
# 查找所有相关进程
pgrep -f "web_server"

# 停止所有相关进程
pkill -f "web_server"

# 强制停止（如果需要）
pkill -9 -f "web_server"
```

### 4. 权限问题

```bash
# 给脚本添加执行权限
chmod +x manage_server.sh
chmod +x quick_start.sh
chmod +x stop_web_server.sh
```

## 📊 系统监控

### 实时监控

```bash
# 实时查看日志
tail -f /home/kuangph/hf-starter/Web_Display/web_server.log

# 监控系统状态
watch -n 5 './manage_server.sh status'
```

### 性能检查

```bash
# 运行完整测试
./manage_server.sh test

# 检查资源使用
htop | grep python3
```

## 🚨 注意事项

1. **端口配置**: 当前使用端口8085，避免与其他服务冲突
2. **权限要求**: 确保脚本有执行权限
3. **文件依赖**: 确保所有必要文件存在于Web_Display目录
4. **网络配置**: 局域网访问需要确保防火墙允许8085端口
5. **进程管理**: 使用管理脚本而不是直接kill进程

## 🔄 版本历史

- **v1.0**: 基础HTTP服务器 (`web_server.py`)
- **v2.0**: 改进版服务器 (`web_server_fixed.py`) - 修复编码和超时问题
- **v2.1**: 统一管理系统 (`manage_server.sh`) - 完整的启停管理

## 📞 问题反馈

如果遇到问题，请：

1. 运行 `./manage_server.sh status` 查看状态
2. 运行 `./manage_server.sh logs` 查看日志
3. 运行 `./manage_server.sh test` 进行测试
4. 提供详细的错误信息和日志内容
