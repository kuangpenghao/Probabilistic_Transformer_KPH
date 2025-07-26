# 模型训练状态监控系统

这个系统用于实时监控模型训练状态，并通过Web界面在局域网中提供实时显示。

## 系统架构

1. **数据层**: `JsonConclusion.py` - 解析模型训练状态并生成JSON数据
2. **服务层**: `web_server.py` - HTTP服务器，提供Web服务和数据更新
3. **展示层**: `HTMLProcessing.html` - Web前端界面，实时显示训练状态

## 文件说明

### 核心文件
- `JsonConclusion.py` - 数据处理脚本，解析模型训练状态
- `web_server.py` - HTTP服务器，集成数据更新和Web服务
- `HTMLProcessing.html` - Web前端界面
- `models_summary.json` - 生成的JSON数据文件

### 控制脚本
- `start_web_server.sh` - 启动Web监控服务
- `stop_web_server.sh` - 停止Web监控服务

### 历史文件（可选）
- `auto_update.py` - 独立的数据更新脚本（已集成到web_server.py中）
- `start_auto_update.sh` / `stop_auto_update.sh` - 数据更新控制脚本

## 快速开始

### 1. 启动Web监控服务
```bash
cd /home/kuangph/hf-starter/Web_Display
./start_web_server.sh
```

### 2. 访问监控界面
- **本机访问**: http://localhost:8080
- **局域网访问**: http://10.20.192.158:8080

### 3. 停止服务
```bash
./stop_web_server.sh
```

## 功能特性

### Web界面功能
- ✅ 实时显示模型训练状态
- ✅ 自动每5秒刷新数据
- ✅ 响应式设计，支持移动端
- ✅ 进度条显示训练进度
- ✅ 详细的训练指标展示
- ✅ 状态分类（已完成/训练中）
- ✅ 美观的卡片式布局

### 服务器功能
- ✅ 局域网访问支持
- ✅ 自动端口检测（8080/8081）
- ✅ 后台数据更新（每10秒）
- ✅ CORS跨域支持
- ✅ 无缓存JSON数据传输
- ✅ 优雅的服务停止

## 数据监控逻辑

### 训练状态判断
1. **已完成**: 存在 `eval_results.json` 文件
2. **训练中**: 不存在 `eval_results.json`，从最新checkpoint获取状态

### 训练进度计算
- 基于当前epoch数 / 总训练轮次(5) × 100%
- 实时更新训练步数、损失值、学习率等指标

### 数据更新频率
- **服务器端**: 每10秒更新JSON数据
- **Web前端**: 每5秒请求最新数据

## 网络配置

### 局域网访问设置
- 服务器绑定到 `0.0.0.0:8080`，允许外部访问
- 防火墙需要开放8080端口
- 局域网内其他设备通过 `http://10.20.192.158:8080` 访问

### 端口说明
- 默认端口: 8080
- 备用端口: 8081（当8080被占用时自动使用）

## 故障排除

### 常见问题

1. **端口被占用**
   ```bash
   # 检查端口占用
   netstat -tuln | grep :8080
   # 或使用备用端口
   python3 web_server.py --port 8081
   ```

2. **无法访问网页**
   - 检查防火墙设置
   - 确认IP地址正确 (10.20.192.158)
   - 验证服务器是否正在运行

3. **数据不更新**
   ```bash
   # 检查日志
   tail -f web_server.log
   # 手动测试数据生成
   python3 JsonConclusion.py
   ```

4. **Web服务无响应**
   ```bash
   # 重启服务
   ./stop_web_server.sh
   ./start_web_server.sh
   ```

### 日志查看
```bash
# 查看Web服务器日志
tail -f web_server.log

# 查看实时更新
tail -f web_server.log | grep "数据已更新"
```

## 自定义配置

### 修改更新频率
编辑 `web_server.py`:
```python
# 服务器端数据更新间隔（秒）
time.sleep(10)  # 改为其他值
```

编辑 `HTMLProcessing.html`:
```javascript
// 前端数据刷新间隔（毫秒）
this.refreshInterval = 5000;  // 改为其他值
```

### 修改端口
```bash
# 启动时指定端口
python3 web_server.py --port 9000
```

### 修改IP地址
编辑脚本中的 `10.20.192.158` 为你的实际IP地址

## 技术栈

- **后端**: Python 3 + HTTP Server
- **前端**: HTML5 + CSS3 + JavaScript (ES6)
- **数据格式**: JSON
- **部署**: 单机部署，局域网访问

## 安全注意事项

- 此服务仅适用于可信的局域网环境
- 没有身份验证机制
- 生产环境建议添加访问控制

## 扩展建议

- 添加历史数据图表
- 集成训练日志查看
- 添加模型性能对比
- 支持训练控制功能
- 集成Telegram/微信通知
