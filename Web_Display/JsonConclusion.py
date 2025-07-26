import os
import json
import re
from typing import Dict, List, Any


def find_latest_checkpoint(model_path: str) -> str:
    """找到最新的checkpoint文件夹"""
    checkpoint_dirs = []
    
    for item in os.listdir(model_path):
        if os.path.isdir(os.path.join(model_path, item)) and item.startswith('checkpoint-'):
            # 提取checkpoint后的数字
            match = re.match(r'checkpoint-(\d+)', item)
            if match:
                checkpoint_num = int(match.group(1))
                checkpoint_dirs.append((checkpoint_num, item))
    
    if not checkpoint_dirs:
        return None
    
    # 选择数字最大的checkpoint
    latest_checkpoint = sorted(checkpoint_dirs, key=lambda x: x[0])[-1][1]
    return os.path.join(model_path, latest_checkpoint)


def get_training_state(checkpoint_path: str) -> Dict[str, Any]:
    """从trainer_state.json中获取最后一个log_history条目"""
    trainer_state_file = os.path.join(checkpoint_path, 'trainer_state.json')
    
    if not os.path.exists(trainer_state_file):
        return {}
    
    try:
        with open(trainer_state_file, 'r', encoding='utf-8') as f:
            trainer_state = json.load(f)
        
        log_history = trainer_state.get('log_history', [])
        if log_history:
            return log_history[-1]  # 返回最后一个条目
        return {}
    
    except (json.JSONDecodeError, FileNotFoundError):
        return {}


def calculate_training_rate(training_state: Dict[str, Any]) -> str:
    """计算训练进度百分比"""
    epoch = training_state.get('epoch', 0)
    if epoch:
        progress = (epoch / 5.0) * 100
        return f"{int(progress)}%"
    return "0%"


def process_model(model_name: str, model_path: str) -> Dict[str, Any]:
    """处理单个模型，返回对应的json数据"""
    model_data = {
        "model_name": model_name
    }
    
    eval_results_file = os.path.join(model_path, 'eval_results.json')
    
    # 检查是否存在eval_results.json文件
    if os.path.exists(eval_results_file):
        # 模型已训练完毕
        model_data["model_state"] = "Completed"
        
        try:
            with open(eval_results_file, 'r', encoding='utf-8') as f:
                eval_result = json.load(f)
            model_data["eval_result"] = eval_result
        except (json.JSONDecodeError, FileNotFoundError):
            model_data["eval_result"] = {}
    
    else:
        # 模型还在训练中
        model_data["model_state"] = "Training"
        
        # 找到最新的checkpoint
        latest_checkpoint = find_latest_checkpoint(model_path)
        
        if latest_checkpoint:
            training_state = get_training_state(latest_checkpoint)
            model_data["training_state"] = training_state
            model_data["training_rate"] = calculate_training_rate(training_state)
        else:
            model_data["training_state"] = {}
            model_data["training_rate"] = "0%"
    
    return model_data


def generate_models_summary() -> List[Dict[str, Any]]:
    """生成所有模型的汇总数据"""
    outputs_dir = "/home/kuangph/hf-starter/outputs"
    models_data = []
    
    if not os.path.exists(outputs_dir):
        print(f"错误：outputs目录不存在: {outputs_dir}")
        return models_data
    
    # 遍历outputs目录中的所有子文件夹
    for item in os.listdir(outputs_dir):
        model_path = os.path.join(outputs_dir, item)
        
        # 只处理文件夹
        if os.path.isdir(model_path):
            model_data = process_model(item, model_path)
            models_data.append(model_data)
    
    return models_data


def save_json_summary(output_file: str = "/home/kuangph/hf-starter/Web_Display/models_summary.json"):
    """生成并保存模型汇总的json文件"""
    models_data = generate_models_summary()
    
    # 创建最终的json对象
    summary_data = {
        "timestamp": None,  # 可以添加时间戳
        "total_models": len(models_data),
        "models": models_data
    }
    
    # 添加当前时间戳
    import datetime
    summary_data["timestamp"] = datetime.datetime.now().isoformat()
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, ensure_ascii=False, indent=2)
        
        print(f"成功生成模型汇总文件: {output_file}")
        print(f"处理了 {len(models_data)} 个模型")
        
        # 打印简要统计信息
        completed_count = sum(1 for model in models_data if model["model_state"] == "Completed")
        training_count = len(models_data) - completed_count
        print(f"已完成训练: {completed_count} 个模型")
        print(f"正在训练: {training_count} 个模型")
        
        return summary_data
        
    except Exception as e:
        print(f"保存文件时出错: {e}")
        return None


if __name__ == "__main__":
    # 测试运行
    result = save_json_summary()
    if result:
        print("\n生成的JSON汇总数据预览:")
        print(f"时间戳: {result['timestamp']}")
        print(f"总模型数: {result['total_models']}")
        for model in result['models'][:3]:  # 显示前3个模型作为预览
            print(f"- {model['model_name']}: {model['model_state']}")
    
    # 也可以直接返回数据而不保存文件
    # models_summary = generate_models_summary()
    # print(json.dumps(models_summary, ensure_ascii=False, indent=2))