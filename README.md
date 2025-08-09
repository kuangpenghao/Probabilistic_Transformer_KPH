# 模型实时训练结果：http://10.15.89.226:8085（局域网访问）

# Huggingface Transformers Starter Code

General starter code for creative model architecture with huggingface transformer library. Users may implement customized llama model in `models` directory.

## Installation

Change the cuda version if it is not compatible. Developped with python 3.12.4.

```bash
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.0.post2/flash_attn-2.8.0.post2+cu12torch2.7cxx11abiFALSE-cp312-cp312-linux_x86_64.whl -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## Usage

Copy `run_clm.sh.template` to `run_clm.sh` and run it.

```bash
# for the first time execution
cp run_clm.sh.template run_clm.sh

# run the script
conda activate pt
sapp bash run_clm.sh
```

srun -G 8 -c 1 --mem=10M nvidia-smi

ls -la outputs/my-llama-tiny/

删除旧的checkpoint：rm -rf outputs/my-llama-tiny/*

tmux new -s session_name
tmux detach 或Ctrl+B,D//临时退出
tmux ls
tmux attach -t session_name//重新进入
杀死会话：进入后exit或tmux kill-session -t session_named

srun -N 1 -n 1 -X -u -p normal --gres=gpu:1 -c 2 --mem=1M -t 0-96:00:00 bash run_clm.sh

ss -tlnp | grep -E ":808[0-9]"

kill -9 1914709

cd /home/kuangph/hf-starter/Web_Display && ./stop_web_server.sh && sleep 2 && python3 web_server.py --port 8081