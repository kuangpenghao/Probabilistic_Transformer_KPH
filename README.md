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
bash run_clm.sh
```

srun -G 8 -c 1 --mem=10M nvidia-smi

ls -la outputs/my-llama-tiny/

删除旧的checkpoint：rm -rf outputs/my-llama-tiny/*

