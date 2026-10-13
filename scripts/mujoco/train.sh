#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

export USE_TF=0 USE_FLAX=0 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
# 使用 StarVLA 原生 Trainer + Accelerate DDP。这里不启用 DeepSpeed。
export STARVLA_DISABLE_DEEPSPEED=1
# 按服务器实际网卡配置取消注释：
# export NCCL_SOCKET_IFNAME=bond0
# export NCCL_IB_HCA=mlx5_2,mlx5_3

# ---------- 模型（已填入你提供的服务器模型路径） ----------
Framework_name=QwenFast
base_vlm=/home/chenzengjue/Qwen2.5-VL-3B-Instruct-Action
fast_tokenizer=playground/Pretrained_models/fast
freeze_module_list=qwen_vl_interface.model.model.visual
use_state_prompt=true
gradient_checkpointing=true
# 修改 codec 参数后重新执行 codec_check；训练/推理必须一致。
action_horizon=8
fast_scale=1000
fast_min_token=-4096
max_new_tokens=256

# ---------- 数据 / 输出 ----------
data_root_dir=data/block_lerobot
data_mix=mujoco_block_train
max_train_episodes=null  # 小规模检查可改成 2。
run_root_dir=outputs
run_id=mujoco_qwenfast_native
output_dir="${run_root_dir}/${run_id}"

# ---------- 官方 Trainer 参数：按 optimizer step 训练和保存 ----------
num_processes="${NPROC:-2}"
per_device_batch_size=1
gradient_accumulation_steps=16
max_train_steps=20000
num_warmup_steps=1000
save_interval=1000
logging_frequency=20
eval_interval=0  # 官方内置评估取训练 batch；独立 val 请用 offline_eval。
learning_rate=2e-5
weight_decay=0.01
gradient_clipping=1.0
num_workers=4
seed=42
is_resume=false

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${CONFIG:-configs/mujoco/qwenfast.yaml}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-configs/mujoco/accelerate_ddp.yaml}"

# 官方入口支持 --key value / --key=value；兼容旧脚本末尾的 key=value。
extra_args=()
for argument in "$@"; do
  if [[ "$argument" != --* && "$argument" == *=* ]]; then
    extra_args+=("--$argument")
  else
    extra_args+=("$argument")
  fi
done

# 参数优先级：YAML < 本脚本变量 < 末尾追加参数。
# 保存 shell 快照和完整终端日志；官方 trainer 另保存 config.full.yaml。
mkdir -p logs
stamp="$(date +%Y%m%d_%H%M%S)_$$"
cp "${BASH_SOURCE[0]}" "logs/launch_${stamp}.sh"
log_file="logs/train_${run_id}_${stamp}.log"
printf 'Native StarVLA trainer; console log: %s\n' "$log_file"
"$PYTHON_BIN" -m accelerate.commands.launch \
  --config_file "$ACCELERATE_CONFIG" \
  --num_processes "$num_processes" \
  starVLA/training/train_starvla.py \
  --config_yaml "$CONFIG" \
  --framework.name "$Framework_name" \
  --framework.qwenvl.base_vlm "$base_vlm" \
  --framework.qwenvl.gradient_checkpointing "$gradient_checkpointing" \
  --framework.use_state_prompt "$use_state_prompt" \
  --framework.action_model.fast_tokenizer "$fast_tokenizer" \
  --framework.action_model.action_horizon "$action_horizon" \
  --framework.action_model.fast_scale "$fast_scale" \
  --framework.action_model.fast_min_token "$fast_min_token" \
  --framework.action_model.max_new_tokens "$max_new_tokens" \
  --datasets.vla_data.data_root_dir "$data_root_dir" \
  --datasets.vla_data.data_mix "$data_mix" \
  --datasets.vla_data.per_device_batch_size "$per_device_batch_size" \
  --datasets.vla_data.num_workers "$num_workers" \
  --datasets.vla_data.max_train_episodes "$max_train_episodes" \
  --trainer.freeze_modules "$freeze_module_list" \
  --trainer.gradient_accumulation_steps "$gradient_accumulation_steps" \
  --trainer.max_train_steps "$max_train_steps" \
  --trainer.num_warmup_steps "$num_warmup_steps" \
  --trainer.save_interval "$save_interval" \
  --trainer.logging_frequency "$logging_frequency" \
  --trainer.eval_interval "$eval_interval" \
  --trainer.learning_rate.base "$learning_rate" \
  --trainer.optimizer.weight_decay "$weight_decay" \
  --trainer.gradient_clipping "$gradient_clipping" \
  --trainer.is_resume "$is_resume" \
  --run_root_dir "$run_root_dir" \
  --run_id "$run_id" \
  --seed "$seed" \
  "${extra_args[@]}" 2>&1 | tee "$log_file"
