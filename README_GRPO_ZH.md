# 已训练 Qwen FAST + MuJoCo + RLinf GRPO

**针对旧 LoRA 掉点的修正配置和启动方式见 [README_GRPO_V2_ZH.md](README_GRPO_V2_ZH.md)，与指定 RLinf 的逐项对照见 [AUDIT_RLINF_GRPO_ZH.md](AUDIT_RLINF_GRPO_ZH.md)。下文中的默认参数描述保留旧配置；新实验请显式使用 `--config configs/mujoco/qwenfast_grpo_v2.yaml`。旧 SFT/LoRA 可继续评估，旧优化器状态不能在 v2 实现中精确续训。**

此入口从现有 **StarVLA SFT checkpoint** 开始，用真实 MuJoCo 闭环结果优化 Qwen FAST。3090 配置默认训练语言注意力 LoRA；新增 **A100 80GB 全参数、多环境配置见 [README_GRPO_A100_ZH.md](README_GRPO_A100_ZH.md)**。默认权重为 `mujoco_qwenfast_native/checkpoints/steps_9000_pytorch_model.pt`，原权重不会被覆盖。

当前实现是**单进程、单 GPU**，支持多环境批量 VLA 推理，MuJoCo 物理仍在 CPU 顺序步进；直接调用指定 `../Rlinf4/Rlinf4/rlinf` 中的 `compute_grpo_advantages` 和 `compute_grpo_actor_loss_fn`，尚未接入 RLinf Ray/FSDP 分布式 Runner。LoRA 的 reference 使用关闭 adapter 的 SFT，full 模式使用独立冻结的 SFT 模型。导入时检查 RLinf 的实际路径，保存算法源文件 SHA256，防止误用工作区外层的另一份 RLinf。

## 运行

以下命令都从 `starvla_mujoco` 目录执行。现有 `starvla` conda 环境可用；额外 RL 依赖是 `peft==0.17.1`，完整依赖列表见 `requirements-mujoco-grpo.txt`。

```bash
conda activate starvla
pip install -r requirements-mujoco-grpo.txt

# 无 VLA 加载、无优化：检查本地资产、MuJoCo 渲染/contract、RLinf 来源。
bash scripts/mujoco/train_grpo.sh --check

# 很短的闭环连通性检查。16 步不足以完成任务，不代表训练效果。
bash scripts/mujoco/train_grpo.sh iterations=1 group_size=2 max_steps=16 \
  output=outputs/grpo_smoke

# 完整任务训练：每组 4 条同初态轨迹，每条最多 500 控制步。
bash scripts/mujoco/train_grpo.sh
```

脚本自动切换到项目目录，可指定解释器：

```bash
PYTHON=/home/czj/anaconda3/envs/starvla/bin/python bash scripts/mujoco/train_grpo.sh --check
```

服务器上的路径不同，在 `configs/mujoco/qwenfast_grpo.yaml` 修改或末尾覆盖：

```bash
bash scripts/mujoco/train_grpo.sh \
  checkpoint=/path/to/run/checkpoints/steps_9000_pytorch_model.pt \
  base_vlm=/path/to/Qwen2.5-VL-3B-Instruct-Action \
  fast_tokenizer=/path/to/fast rlinf_root=/path/to/Rlinf4 \
  output=outputs/my_grpo_run
```

SFT 权重旁需保留原 run 的 `config.full.yaml` 和 `normalization.json`。FAST horizon、scale、min_token、相机、控制周期、状态提示和动作归一化继承 SFT；不重新拟合归一化。`MUJOCO_GL` 默认 `osmesa`，可通过环境变量选 `egl`。默认 bf16 加载，LoRA rank 8；不采用量化。

## 训练语义

1. 每组使用同一个场景 seed 重置，得到相同的方块初始位置；只重置仿真随机数，不重复设置策略随机数。不同成员从策略独立采样。默认使用新 seed，从 4000000 开始，不读取 train/val/test 示范。
2. Qwen 接收原相机 RGB、21 维机器人状态和任务文字，预测 8 步 FAST 动作，每次执行 8 步后重新观察。只允许 2048 个 FAST token 和 `<|im_end|>`，temperature=0.7，关闭 top-k/top-p 截断。不调用专家。
3. 保存每个实际采样 token 和其 behavior log-prob。更新时把相同 token 拼回原始多模态 prompt，以相同词表约束和温度回算 log-prob；prompt 不参与损失，EOS 参与。没有通过动作重新编码来猜测采样序列。
4. 组内 episode 总奖励按 RLinf 的均值和样本标准差归一化，广播到该轨迹全部 action token。用 RLinf PPO clip 损失做 GRPO 更新，每条轨迹先对其 token 求均值，再对轨迹求均值，避免长轨迹主导梯度。无需 critic。
5. 默认加入 beta=0.02 的 sampled k3 KL。关闭 LoRA adapter 得到固定 SFT reference，因此不需额外加载一份 reference。所有旧概率/参考概率均无梯度。
6. 非法 FAST、空序列、缺少 EOS 都终止该轨迹，奖励 -1；仍保留 token 参加训练，不把非法结果替换为零动作。成功立即停止，达到 max_steps 按截断 episode 结算。
7. 同组奖励全相同时优势为 0。修正后的实现仍保留已启用的 SFT KL；若策略梯度和 KL 梯度均为零，则跳过 optimizer step，记录 `updated=false`。

默认 `reward_mode=staged`，奖励只在 episode 结束算一次：

| 项目 | 奖励 |
|---|---:|
| 原场景严格成功判定 | +1 |
| 曾双指接触绿块 | +0.1 |
| 曾黑爪真实抬升绿块 | +0.2 |
| 已抬升后的终点接近目标 | +0.2 × exp(-XY 距离 / 0.05m) |
| 曾发生非预期接触 | -0.2 |
| 红块位移惩罚 | -0.2 × min(最大位移 / 0.05m, 1) |
| 非法 FAST 输出 | 总奖励固定为 -1 |

`reward_mode=sparse` 时只有成功 1、未成功 0、非法输出 -1。部分进展的奖励不等同成功率；成功仍要求真实抓取抬升、释放后稳定放到右边、红块不明显移动、白臂不动、无异常接触。方块真值仅用于奖励/诊断，不进入模型输入。若 valid/invalid 和部分进展都没有组内差异，GRPO 无法产生更新；应先评估 SFT 基线、检查无效率，再调整温度或奖励。

## 输出、恢复、评估

```text
outputs/mujoco_qwenfast_grpo/
  grpo_config.yaml
  rlinf_sources.json
  metrics.jsonl
  checkpoints/iteration_000005/
    adapter/                         # LoRA 权重，依赖原始 SFT base
    training_state.pt                # optimizer、CPU/CUDA RNG、iteration
    grpo_config.yaml
    grpo_metadata.json               # SFT 来源及 RLinf 指纹
    config.full.yaml
    normalization.json
    COMPLETE
```

指标含 `success_rate`、`invalid_rate`、`reward_mean`、`nonzero_advantage_fraction`、`loss`、`kl`、`grad_norm`，并保留每条轨迹的 seed、最终任务证据和失败原因。iteration 是一次组采集加 update_epochs 次更新，不是环境步数。每 save_interval 保存一次，最后一轮也保存。写入先经过 `.partial`，完成后原子重命名，已有输出目录拒绝覆盖。

恢复只支持 iteration 边界，不支持正在执行的轨迹；会恢复 adapter、optimizer、CPU/CUDA RNG，继续下一个场景 seed。使用同硬件和依赖环境；GPU 数值计算不承诺逐位重现。恢复时除总 iterations、输出位置和保存间隔外，配置及 RLinf 算法指纹必须一致。

```bash
bash scripts/mujoco/train_grpo.sh \
  --resume outputs/mujoco_qwenfast_grpo/checkpoints/iteration_000005 \
  iterations=200 output=outputs/mujoco_qwenfast_grpo_resumed
```

已有闭环评估入口支持 GRPO adapter 目录。`--decoding auto` 对旧 SFT/LoRA 保留原有 greedy 方式，对 v2 checkpoint 恢复保存的解码配置。为 base 和 RL checkpoint 使用相同 seeds、execute、max_steps、加载精度及解码规则，先用 val 选模型，再使用 test：

```bash
USE_TF=0 USE_FLAX=0 MUJOCO_GL=osmesa python -m mujoco_task.evaluate \
  --checkpoint outputs/mujoco_qwenfast_grpo/checkpoints/iteration_000005 \
  --base-vlm /home/czj/agent-vla-gpt/Qwen2.5-VL-3B-Instruct-Action \
  --fast-tokenizer playground/Pretrained_models/fast \
  --torch-dtype bfloat16 --device cuda \
  --data-root data/block_lerobot --split val --episodes 20 \
  --execute 8 --max-steps 500 --video --output outputs/grpo_val
```

迁移到其他机器时，需要一起传 SFT 原权重/metadata 和 GRPO adapter。SFT 位置变化时在评估命令中加 `--sft-checkpoint /new/path/to/steps_9000_pytorch_model.pt`。不把 LoRA adapter 当作可独立运行的完整模型。

## 测试与限制

```bash
USE_TF=0 USE_FLAX=0 STARVLA_DISABLE_DEEPSPEED=1 python -m pytest -q tests/mujoco

# 显式可选：加载本地真实 SFT，采样、对齐 log-prob、反传和 adapter 保存重载。
# 这个测试用人工优势验证梯度链路；不是任务奖励训练或成功率实验。
RUN_GRPO_GPU_TEST=1 MUJOCO_GL=osmesa USE_TF=0 USE_FLAX=0 \
  STARVLA_DISABLE_DEEPSPEED=1 python -m pytest -q -s tests/mujoco/test_grpo_gpu.py
```

单元测试覆盖组内归一化、EOS/temperature/词表约束、next-token 对齐、零方差跳过、真实 RLinf 损失反传、非法轨迹保留、成功提前结束和参数校验。GPU 测试缺省跳过，不自动下载模型。正式长时间训练及成功率提升需由完整任务实验验证；顺序采样大量自回归 token 会较慢。

2026-09-24 本机验证记录：

- MuJoCo 资产/渲染/contract 预检查通过，Ruff 和 shell 语法检查通过。
- 使用真实 `steps_9000`，在 RTX 3090 上生成 167 个 token、解码为有效动作；采样与 teacher-forcing log-prob 最大差 `3.8147e-6`。人工优势梯度测试完成 LoRA 更新、固定参考检查、保存和重载一致性检查；测试峰值 allocated 显存约 7.48 GiB，不代表完整训练的最大需求。
- `outputs/grpo_smoke_validation` 完成 1 轮、2 条 × 16 控制步短闭环，无非法 FAST；两条奖励相同，因此 `updated=false`。这不是完成任务的时长，成功率 0 不能作为完整任务成功率估计。
- `outputs/grpo_resume_validation` 从上述 checkpoint 恢复，继续 iteration 2、seed 4000001，并保存 `iteration_000002`。
- 尚未启动默认 100 轮完整训练，尚无 GRPO 成功率提升结论。上述 GPU 梯度测试保存的 adapter 仅用于测试，不应用作训练结果。

TODO(agent): 此实现尚未接入 RLinf Ray worker/FSDP，多卡分布式训练需要另做模型/环境 worker 适配；当前脚本不能通过 `torchrun` 或 `accelerate launch` 启动多进程。


USE_TF=0 USE_FLAX=0 STARVLA_DISABLE_DEEPSPEED=1 MUJOCO_GL=osmesa \
python -m mujoco_task.evaluate \
  --checkpoint /home/czj/disk1/RLinf-Wan2/starvla_mujoco/outputs/mujoco_qwenfast_grpo/checkpoints/iteration_000040 \
  --base-vlm /home/czj/agent-vla-gpt/Qwen2.5-VL-3B-Instruct-Action \
  --fast-tokenizer /home/czj/disk1/RLinf-Wan2/starvla_mujoco/playground/Pretrained_models/fast \
  --data-root /home/czj/disk1/RLinf-Wan2/starvla_mujoco/data/block_lerobot \
  --split train --episodes 10 \
  --execute 8 --max-steps 500 \
  --device cuda --torch-dtype bfloat16 \
  --video \
  --output /home/czj/disk1/RLinf-Wan2/starvla_mujoco/outputs/eval_grpo_iter40_train10_lora
