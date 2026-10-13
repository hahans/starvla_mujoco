# A100 80GB：Qwen FAST 全参数 GRPO 与多环境采集

掉点排查后的新实验建议使用 [v2 说明](README_GRPO_V2_ZH.md) 中的 A100 命令，以同时启用解码对齐、FAST 长度约束、固定验证和修正后的 KL。下文保留旧 A100 参数及显存预算。

新增 `configs/mujoco/qwenfast_grpo_a100.yaml`，沿用同一个 MuJoCo 搬方块任务、SFT checkpoint、FAST 编解码和 RLinf GRPO 算法。**本机只有 RTX 3090，A100 上的完整 3.8B 全参数训练尚待实测。** 配置通过本地预检查；完整模型的显存预算不能用 LoRA 的约 7.5 GiB 测试数字推断。

| 配置 | 默认值及含义 |
|---|---|
| `train_mode` | `full`：训练所有 Qwen 视觉和语言参数，无 LoRA |
| `torch_dtype` | `float32`：主权重、梯度及 Adam 状态 FP32；CUDA forward 使用 BF16 autocast |
| `num_envs` | 8 个独立 MuJoCo 实例，同时进入一次批量 VLA 生成 |
| `groups_per_iteration` | 2 个不同初始场景 |
| `group_size` | 每个场景采样 4 条轨迹，共 8 条/iteration |
| `max_steps` / `execute` | 每轨迹最多 500 步，每次执行 8 步动作 chunk |
| `learning_rate` | 1e-6 |
| `kl_beta` | 0.02，独立冻结的 BF16 SFT reference |
| 更新微批量 | 每次反传 1 个 action chunk，累积整个 iteration 后更新 |

`num_envs` 控制批量生成宽度；`group_size` 控制优势归一化分组，两者不是同一个参数。环境数不足时分多批采集，保留原有 group 顺序；提前成功或生成无效 FAST 的环境退出当前推理 batch，其 EOS 后 padding 不参与概率与损失。MuJoCo 物理/渲染仍在同一个进程顺序执行，目前没有 CPU 子进程并行或 Ray 多机调度。

## 显存和磁盘预算

本地 Qwen checkpoint 的索引记录 `total_parameters=3758262272`。按全部参数可训练、普通 FP32 AdamW、GPU 上保留 BF16 reference 计算：

| 项目 | 估算占用 |
|---|---:|
| FP32 主权重 | 14.0 GiB |
| FP32 梯度 | 14.0 GiB |
| 两份 FP32 Adam moments | 28.0 GiB |
| BF16 固定 reference | 7.0 GiB |
| 上述静态部分合计 | **约 63.0 GiB** |

此外还有 autocast 权重缓存、activation、KV cache、生成 scores、CUDA context 和 allocator 预留。启用非重入梯度检查点；AdamW 显式 `foreach=False`，避免 foreach 更新额外持有整组参数大小的临时张量。80GB 是可尝试的目标，不能保证 8 环境一定不 OOM，实际看 `peak_gpu_allocated_gib` 和 `peak_gpu_reserved_gib`。

全参数 checkpoint 含 FP32 模型和 Adam 状态，优化器初始化后约 **42 GiB/份**，未计原始 SFT 和基础模型资产；保存前会检查空间。默认 100 轮、每 5 轮保存一次，会累计约 840 GiB，代码不会自动删除旧 checkpoint。请选择足够大的输出盘，或显式增大 `save_interval`。当前开发机可用磁盘不足以保存一个已更新的完整 checkpoint，不应在这里做 full 3.8B 优化测试。

## 在 A100 服务器启动

先把 `starvla_mujoco` 代码、原生 SFT run metadata/权重、Qwen2.5-VL-Action 基础模型、FAST tokenizer 和指定版本的 `Rlinf4/Rlinf4` 带到服务器。使用原已验证依赖环境（或 `requirements-mujoco-grpo.txt`）；PyTorch/CUDA 安装需匹配该服务器。完整训练无需额外安装 Ray/FSDP。

在 `starvla_mujoco` 目录执行，先修改 A100 YAML 的 `checkpoint`、`base_vlm`、`fast_tokenizer`、`rlinf_root`、`output` 为服务器上的路径：

```bash
conda activate starvla
nvidia-smi

# 检查资产、渲染及 contract；不会分配完整模型，不能据此宣称显存够用。
bash scripts/mujoco/train_grpo.sh --config configs/mujoco/qwenfast_grpo_a100.yaml --check

# 先验证 2 环境短闭环。8 步不能完成任务，同奖励时会跳过 optimizer step。
bash scripts/mujoco/train_grpo.sh --config configs/mujoco/qwenfast_grpo_a100.yaml \
  num_envs=2 groups_per_iteration=1 group_size=2 max_steps=8 iterations=1 \
  output=/large_disk/grpo_a100_smoke

# 正式 8 环境、全参数训练。
bash scripts/mujoco/train_grpo.sh --config configs/mujoco/qwenfast_grpo_a100.yaml
```

短闭环若 `updated=false`，Adam moments 尚未初始化，峰值显存会明显低估真正优化时的需求。要验证真实 backward + Adam + 完整保存/重载，可在空闲 A100 80GB 上运行下列可选 GPU 测试。它使用**人工优势，仅测试工程链路**，不代表任务学习效果；测试会生成至少一份完整 checkpoint，请确保临时目录有足够空间：

```bash
mkdir -p /large_disk/grpo_test_tmp
TMPDIR=/large_disk/grpo_test_tmp RUN_GRPO_GPU_TEST=1 \
GRPO_GPU_CONFIG=configs/mujoco/qwenfast_grpo_a100.yaml \
MUJOCO_GL=osmesa USE_TF=0 USE_FLAX=0 STARVLA_DISABLE_DEEPSPEED=1 \
python -m pytest -q -s tests/mujoco/test_grpo_gpu.py
```

完整检查需要较多主机 RAM（加载 SFT、恢复 optimizer、保存/重载）；建议为该流程准备至少 128GB 可用主机内存。测试会输出峰值显存和 batch 大小；低于 80GB 级别的 GPU 会跳过 full 测试。CUDA 设备应为空闲或有足够剩余显存，不能只看显卡总容量。

## 调整环境数

先验证 8 环境；如果显存不足，将 `num_envs` 降到 4 或 2。这仍然每轮收集 8 条轨迹，只是分批推理，GRPO 分组不变。

```bash
# 每轮 8 条轨迹，分成两批推理。
bash scripts/mujoco/train_grpo.sh --config configs/mujoco/qwenfast_grpo_a100.yaml \
  num_envs=4 output=/large_disk/grpo_a100_env4

# 有显存余量后：每轮 4 个场景 × 4 条轨迹，共 16 环境。
bash scripts/mujoco/train_grpo.sh --config configs/mujoco/qwenfast_grpo_a100.yaml \
  num_envs=16 groups_per_iteration=4 output=/large_disk/grpo_a100_env16
```

如果 OOM 发生在全参数反传/Adam 更新阶段，减小环境数未必有用，因为更新微批量已经是 1；需要减少可训练参数、优化器状态 offload 或 FSDP 等额外适配。当前代码不会自动切回 LoRA，也不会在失败后悄悄降低 batch。

日志包含 `train_mode`、`trainable_parameters`、`num_envs`、`rollout_seconds`、`env_steps_per_second`、峰值 allocated/reserved 显存及原有奖励/成功率指标。只有在同模型、相同轨迹长度和实际发生更新的条件下，显存与速度比较才有意义。

## 恢复和评估

full checkpoint 目录使用 `pytorch_model.pt` + `training_state.pt`，不生成 `adapter/`。`grpo_metadata.json` 记录 `train_mode=full`。恢复时仍从原始 SFT 构建 reference，再加载已更新的 actor 和 optimizer，避免把上次 actor 误当作新的 SFT reference。

```bash
bash scripts/mujoco/train_grpo.sh --config configs/mujoco/qwenfast_grpo_a100.yaml \
  --resume /large_disk/grpo_a100/checkpoints/iteration_000005 \
  iterations=200 output=/large_disk/grpo_a100_resumed
```

恢复配置必须与原 run 一致（除 iterations、output、save_interval），并使用相同实现版本。v2 拒绝精确恢复旧实现的 optimizer，但保留旧权重评估能力。不能直接拿 LoRA optimizer 状态恢复成 full。更换训练方式需从相应完整权重新开实验。

现有 `mujoco_task.evaluate --checkpoint /path/to/iteration_000005` 可识别 full 目录；不依赖原 SFT 权重文件，但仍需要对应 Qwen processor/基础架构和 FAST 资产。其他评估参数见 [通用 GRPO 说明](README_GRPO_ZH.md)。

## 本地验证记录（2026-09-24）

- 任务测试集：39 passed、2 skipped；Ruff lint/format 检查通过。
- 微型真实 Qwen：不同长度、多图像输入的批量生成与独立 log-prob 一致；视觉参数获得非零梯度；full 更新保持 reference 不变；保存与恢复 actor/optimizer、full 目录评估加载通过。
- 真实 `steps_9000` + 3090：2 路 batch 的生成、LoRA 更新与重载通过；log-prob 最大差 `7.6294e-6`。这验证批量路径，不代表 full 3.8B 已在 A100 跑通。
- `outputs/grpo_batch2_validation` 完成两个独立 MuJoCo 环境各 8 步，输出有效、分组正确；短测组内同奖励，按设计跳过更新。A100 full 的显存、吞吐和任务学习效果尚未测量。
