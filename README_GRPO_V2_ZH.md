# Qwen FAST GRPO 修正配置

这版从 `mujoco_qwenfast_native/checkpoints/steps_9000_pytorch_model.pt` 重新开始。旧 SFT、LoRA 和评估结果保留；没有把旧 adapter 合并回 SFT。RLinf 调用路径的审计见 [AUDIT_RLINF_GRPO_ZH.md](AUDIT_RLINF_GRPO_ZH.md)。

## 先了解改变了什么

- 训练、reference、概率回算、验证共用 FAST 词表、UTF-8 长度约束和 `repetition_penalty=1.05`。回算惩罚只使用当前 token **之前**的前缀，不看未来标签。采样温度为 0.7，验证为 greedy。
- FAST 每块必须恰好解码出 8 × 7 个系数，禁止提早 EOS 和超长系数序列；按 ByteLevel BPE 的原始字节识别跨 token 的 UTF-8 字符，不补零、不截断动作。格式合法不保证动作能成功。
- 保留有效的 KL 梯度，即使整批奖励相同；KL 增加与 RLinf 相同的数值范围保护。真正零梯度时不调用 Adam，防止仅靠残余动量改变模型。
- 增加 `ratio`、`clip_fraction`、`policy_loss`、`replay_logprob_abs_max`、`kl_only` 指标。采样概率和回算概率的差异只记入这些指标，不再中止更新。
- 与 RLinf `importance_sampling_fix` 相同：PPO 的 `old_logprobs` 保持采样时的 rollout log-prob；优势乘以 `clamp(exp(回算 − 采样), min=importance_sampling_clip)`，默认下界是 1.0，没有上限。权重无梯度。额外记录概率平均误差、未修正时超出 clip 区间的 token 比例和重要性权重均值。
- 用原任务成功标记做 sparse reward；学习率 2e-5，KL 系数 0，clip 0.2。一次更新前采集：每轮 8 个初态 × 每组 8 条 × 2 轮，共 128 条轨迹。每一轮重新抽初态，同一组 8 条仍共用一个 seed；优势按每组 8 条计算。这 128 条打乱后按 minibatch 32 做 4 步 Adam。这是待实测的实验设置，不能保证比旧参数更好。
- 训练前测 SFT；每五轮在配置的固定种子上评估。先比较成功数，再比较非法输出数；完全相同则保留原最佳模型。`early_stop_patience: 0` 时跑满 `iterations`，只记录最佳 checkpoint，不中断训练。设为正整数时，连续这么多次评估未改善才停止。没有超过 SFT 时，`best_checkpoint.json` 继续指向原 SFT。
- 执行动作 chunk 时，只渲染最后一步会被策略读取的 RGB；仍执行每个物理控制步和成功检查。MuJoCo 物理仍是单进程 CPU 执行。

原生 SFT、旧 LoRA 的 `evaluate` 默认继续走历史解码。带 v2 解码配置的新 checkpoint 会自动恢复其约束和重复惩罚。不能把不同解码设置下的成功率直接归因于权重变化。

## 在 3090 上启动

当前 v2 默认 `train_seed_min: 0`、`train_seed_max: 20`，包含两端，共 21 个可能的场景 seed。每个 GRPO 组均匀、有放回地抽一个 seed，同组 8 条轨迹共用它。`rollout_epoch: 2` 在同一次更新前采 2 轮，每一轮重新抽 seed。第一轮沿用原来按 iteration 确定的序列，后面各轮把轮次放进随机种子，因此恢复时不必先重放前面的轮次。`seed` 控制这条随机序列和 PyTorch 随机性，不再是场景 seed 起点。验证和 `num_envs` 不会改变场景序列。改变 `rollout_epoch` 是新实验，不能续到旧 checkpoint 上。

命令末尾加 `train_seed_min=0 train_seed_max=20` 可显式指定范围；修改为其他上下界即可换范围。两项同时设为 `null` 可恢复旧的顺序递增规则。更换范围需新开实验，不能直接精确续训之前采用不同 seed 规则的 checkpoint。训练集监测允许与训练 seed 重叠；切回 `validation_split=val` 后仍会检查训练范围是否与固定验证 seed 重叠。

以下命令从 `starvla_mujoco` 目录执行：

```bash
conda activate starvla
bash scripts/mujoco/train_grpo.sh --config configs/mujoco/qwenfast_grpo_v2.yaml --check

# 仅测试采样、验证、保存流程；8 步不足以完成任务。
bash scripts/mujoco/train_grpo.sh --config configs/mujoco/qwenfast_grpo_v2.yaml \
  num_envs=2 groups_per_iteration=1 group_size=2 max_steps=8 iterations=1 \
  validation_episodes=2 validation_interval=1 output=outputs/grpo_v2_smoke

# 完整任务训练，先保留每轨迹 500 步。
bash scripts/mujoco/train_grpo.sh --config configs/mujoco/qwenfast_grpo_v2.yaml
```

当前默认读取 `data/block_lerobot/block_train/meta/source.json` 中前 `validation_episodes` 条记录的 seed。训练前 SFT baseline、定期评估和最佳模型选择都使用这些固定初态的 greedy rollout，不是直接回放数据中的动作。结果 JSON 的 `split` 会标记为 `train`，因此这些数值属于训练集监测，不是独立验证集成功率。

可在命令末尾用 `validation_split=val validation_episodes=4` 切回验证集前四条；用 `validation_split=val validation_episodes=20` 做更完整的验证。更换 split 要新开实验，不能把旧 split 的最佳分数与新 split 混合比较；旧配置没有 `validation_split` 时仍按原来的 `val` 处理。

检查 `outputs/mujoco_qwenfast_grpo_v2/best_checkpoint.json` 中的 `best_checkpoint`。训练未优于基线时它是原 SFT `.pt`，不是 LoRA 目录。最佳记录、baseline 指标和早停计数也随每个 checkpoint 保存；中途恢复会保留它们。

## 在 A100 80GB 上使用同一套修正

```bash
bash scripts/mujoco/train_grpo.sh --config configs/mujoco/qwenfast_grpo_v2.yaml \
  train_mode=full torch_dtype=float32 num_envs=8 learning_rate=0.000001 \
  output=/large_disk/mujoco_grpo_v2_full
```

本机无 A100，这个完整模型的全参数组合尚未实测。显存和磁盘预算见 [A100 说明](README_GRPO_A100_ZH.md)。全参数模式包含独立冻结 SFT reference。验证时也会保存 checkpoint，所以开启每五轮验证时，仅增大 `save_interval` 不会减少验证节点的保存次数；一份已更新的 full checkpoint 约 42 GiB。

## 同条件比较 SFT 和已有 LoRA

```bash
USE_TF=0 USE_FLAX=0 STARVLA_DISABLE_DEEPSPEED=1 MUJOCO_GL=osmesa \
python -m mujoco_task.compare_grpo \
  --checkpoints outputs/mujoco_qwenfast_grpo/checkpoints/iteration_000040 \
  --base-vlm /home/czj/agent-vla-gpt/Qwen2.5-VL-3B-Instruct-Action \
  --fast-tokenizer playground/Pretrained_models/fast \
  --split val --episodes 20 --num-envs 4 \
  --execute 8 --max-steps 500 --decoding constrained --repetition-penalty 1.05 \
  --output outputs/grpo_paired_val
```

脚本只加载一次 SFT，再依次启用 LoRA，保证同一组种子、BF16 精度、解码规则和任务条件。每个候选结果立即落盘；只在 `val` 上输出选中的模型，train/test 对照仅用于诊断。旧 adapter 必须能找到 metadata 中记录的原 SFT 路径。

单独查看新权重的视频，把下列 `--checkpoint` 换成实际目录：

```bash
USE_TF=0 USE_FLAX=0 STARVLA_DISABLE_DEEPSPEED=1 MUJOCO_GL=osmesa \
python -m mujoco_task.evaluate \
  --checkpoint outputs/mujoco_qwenfast_grpo_v2/checkpoints/iteration_000005 \
  --base-vlm /home/czj/agent-vla-gpt/Qwen2.5-VL-3B-Instruct-Action \
  --fast-tokenizer playground/Pretrained_models/fast \
  --device cuda --torch-dtype bfloat16 --decoding auto \
  --data-root data/block_lerobot --split val --episodes 20 \
  --execute 8 --max-steps 500 --video --output outputs/grpo_v2_val_video
```

若单独评估 **SFT** 并与 v2 对比，必须显式加 `--decoding constrained --repetition-penalty 1.05`；SFT 的 `auto` 保留历史解码。重现旧评估用 `--decoding legacy`。`aligned` 只限制 FAST 词表；`constrained` 还限制系数数量。两者都可显式指定重复惩罚。

## 恢复与验证

v2 保存 `implementation_version=2`。旧权重仍可评估，但旧优化器状态不能在修正后的实现里声称“精确续训”，入口会拒绝这种恢复。新实验从 SFT 开始；v2 checkpoint 可以正常恢复到新输出目录：

```bash
bash scripts/mujoco/train_grpo.sh --config configs/mujoco/qwenfast_grpo_v2.yaml \
  --resume outputs/mujoco_qwenfast_grpo_v2/checkpoints/iteration_000005 \
  iterations=200 output=outputs/mujoco_grpo_v2_resumed

USE_TF=0 USE_FLAX=0 STARVLA_DISABLE_DEEPSPEED=1 MUJOCO_GL=osmesa \
python -m pytest -q tests/mujoco

RUN_GRPO_GPU_TEST=1 GRPO_GPU_CONFIG=configs/mujoco/qwenfast_grpo_v2.yaml \
GRPO_AUDIT_CHECKPOINT=outputs/mujoco_qwenfast_grpo/checkpoints/iteration_000040 \
GRPO_AUDIT_TRAJECTORIES=outputs/rollout_train_native \
USE_TF=0 USE_FLAX=0 STARVLA_DISABLE_DEEPSPEED=1 MUJOCO_GL=osmesa \
python -m pytest -q -s tests/mujoco/test_grpo_gpu.py
```

GPU 测试的人工优势仅验证概率、梯度、SFT reference 和保存重载。完成新一轮完整任务训练、固定 val 模型选择及独立 test 评估之前，不能据此声称强化学习成功率已提升。

`GRPO_AUDIT_TRAJECTORIES` 是可选的历史评估记录目录，提供时按已记录的实际动作重放到第 0/80/160/240 个控制步，覆盖更广的观察状态；它仅用于审计，不是训练数据或专家控制器。
