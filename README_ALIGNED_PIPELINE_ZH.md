# 对齐真机后的仿真：采集 600 条 → LeRobot → QwenFast 训练

适用于按 `real_raw2` 对齐后的仿真（UR5 CB 运动学、`policy_cam` 标定相机、DM 夹爪、实测桌面和方块位置，见 [README_REAL_ROBOT_ZH.md](README_REAL_ROBOT_ZH.md)“仿真与真机对齐”）。旧数据和旧权重的场景指纹不同，不能混用。

所有命令都在 `starvla_mujoco` 目录执行。下文用到的名字：

| 名字 | 含义 |
|------|------|
| `data/block_raw_real2` | 原始采集（HDF5 + `manifest.json`） |
| `data/block_lerobot_real2` | 转换后的 LeRobot v2.1 数据 |
| `mujoco_qwenfast_real2` | 训练 run 名，输出在 `outputs/mujoco_qwenfast_real2` |

## 0. 打包代码并上传

本机（Mac 上加 `COPYFILE_DISABLE=1`，避免打进 `._*` 文件）：

```bash
cd /Users/neow/Desktop/starvla_mujoco/starvla_mujoco
COPYFILE_DISABLE=1 bash scripts/mujoco/package.sh      # 生成 ../starvla_mujoco_code.tar.gz
scp ../starvla_mujoco_code.tar.gz USER@SERVER:/path/to/work/
```

服务器：

```bash
cd /path/to/work && tar -xzf starvla_mujoco_code.tar.gz && cd starvla_mujoco
```

## 1. 环境

```bash
conda create -n starvla python=3.10 -y
conda activate starvla
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124   # 按驱动选 CUDA 版本
pip install -r requirements-mujoco.txt
python -m mujoco_task.download_models --fast-only    # FAST tokenizer → playground/Pretrained_models/fast
```

渲染用 NVIDIA EGL（GPU 离屏渲染，不需要显示器和 OSMesa）。需要 NVIDIA 驱动自带的 EGL 库：`ls /usr/share/glvnd/egl_vendor.d/` 里应有 `10_nvidia.json`。Docker 容器要带 `NVIDIA_DRIVER_CAPABILITIES=all`（或包含 `graphics`），否则 EGL 会悄悄退回 Mesa 的 CPU 渲染。

另外需要 `Qwen2.5-VL-3B-Instruct-Action` 模型目录（带 `<robot_action_*>` 词表），下文记作 `$BASE_VLM`。

统一的环境变量（每个新终端先执行）：

```bash
export USE_TF=0 USE_FLAX=0 STARVLA_DISABLE_DEEPSPEED=1 TOKENIZERS_PARALLELISM=false
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export BASE_VLM=/path/to/Qwen2.5-VL-3B-Instruct-Action
```

确认 EGL 走的是 NVIDIA GPU：

```bash
python -c "
from mujoco_task.env import TransferEnv; from OpenGL import GL
e = TransferEnv(); e.reset(seed=0); print(GL.glGetString(GL.GL_RENDERER).decode()); e.close()" 2>/dev/null
# 正确：NVIDIA A100-SXM4-40GB/PCIe/SSE2 之类；出现 llvmpipe 说明退回了 Mesa，先修环境
```

> **渲染后端要全程一致。** 采集、`codec_check`、`evaluate`、GRPO 都要在 `MUJOCO_GL=egl` + NVIDIA 驱动下运行；NVIDIA EGL 与 Mesa/OSMesa 的像素不同，会改变策略输出（见 [AUDIT_RENDERING_ZH.md](AUDIT_RENDERING_ZH.md)）。各脚本没设 `MUJOCO_GL` 时默认用 `osmesa`，所以每个终端都要先执行上面的 `export`。仿真评估最好和采集用同型号 GPU、同版本驱动。采集用的驱动记录在 `manifest.json` 的 `gl_renderer`。

## 2. 采集前自检（约 1 分钟）

```bash
python -m pytest -q tests/mujoco/test_data.py tests/mujoco/test_ur5_robot.py tests/mujoco/test_real_robot.py
python -m mujoco_task.scene.run_scene --validate-task      # 专家完成一次搬运，失败会报错
```

## 3. 采集 600 / 20 / 10 条

```bash
python -m mujoco_task.collect --output data/block_raw_real2 \
  --train 600 --val 20 --test 10 --workers 16 --expect-renderer NVIDIA
```

- 每条 = `TransferEnv.reset(seed)` + 脚本专家，20 Hz，318 帧，128×128 图像。
- seed 互不重叠：train 从 0、val 从 100000、test 从 200000 开始；专家失败的 seed 自动跳过并记入 `skipped_unsuccessful_seeds`。
- 先写入 `data/block_raw_real2.partial`，全部完成才改名；已存在的目录不会覆盖。中断后删掉 `.partial` 重来。
- `--expect-renderer NVIDIA`：渲染器不是 NVIDIA 就直接报错退出，不写任何数据。
- 参考：A100 + NVIDIA EGL，4 进程 8 条约 13 s；630 条视 CPU 核数约几分钟到十几分钟。原始数据约 2.5 GB。
- 每个进程在默认 GPU（EGL 设备 0）上建一个渲染上下文，显存占用很小；`--workers` 不要超过 CPU 核数（物理仿真仍在 CPU 上）。

## 4. 转换为 LeRobot

```bash
python -m mujoco_task.convert --source data/block_raw_real2 --output data/block_lerobot_real2
```

逐条校验 SHA256、形状、时间戳、seed 和成功标记，生成 `block_train/ block_val/ block_test/`，以及只用 train 统计的 `normalization.json`。

## 5. 训练前检查

```bash
# 数据能被训练 loader 读出（不加载 VLM）
python -m mujoco_task.check_data --datasets.vla_data.data_root_dir data/block_lerobot_real2

# FAST 编解码误差 + 回放原始/解码动作，必须全部成功（非零退出码就先排查）
python -m mujoco_task.codec_check --data-root data/block_lerobot_real2 \
  --episodes 10 --replay --output outputs/codec_real2_train10.json

# 可选：数据集图像/状态与实时渲染逐帧对比，确认渲染后端一致
python -m mujoco_task.audit_observations --data-root data/block_lerobot_real2 \
  --split train --episodes 2 --output outputs/audit_real2
```

## 6. 训练

先把 `scripts/mujoco/train.sh` 顶部的 `base_vlm=` 改成 `$BASE_VLM` 的路径（或在命令末尾加 `--framework.qwenvl.base_vlm $BASE_VLM`）。末尾追加的参数优先级最高。

```bash
# 6.1 小规模拟合（2 条、200 步），确认能跑通、loss 下降
CUDA_VISIBLE_DEVICES=0,1 NPROC=2 bash scripts/mujoco/train.sh \
  --framework.qwenvl.base_vlm $BASE_VLM \
  --datasets.vla_data.data_root_dir data/block_lerobot_real2 \
  --datasets.vla_data.max_train_episodes 2 \
  --trainer.max_train_steps 200 --trainer.num_warmup_steps 10 --trainer.save_interval 100 \
  --run_id mujoco_real2_debug

# 6.2 正式训练（与之前相同：2 卡 × batch 1 × 累积 16 = 32，20000 步，每 1000 步存一次）
CUDA_VISIBLE_DEVICES=0,1 NPROC=2 bash scripts/mujoco/train.sh \
  --framework.qwenvl.base_vlm $BASE_VLM \
  --datasets.vla_data.data_root_dir data/block_lerobot_real2 \
  --run_id mujoco_qwenfast_real2
```

- 无 state 版本：再加 `--framework.use_state_prompt false`，并换一个 `--run_id`。
- 600 条 × 318 帧 = 190800 帧，与旧数据相同，约 5963 步遍历一次。
- 日志在 `logs/train_*.log`，权重在 `outputs/mujoco_qwenfast_real2/checkpoints/steps_*_pytorch_model.pt`。
- 中断后继续：同样的命令加 `--trainer.is_resume true`（只恢复模型和 step，优化器重新初始化）。

## 7. 评估

```bash
CKPT=outputs/mujoco_qwenfast_real2/checkpoints/steps_10000_pytorch_model.pt

# 离线：val 上的 token CE、非法生成率、动作误差（不渲染）
python -m mujoco_task.offline_eval --checkpoint $CKPT --base-vlm $BASE_VLM \
  --fast-tokenizer playground/Pretrained_models/fast \
  --data-root data/block_lerobot_real2 --split val --episodes 20 --stride 8 \
  --device cuda --output outputs/offline_val_real2.json

# 闭环：先 train seed 看是否学会，再用 val 选 checkpoint，最后测 test
for split in train val test; do
  CUDA_VISIBLE_DEVICES=0 python -m mujoco_task.evaluate --checkpoint $CKPT --base-vlm $BASE_VLM \
    --fast-tokenizer playground/Pretrained_models/fast --device cuda --torch-dtype bfloat16 --decoding auto \
    --data-root data/block_lerobot_real2 --split $split --episodes 10 --num-envs 10 \
    --execute 8 --max-steps 500 --video --output outputs/eval_real2_$split
done
```

`results.json` 给出成功率，`--video` 保存每条的视频。

## 8. 上真机

训练机只需要传回这三个文件（保持目录结构）：

```bash
tar -czf real2_step10000.tar.gz -C outputs \
  mujoco_qwenfast_real2/config.full.yaml mujoco_qwenfast_real2/normalization.json \
  mujoco_qwenfast_real2/checkpoints/steps_10000_pytorch_model.pt
```

之后按 [README_REAL_ROBOT_ZH.md](README_REAL_ROBOT_ZH.md) 启动 `serve` / `run`，注意：

- 先 `python -m mujoco_task.ur5_robot go-start --ip 192.168.1.101 --gripper dm --gripper-stroke-m 0.07`。起始姿态就是 `go_home.py` 的姿态、夹爪全开，`joint_offset` 不要再加。
- 相机保持采 `real_raw2` 时的位置；挪动过就重新录一条，用 `python -m mujoco_task.real_overlay` 检查叠加是否还对齐。
- 仿真专家一条约 16 s，真机遥操演示 54–80 s，模型学到的是仿真速度；首次上机用 `--max-joint-delta 0.03` 并 `--confirm`。
