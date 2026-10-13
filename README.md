# 双 UR5e 黑爪搬绿色方块：StarVLA QwenFast

目标：绿色方块起初在红色左边，黑色夹爪把绿色放到红色右边。白色机械臂保持初始姿态。继续使用原来的 MuJoCo 场景、相机、控制器和成功判定。

**渲染一致性（2026-09-27）**：当前 SFT 数据与本机 OSMesa/Mesa 画面一致；默认 NVIDIA EGL 的实际抗锯齿及光照结果不同，会影响策略。完整本机对照和 Mesa EGL 启动方式见 [AUDIT_RENDERING_ZH.md](AUDIT_RENDERING_ZH.md)。仅把 `MUJOCO_GL` 改为 `egl` 不保证输入画面一致。

真机 RGB 的保边平滑、固定颜色校正及参考图匹配脚本见 [README_RGB_TO_SIM_ZH.md](README_RGB_TO_SIM_ZH.md)。这是可选图像预处理，不保证还原渲染结果或恢复抓取性能。

现有 600 条示范的 HRC Planner 四类子任务标注与 DeepSeek API 批处理入口见 [README_PLANNER_ANNOTATION_ZH.md](README_PLANNER_ANNOTATION_ZH.md)。

**带子任务标注的 HDF5 转 LeRobot**：见 [README_SUBTASK_LEROBOT_ZH.md](README_SUBTASK_LEROBOT_ZH.md)。该版本按帧保存四类子任务，并让训练读取器使用对应指令。

本目录已克隆 StarVLA，并在其上增加完整任务适配。下面保留原 SFT 接入流程的记录；现在已有本地 `steps_9000` 权重，新增的 **RLinf GRPO 强化学习入口见 [README_GRPO_ZH.md](README_GRPO_ZH.md)**。不需要 PocketVLA 的模型或训练包；只有转换时需要原始 HDF5 数据。

上游固定版本：`2f17402a5ccaa09907516ae5e542b0fa6ee5d155`，分支 `starVLA_dev`。

**2026-09-15 起已切换为 StarVLA 官方训练框架**：`accelerate launch → starVLA/training/train_starvla.py → VLATrainer`。模型仍是 `QwenFast.py / Qwenvl_Fast`，前向使用离散 FAST token 交叉熵。此前自行编写的 `mujoco_task/train.py` 训练循环已停用。

任务适配只负责把现有 LeRobot 数据读成官方模型要求的样本，并通过官方 `build_dataloader` 的 `dataset_py=mujoco_lerobot` 分支接入；它使用原始关节绝对目标和固定的训练集归一化，未使用默认多机器人 loader 的相对动作变换。优化器、学习率计划、训练循环、日志与 checkpoint 由官方 `VLATrainer` 管理。

配置格式也随之改为官方格式。旧版 `trainer.epochs`、`task.output_dir`、`trainer.learning_rate=...` 不再用于训练，分别换成 `trainer.max_train_steps`、`run_root_dir + run_id`、`trainer.learning_rate.base`。现有 LeRobot 数据无需重新转换。

## 1. 本机转换现有 600 条训练数据

在本机终端执行：

```bash
cd /home/czj/Desktop/RLinf-Wan2/starvla_mujoco
bash scripts/mujoco/convert.sh
```

默认输入 `../pocketvla_mujoco/data/block_transfer2`，输出 `data/block_lerobot`。现有 manifest 是 **600 train / 20 val / 10 test**，不是只有 10 条非训练数据。脚本逐条校验 SHA256、形状、时间戳、成功标记、指令和 seed，拒绝跨 split 重复轨迹。输出目录已经存在时不会覆盖；失败的 `.partial` 目录需检查后手动删除或换一个输出目录。

指定其他路径：

```bash
bash scripts/mujoco/convert.sh /path/to/raw_hdf5 /path/to/block_lerobot
```

生成三个独立的 **LeRobot v2.1** 数据仓库，训练不会混入 val/test：

```text
data/block_lerobot/
  normalization.json                # 只用 train 计算，val/test/推理共用
  conversion.json                   # 原始 manifest 指纹和转换数量
  block_train/                      # block_val / block_test 同结构
    meta/info.json
    meta/episodes.jsonl
    meta/episodes_stats.jsonl
    meta/tasks.jsonl
    meta/modality.json
    meta/source.json                # 原始 split、seed、SHA256、仿真 contract
    data/chunk-000/episode_000000.parquet
    videos/chunk-000/observation.images.primary/episode_000000.mp4
```

| 字段 | 含义 |
|---|---|
| `observation.images.primary` | 原始 128×128 RGB、20 fps；无损 H.264 RGB 视频，PyAV 可解码 |
| `observation.state` | 21 维：q[6]、dq[6]、两指位置[2]、上次施加动作[7] |
| `action` | 7 维：六关节**绝对目标** rad + 公共手指目标 m；不是末端增量 |
| `timestamp` | 每步执行动作之前的观察时间，间隔 0.05 s |
| `task_index` | 固定任务文字索引 |

原始浮点动作不变，夹爪范围为 `[0, 0.035]` m。归一化仅在读取训练样本时进行：关节使用 train min/max，夹爪使用固定物理范围，映射至约 `[-1,1]`。val/test 超出范围不会被悄悄截断。每个 action chunk 长 8，只预测当前及未来动作；末尾重复最后一个目标，不跨 episode。`terminal_image` 没有对应动作，因此不当训练帧追加。`phase` 和方块真实位置只用于原始数据诊断，不进入模型输入。

无损视频采用 `gbrp`，一些浏览器不能直接播放；训练的 PyAV 和已检查的 LeRobot 0.3.3 `video_backend="pyav"` 可以读取。LeRobot v2.1 的官方读取器已实际验证。

### 在 Cursor 中预览出现偏色

如果 Cursor 的 MP4 预览颜色不对，先不要交换训练图像的 RGB/BGR 通道。已抽查完整转换数据 train/val/test 各一条轨迹、共 954 帧，PyAV 解码与 HDF5 原图逐像素一致，最大误差为 0。原视频是 `High 4:4:4 Predictive / gbrp`，编辑器预览的解码/显示路径可能不兼容。

生成普通播放器更容易支持的 H.264 YUV420/BT.601 预览：

```bash
# 本机 hdp 环境即可，需要系统 ffmpeg。
/home/czj/anaconda3/envs/hdp/bin/python -m mujoco_task.preview --split train --episode 0
# 输出 outputs/video_preview/train_episode_000000.mp4，可直接在 Cursor 打开。
# 如需导出整个 split：
/home/czj/anaconda3/envs/hdp/bin/python -m mujoco_task.preview --split train --all
```

预览脚本拒绝覆盖已有文件，可用 `--output outputs/video_preview_v2` 指定新目录。预览副本含有 YUV420 色度采样损失，仅用于观看；原始训练视频和元数据不变，不需要重新转换训练数据。若兼容副本在 Cursor 中仍偏色，应继续检查其播放器/硬件加速路径。

## 2. 本机检查数据和 FAST 离散化

转换可直接使用原 `hdp` 环境。模型接口和 tokenizer 检查使用独立环境，以免升级 Transformers 影响旧采集程序：

```bash
bash scripts/mujoco/setup_local.sh
source .venv-mujoco/bin/activate
export USE_TF=0 USE_FLAX=0 MUJOCO_GL=osmesa
python -m mujoco_task.download_models --fast-only
python -m mujoco_task.check_data
python -m mujoco_task.codec_check --data-root data/block_lerobot \
  --episodes 10 --replay --output outputs/codec_train10.json
```

`check_data` 不加载 VLM、不做优化。`codec_check` 只用小型 FAST tokenizer；先对所有选中帧的 8 步动作做编码/解码，然后回放原始动作和解码动作，分别测试 execute=1/8。任一回放失败会返回非零退出码，**应先排查再开始训练**。它不是策略成功率评估，因为动作来源仍是示范。

重要适配：原版 FAST 默认 `scale=10`。在当前两条 smoke 示范上，默认量化最大关节误差约 0.04636 rad；execute=1 的回放为 0/2 成功，execute=8 为 2/2，因此综合检查失败。配置改为 **`fast_scale=1000, fast_min_token=-4096`**，保留原 2048 项 BPE 词表和 `<robot_action_i>` 映射；不是更换损失函数，也没有拟合新 tokenizer。两条示范最大关节误差降至约 0.000690 rad，夹爪误差约 0.00001335 m，原始动作与两种解码回放都成功。

代价：当前最大约 **168 个动作 token**，默认精度约 13 个，因此高精度自回归推理更慢。`max_new_tokens=256` 包含终止 token 的余量。这些数字来自 2 条 smoke 轨迹，不代表完整 600 条均通过；上面的 10 条检查应针对你转换后的完整训练统计运行。任何修改 horizon、归一化或 codec 参数，都必须重新检查且重新训练，不能只改推理端。

## 3. 打包并 scp 到服务器

```bash
bash scripts/mujoco/package.sh
tar -czf ../block_lerobot.tar.gz -C data block_lerobot
scp ../starvla_mujoco_code.tar.gz ../block_lerobot.tar.gz USER@SERVER:/path/to/work/
```

代码包包含修改后的 StarVLA、任务脚本、测试、MuJoCo XML 和网格/许可证；排除 `.git`、虚拟环境、数据、模型权重和运行输出。**只重新 git clone 上游不会包含这里的适配代码，需传此目录或代码包。** 不依赖本机绝对路径和软链接。

服务器解包：

```bash
cd /path/to/work
tar -xzf starvla_mujoco_code.tar.gz
mkdir -p starvla_mujoco/data
tar -xzf block_lerobot.tar.gz -C starvla_mujoco/data
cd starvla_mujoco
```

## 4. 服务器安装与官方训练入口

使用独立 Python 3.10 环境。下面 CUDA 12.4 是安装示例，PyTorch wheel 应与服务器驱动匹配：

```bash
conda create -n starvla python=3.10 -y
conda activate starvla
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements-mujoco.txt
export USE_TF=0 USE_FLAX=0
python -m mujoco_task.download_models --fast-only
python -m mujoco_task.check_data
```

`train.sh` 已填入你提供的 `base_vlm=/home/chenzengjue/Qwen2.5-VL-3B-Instruct-Action`。FAST 是独立动作分词器，默认 `playground/Pretrained_models/fast`；该目录不存在时用上面的 `--fast-only` 下载。VLM 必须带 `<robot_action_0>` 到 `<robot_action_2047>`，不能直接使用普通 Qwen Instruct。若需下载 VLM 和 FAST 两者，运行 `python -m mujoco_task.download_models` 并修改脚本中的 VLM 路径。

现在的启动脚本沿用 `run_robocasa.sh` 风格，直接执行官方入口：

```bash
python -m accelerate.commands.launch \
  --config_file configs/mujoco/accelerate_ddp.yaml \
  --num_processes 2 \
  starVLA/training/train_starvla.py \
  --config_yaml configs/mujoco/qwenfast.yaml \
  --framework.name QwenFast \
  --datasets.vla_data.dataset_py mujoco_lerobot
```

通常直接改 `scripts/mujoco/train.sh` 顶部参数并启动即可。默认仅 GPU 0、1，两进程，DDP；没有启用 DeepSpeed。`STARVLA_DISABLE_DEEPSPEED=1` 是该官方版本已有的开关。DDP 每卡保留完整模型和优化器，不自动分摊显存；本配置是 FP32 参数/Adam 状态、bf16 前向，3B 模型需要数十 GB 显存，最终显存峰值需在服务器实测。

常用配置映射：

| 脚本变量 | 官方参数 | 默认值 |
|---|---|---|
| `base_vlm` | `framework.qwenvl.base_vlm` | 你提供的服务器路径 |
| `data_root_dir` | `datasets.vla_data.data_root_dir` | `data/block_lerobot` |
| `per_device_batch_size` | `datasets.vla_data.per_device_batch_size` | 1 |
| `gradient_accumulation_steps` | `trainer.gradient_accumulation_steps` | 16 |
| `max_train_steps` | `trainer.max_train_steps` | 20000 |
| `num_warmup_steps` | `trainer.num_warmup_steps` | 1000 |
| `save_interval` | `trainer.save_interval` | 1000 |
| `learning_rate` | `trainer.learning_rate.base` | 2e-5 |
| `freeze_module_list` | `trainer.freeze_modules` | Qwen 视觉编码器 |
| `run_root_dir` / `run_id` | 同名顶层参数 | `outputs/mujoco_qwenfast_native` |

两卡等效 batch = 1 × 16 × 2 = 32。600 条训练数据有 190800 帧，因此约 5963 个 optimizer step 遍历一次数据；20000 steps 约相当于 3.35 次遍历，不能再把它当作 20000 个 epoch。

```bash
# 小规模拟合诊断；你执行才会开始训练。
CUDA_VISIBLE_DEVICES=0,1 NPROC=2 bash scripts/mujoco/train.sh \
  --datasets.vla_data.max_train_episodes 2 \
  --trainer.max_train_steps 200 --trainer.num_warmup_steps 10 \
  --trainer.save_interval 100 --run_id mujoco_native_debug

# 正式启动，使用完整训练集：
CUDA_VISIBLE_DEVICES=0,1 NPROC=2 bash scripts/mujoco/train.sh

# 临时修改学习率、步数和输出名称：
bash scripts/mujoco/train.sh --trainer.learning_rate.base 1e-5 \
  --trainer.max_train_steps 10000 --run_id mujoco_native_lr1e5
```

参数优先级：YAML < 脚本变量 < 末尾追加参数。优先使用官方 `--key value` 写法；bash 入口兼容原先末尾追加的 `key=value`，会转为官方解析器可识别的形式。直接调用官方 Python 入口时使用 `--key value` 或 `--key=value`。

原版 `eval_action_model` 从**训练 loader** 取一个 batch 做动作误差检查，且初期 QwenFast 可能生成无效 token。因此默认 `trainer.eval_interval=0` 关闭这项诊断；独立 val/test 用下文的 `offline_eval` 和本机 MuJoCo 闭环测试。训练日志里的 `action_dit_loss` 是上游遗留指标名，在 QwenFast 中实际是动作 token CE，不是 diffusion loss，也不是成功率。

## 5. 日志、checkpoint 和恢复的变化

启动脚本自动通过 `tee` 保存完整终端输出，并保留退出码：

```text
logs/
  train_mujoco_qwenfast_native_时间_PID.log
  launch_时间_PID.sh
outputs/mujoco_qwenfast_native/
  config.full.yaml                  # 完整合并配置
  config.yaml                       # 上游访问过的配置项快照
  normalization.json                # 任务动作归一化和仿真 contract
  dataset_statistics.json
  summary.jsonl                     # 已保存 checkpoint 的 step
  checkpoints/
    steps_1000_pytorch_model.pt
    steps_2000_pytorch_model.pt
  final_model/
    pytorch_model.pt                # 训练正常结束后保存
```

默认每 1000 个 **optimizer step** 保存一次，已不再生成 `epoch_0001/`、`COMPLETE`、HF 权重目录或每 epoch 的 `metrics.jsonl`。`WANDB_MODE=disabled` 默认关闭 W&B；日志始终保存在 `logs/`。运行名称的命令行覆盖会影响实际 checkpoint 目录，日志文件名前缀取脚本中的 `run_id`；准确训练配置以 `config.full.yaml` 为准。为不同实验使用新的 `run_id`，避免覆盖旧结果。

**该上游版本的 checkpoint 仅保存模型 state dict，没有保存 optimizer/RNG/DataLoader 完整状态。** `trainer.is_resume=true` 会从当前 run 最新模型恢复、读回 step 并推进学习率计划，不是精确断点续训；不要套用之前独立 trainer 的恢复语义。

```bash
# 从当前 run 最新官方 checkpoint 继续（优化器重新初始化）：
bash scripts/mujoco/train.sh --trainer.is_resume true

# 从指定原生权重开始新实验，step 从 0 开始：
bash scripts/mujoco/train.sh --run_id mujoco_native_finetune \
  --trainer.pretrained_checkpoint /path/to/steps_1000_pytorch_model.pt
```

旧独立 trainer 的 `accelerator_state` 不能直接传给官方入口。旧 HF 导出目录仍可用于本机推理，但不应声称它能精确恢复到这套官方训练流程。

## 6. 将官方 checkpoint 传回本机测试

一并传递模型 state dict、`config.full.yaml` 和 `normalization.json`，保留目录关系：

```bash
# 服务器，按实际 step 修改：
tar -czf mujoco_native_step1000.tar.gz -C outputs mujoco_qwenfast_native/config.full.yaml \
  mujoco_qwenfast_native/normalization.json \
  mujoco_qwenfast_native/dataset_statistics.json \
  mujoco_qwenfast_native/checkpoints/steps_1000_pytorch_model.pt
scp mujoco_native_step1000.tar.gz LOCAL_USER@LOCAL_HOST:/path/to/starvla_mujoco/

# 本机：
mkdir -p outputs
tar -xzf mujoco_native_step1000.tar.gz -C outputs
source .venv-mujoco/bin/activate
export USE_TF=0 USE_FLAX=0 MUJOCO_GL=osmesa
```

本机需要对应的原始 Qwen2.5-VL-3B-Instruct-Action 目录及 FAST tokenizer，用于初始化模型结构/processor，然后加载官方训练权重。服务器绝对路径不同，通过 `--base-vlm` / `--fast-tokenizer` 覆盖。两个位置参数只改变文件来源，不改变图像、状态、动作归一化或 FAST 精度。

```bash
# 首次需要时，本机下载 VLM 与 FAST（此命令不会训练）：
python -m mujoco_task.download_models

# 独立 val 离线评估：报告 CE、无效生成率、有效动作的 RMSE。
python -m mujoco_task.offline_eval \
  --checkpoint outputs/mujoco_qwenfast_native/checkpoints/steps_1000_pytorch_model.pt \
  --base-vlm playground/Pretrained_models/Qwen2.5-VL-3B-Instruct-Action \
  --fast-tokenizer playground/Pretrained_models/fast \
  --data-root data/block_lerobot --split val --episodes 2 --stride 8 \
  --device cuda --output outputs/offline_val_native.json

# 先检查训练 seed 的本机闭环效果，再用 val 选模型，最后测 test。
python -m mujoco_task.evaluate \
  --checkpoint /home/czj/disk1/RLinf-Wan2/starvla_mujoco/mujoco_qwenfast_native/checkpoints/steps_9000_pytorch_model.pt \
  --base-vlm /home/czj/agent-vla-gpt/Qwen2.5-VL-3B-Instruct-Action \
  --fast-tokenizer /home/czj/disk1/RLinf-Wan2/starvla_mujoco/playground/Pretrained_models/fast \
  --data-root data/block_lerobot --split train --episodes 10 \
  --execute 8 --device cuda --video --output outputs/rollout_train_native2

CUDA_VISIBLE_DEVICES=0 \
USE_TF=0 USE_FLAX=0 STARVLA_DISABLE_DEEPSPEED=1 MUJOCO_GL=osmesa \
python -m mujoco_task.evaluate \
  --checkpoint /home/czj/disk1/RLinf-Wan2/starvla_mujoco/mujoco_qwenfast_native/checkpoints/steps_9000_pytorch_model.pt \
  --base-vlm /home/czj/agent-vla-gpt/Qwen2.5-VL-3B-Instruct-Action \
  --fast-tokenizer playground/Pretrained_models/fast \
  --device cuda --torch-dtype bfloat16 --decoding auto \
  --data-root data/block_lerobot --split train --episodes 10 \
  --execute 8 --max-steps 500 --video \
  --output outputs/sft_1

CUDA_VISIBLE_DEVICES=0 \
USE_TF=0 USE_FLAX=0 STARVLA_DISABLE_DEEPSPEED=1 MUJOCO_GL=osmesa \
python -m mujoco_task.evaluate \
  --checkpoint /home/czj/disk1/RLinf-Wan2/starvla_mujoco/mujoco_qwenfast_native/checkpoints/steps_9000_pytorch_model.pt    \
  --base-vlm /home/czj/agent-vla-gpt/Qwen2.5-VL-3B-Instruct-Action \
  --fast-tokenizer playground/Pretrained_models/fast \
  --device cuda --torch-dtype bfloat16 --decoding auto \
  --data-root data/block_lerobot --split train --episodes 10 --num-envs 10 \
  --execute 8 --max-steps 500 --video \
  --output outputs/grpo_8-16-eval10_train10_batch

python -m mujoco_task.evaluate   --checkpoint /home/czj/disk1/RLinf-Wan2/starvla_mujoco/mujoco_qwenfast_native/checkpoints/steps_9000_pytorch_model.pt   --base-vlm /home/czj/agent-vla-gpt/Qwen2.5-VL-3B-Instruct-Action   --fast-tokenizer /home/czj/disk1/RLinf-Wan2/starvla_mujoco/playground/Pretrained_models/fast   --data-root data/block_lerobot --split train --episodes 10   --execute 8 --device cuda --video --output outputs/rollout_train_native
```

`--execute 1` 与 `--execute 8` 可分别比较。`--split test --episodes 10` 使用现有 10 条 test 的初始 seed。`results.json` 报告任务成功率，逐 seed JSON 记录 token、实际动作和接触/抬升指标。成功判定仍包含双指真实接触、搬起绿块、放到右侧、红块不明显移动、白臂不动等。

本机此前 CUDA 检查遇到错误 804；代码未修改驱动。可用 `--device cpu` 做慢速推理，但 3B 模型需要较多内存。GPU 路径、训练显存和正式训练结果未在本机验证。MuJoCo 在推理期间暂停仿真时间，每个动作仍推进 0.05 s。策略不调用专家，也不使用方块真值修正动作。

## 7. 相对原版的修改范围与验证

- 官方 `train_starvla.py` 保留原训练架构。针对梯度累积做局部修正：从 YAML 设置真实累积步数；在 optimizer step 之后清零梯度；只在同步步裁剪梯度、记录/评估/保存；`eval_interval=0` 支持关闭原有训练 batch 诊断。
- 官方 `build_dataloader` 新增 `mujoco_lerobot` 分支，任务 adapter 只处理数据；QwenFast 工厂按需加载，避免引入无关模型的依赖。
- QwenFast 保留先前验证的状态提示、动作 token/EOS 掩码、高精度 FAST 与严格解码，并通过配置开启梯度检查点。
- 本机评估同时支持官方 `.pt`/`.safetensors` 和旧的 HF 导出目录。官方权重需要同一次 run 的归一化文件。
- SFT 的 `predict_action` 保持原有接口；新增 RL 入口 `mujoco_task.grpo` 使用独立随机采样、token log-prob、任务奖励及 RLinf GRPO 优势/损失，见 [GRPO 说明](README_GRPO_ZH.md)。

```bash
USE_TF=0 USE_FLAX=0 STARVLA_DISABLE_DEEPSPEED=1 python -m pytest -q tests/mujoco
```

检查包括：官方 trainer 的入口连线、数据 factory、梯度累积调用边界、checkpoint 命名/内容、真实 processor + 微型随机 Qwen 前向和原生权重重新加载。测试使用 mock 优化器或无梯度前向，没有执行 VLA 训练。LeRobot 官方读取和真实 processor 的可选测试需要对应本地依赖/小型 tokenizer 资产，缺失时会跳过。

来源：[StarVLA](https://github.com/starVLA/starVLA)、[FAST](https://huggingface.co/physical-intelligence/fast)、[Qwen 动作词表模型](https://huggingface.co/StarVLA/Qwen2.5-VL-3B-Instruct-Action)。机器人网格许可证见 `mujoco_task/scene/assets/ROBOSUITE_LICENSE`。
