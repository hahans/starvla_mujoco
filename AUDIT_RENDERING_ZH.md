# EGL / OSMesa 差异排查（本机，2026-09-27）

## 新 EGL SFT 仍失败时的检查（2026-09-30）

用户提供的新模型 train seed 0 结果：执行满 500 步、`failure=null`、未形成双侧抓取、未满足抬起判定，并有夹爪碰桌接触。这不是之前 FAST 系数数量错误导致的提前终止。仅凭该条结果无法确定是渲染不一致、模仿拟合不足还是闭环误差累积；也无法计算 10 条总体成功率。

新增 `mujoco_task.audit_observations`，不加载 VLA，直接在**评估所用的 TransferEnv** 回放数据集动作，比较数据集保存的 RGB/21 维状态与实时观测。默认检查第 0、8、48 和末帧，记录实际 GL vendor/renderer/version、MSAA，以及专家动作回放是否成功。不会修改数据、训练参数或默认渲染设置。

在新模块已同步到服务器后，用与评估相同的环境变量执行：

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl CUDA_VISIBLE_DEVICES=5 MUJOCO_EGL_DEVICE_ID=5 \
python -m mujoco_task.audit_observations \
  --data-root data/block_lerobot_egl --split train --episodes 2 \
  --output outputs/audit_egl_observations
```

输出 `report.json` 和并排图（左为数据集、右为实时回放）。如果状态一致但图像不同，继续查采集/评估渲染；如果专家回放失败或状态不同，查动作/仿真接口；如果检查都通过，则把重点转向模型在保存的专家观测上的预测质量。抽查一致只覆盖这些状态，不等于所有状态都一致。

接着可以使用已有的 `mujoco_task.offline_eval`：从新 checkpoint 对应的 config 和 normalization 加载模型，在**保存的训练图像**上测 token CE、生成非法率和关节/夹爪误差。这一步不进行实时渲染，能够排除评估时渲染导致的输入变化。离线误差小也不能保证闭环成功，需要结合失败视频和逐 chunk 动作分析。

本机工具验证：旧 OSMesa 数据 train seed 0 在 OSMesa 下四个检查帧 RGB/状态完全一致、专家回放成功；相同数据在 NVIDIA EGL 下能检出图像差异，而状态仍一致、专家回放成功。3 项数值单元测试通过。未连接服务器，尚未拿到新 EGL 数据/权重进行本机验证。

## 结论

当前 SFT 的视觉输入依赖 Mesa 的渲染结果。**EGL 接口本身没有导致失败：本机 EGL + Mesa llvmpipe 能与 OSMesa + 同版本 llvmpipe 生成逐像素一致的图像。** 默认 EGL 选中的 NVIDIA 驱动与 Mesa 的输出不同，改变了 VLA 的动作。相同 seed 只约束初始状态，不约束 OpenGL 驱动和最终像素。

本轮所有新增实验都在本机 RTX 3090 完成。未继续在集群测试、安装依赖或修改服务器。此前保存的远程输出仅用于离线比较。

## 直接证据

- `steps_9000_pytorch_model.pt` 此前核对本机/集群 SHA256 相同，关键推理、环境、FAST 代码一致。
- 固定 seed、相机、关节状态并回放相同动作，OSMesa 与 NVIDIA EGL 的 `qpos` 完全一致，图像不同。
- 本机训练集 seed 1 首帧与 OSMesa 重渲染逐像素一致；与 NVIDIA EGL 的 RGB 通道平均绝对误差为 **2.45929 / 255**。
- 固定本机模型和状态，仅替换渲染图像，第 8、48 步的动作 token 改变；文本 input IDs、attention mask、图像网格一致，预处理后的像素张量不同。
- 本机 NVIDIA EGL 与此前保存的 A100 EGL 四个状态图像，MAE 仅约 **0.00022–0.00028 / 255**。主要差异可在本机复现。

## 抗锯齿是一个明确的不一致，但不是全部原因

`mujoco_task/env.py` 请求 `model.vis.quality.offsamples = 2`。读取实际 framebuffer 的采样数，得到：

| 请求 samples | Mesa llvmpipe 实际 samples | NVIDIA 实际 samples |
|---|---:|---:|
| 0 | 0 | 0 |
| 1 | 4 | 2 |
| 2（原配置） | 4 | 2 |
| 4 | 4 | 4 |
| 8 | 4 | 8 |

驱动会选择其支持的采样数，不能把请求值当作实际值。seed 1 初始画面对比：

| 实验 | 两后端图像 MAE（8 位 RGB） |
|---|---:|
| 原配置请求 2 samples | 2.45929 |
| 都请求 4 samples | 0.97473 |
| 4 samples，双方关闭阴影 | 0.91018 |
| 4 samples，双方关闭 dithering | 0.97473 |
| 4 samples，双方移除纹理 | 0.83571 |
| 4 samples，双方改为纯环境光 | 0.10077 |
| 4 samples，纯环境光且移除纹理 | 0.02960 |

这些消融显示：采样数差异解释了很大一部分误差，剩余差异主要与光照及纹理渲染有关。尚未定位到驱动内部的具体算术实现；不能声称其中某个驱动渲染“错误”。关闭光照会改变 SFT 输入分布，因此没有把这些消融配置作为正式修复。

## 闭环与完整回放验证

固定原 SFT、float32 权重加载（模型内部仍按原逻辑 autocast BF16）、legacy greedy、execute=8、seed=1：

| 本机实验 | 结果 |
|---|---|
| NVIDIA EGL，原 2 samples 请求 | 第 40 步 FAST 解码失败，未抓起方块 |
| NVIDIA EGL，改为 4 samples 请求 | 跑满 500 步，抓起过方块，但任务失败 |

因此“把 2 改成 4”不能视为已修复。提前停止的具体原因是 `Malformed FAST chunk: 58 coefficients; expected 56`。每个 chunk 是 8 步 × 7 维，应有 56 个系数；不是 48 步上限，也不是 MuJoCo 提前结束。此前服务器在第 48 步发生同类错误。

然后把历史 OSMesa 成功 seed 1 的 **370 个动作**分别在本机 OSMesa 和 **EGL + Mesa llvmpipe** 回放：

- **371 帧 RGB、proprio、qpos、qvel 的逐帧 SHA256 全部相同**。
- 最终任务指标完全相同，均判定成功。
- 这是完整固定动作回放验证，不是重新运行 VLA 的 10 条闭环成功率测试。新 OSMesa 模型闭环对照因用户另外两个评估进程占用显存而 OOM，未中断这些进程，也未改成较低精度替代对照。

## 本机启动：EGL 接口 + Mesa 渲染

新增 `scripts/mujoco/evaluate_mesa_egl.sh`：使用已有 Mesa EGL / llvmpipe，不加载 OSMesa 接口，不安装依赖；模型仍使用 CUDA。脚本先创建真实渲染上下文，确认 `GL_RENDERER` 是 llvmpipe，再加载模型。找不到 Mesa 时明确失败，不静默回退到 NVIDIA。

```bash
cd /home/czj/disk1/RLinf-Wan2/starvla_mujoco
PYTHON_BIN=/home/czj/anaconda3/envs/starvla/bin/python \
CUDA_VISIBLE_DEVICES=0 \
bash scripts/mujoco/evaluate_mesa_egl.sh \
  --checkpoint mujoco_qwenfast_native/checkpoints/steps_9000_pytorch_model.pt \
  --base-vlm /home/czj/agent-vla-gpt/Qwen2.5-VL-3B-Instruct-Action \
  --fast-tokenizer playground/Pretrained_models/fast \
  --data-root data/block_lerobot --split train --episodes 10 \
  --execute 8 --device cuda --torch-dtype float32 \
  --video --output outputs/rollout_train_mesa_egl
```

输出目录必须未存在。float32 权重约占 14 GiB，先等其他模型评估释放足够显存。不要为比较渲染后端同时更换权重精度或解码模式。

脚本的关键环境变量为：

```bash
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json
export LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe
unset MUJOCO_EGL_DEVICE_ID
```

仅 `MUJOCO_GL=egl` 或仅 `LIBGL_ALWAYS_SOFTWARE=1` 不足以表达“必须选择 Mesa”。脚本保留 `CUDA_VISIBLE_DEVICES`；渲染使用 CPU，VLA 使用 GPU。它不提供 NVIDIA GPU 渲染加速。不同机器的 Mesa/LLVM 版本也可能影响结果；尚未验证集群是否具备 Mesa EGL/llvmpipe，不能保证上述方案在那里直接可用。

## 代码层面遗漏与后续训练

当前 `env.contract()` 只校验 MuJoCo 版本、场景、相机、控制器等，未包含实际 GL vendor/renderer/version 和 framebuffer samples，所以两种图像不同的环境都能通过校验。后续采集、评估、GRPO 应记录这些渲染指纹，并以训练集固定帧检查像素一致性；仅比较场景 hash 不够。

本次没有改变正式环境的渲染默认值、checkpoint 或 GRPO 参数。若必须使用 NVIDIA GPU 渲染，应该先统一实际 samples，再用该渲染器补充 SFT 数据或做视觉增强，并在一致的渲染器上比较 SFT/GRPO。FAST 约束解码能避免非法系数数量，但不能修复视觉分布变化，也不能保证抓取成功。

## 复现产物

均在 `outputs/cluster_audit_20260926/`：

- `render_probe.py`、`render_ablation.py`、`render_materials.py`：固定状态及消融实验。
- `local_osmesa/`、`local_egl/`、`local_egl_mesa/`：图像、状态、GL 指纹。
- `ablation_{osmesa,egl}/`、`materials_{osmesa,egl}/`：采样、材质消融。
- `local_policy.json`：同一本机模型在不同图像下的输入指纹和动作 token。
- `local_egl_rollout/`、`local_egl_msaa4_rollout/`：本次两个实际 VLA 闭环结果。
- `replay_fingerprint.py`、`full_replay_osmesa.json`、`full_replay_mesa_egl.json`：完整 371 帧对照。

启动器已实测创建 llvmpipe EGL 上下文并进入 `evaluate --help`，另有轻量测试覆盖参数透传、保留 CUDA 选择、预检失败停止和缺少 Mesa 的情况。
