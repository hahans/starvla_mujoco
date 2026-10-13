# 真机 RGB 外观预处理

入口：`python -m mujoco_task.rgb_to_sim`。只依赖已有 NumPy、Pillow，无需 OpenCV、MuJoCo、GPU 或模型权重。

此工具做**传统图像预处理**：固定白平衡/曝光、Gamma、饱和度、轻度双边滤波，以及可选的参考图颜色匹配。它不会重建 OSMesa 渲染，不保证策略成功率。默认只启用 25% 强度的保边平滑，默认不改变颜色、尺寸和视野。

## 单张图像

激活现有环境后，在项目根目录执行（替换真实输入路径）：

```bash
python -m mujoco_task.rgb_to_sim \
  --input /path/to/real_rgb.png \
  --output outputs/real_rgb_processed
```

输出目录必须不存在。原图不修改，输出：

- `images/real_rgb.png.png`：转换结果，RGB 无损 PNG。
- `comparisons/real_rgb.png.png`：左边是裁剪/缩放后的原图，右边是处理结果。
- `profile.json`：固定参数，可供下一批图像和实时相机复用。
- `manifest.json`：输入路径、尺寸、输出位置和图像变化量。变化量不是转换质量或抓取性能指标。

保留输入扩展名是为了避免同目录下 `frame.jpg` 与 `frame.png` 相互覆盖。

## 使用 OSMesa 参考图

```bash
python -m mujoco_task.rgb_to_sim \
  --input /path/to/real_rgb.png \
  --reference outputs/cluster_audit_20260926/local_osmesa/seed_1.png \
  --strength 0.35 \
  --smooth 0.25 \
  --output outputs/real_rgb_matched
```

这是可选的**全局颜色统计近似**：根据每个 RGB 通道的 10%、50%、90% 分位数，估计有界的对比度和偏移，再按 `strength` 混合。不是像素配准，也不保证保留准确的色度学颜色。

参考图应尽量有相似的视角、背景和物体占比。上述现成参考图只适合作为本任务场景示例；如果真实相机视野差别很大，先不用自动参考匹配，改为手动调参。物体占比差异会被统计匹配误认为颜色差异，尤其需要检查红绿方块是否偏色。

参数只在第一张输入图上估计一次。处理文件夹时，第一张是按路径排序的第一张图片；建议先用有代表性的单帧标定，再复用 profile。将参考图放在输入文件夹之外，避免当作待处理图像。

## 手动调整与几何对齐

```bash
python -m mujoco_task.rgb_to_sim \
  --input /path/to/real_rgb.png \
  --wb-gains 1.02 1.00 0.98 \
  --exposure 0.1 --gamma 1.0 --saturation 0.95 \
  --smooth 0.25 \
  --output outputs/real_rgb_manual
```

这些是演示数值，不是已经标定好的真机参数。

| 参数 | 默认 | 含义 |
|---|---|---|
| `--wb-gains R G B` | `1 1 1` | 在线性光 RGB 中乘以固定通道增益 |
| `--exposure` | `0` | 曝光补偿 EV，正值提亮 |
| `--gamma` | `1` | `output = input ** (1/gamma)`，大于 1 提亮 |
| `--saturation` | `1` | 相对灰度的颜色强度；0 为灰度 |
| `--smooth` | `0.25` | 原图与 5×5 RGB 双边滤波的混合强度，0 关闭 |
| `--strength` | `0.35` | 仅在使用参考图时控制匹配强度 |
| `--crop X Y W H` | 不裁剪 | 指定真实像素坐标区域，不猜测视野 |
| `--size W H` | 不缩放 | 保持长宽比缩放，比例不符时报错 |

例如输入是 640×480，且你已确认中央 480×480 区域包含完整任务，可以加：

```text
--crop 80 0 480 480 --size 128 128
```

不要直接把整张 16:9 画面拉伸到 128×128。裁剪必须根据相机视野决定，不能保证上述示例坐标适合你的相机。对比图左侧已经应用裁剪/缩放，请另外保留原图检查是否漏掉夹爪。

处理顺序是：EXIF 方向 → 显式裁剪/缩放 → 白平衡/曝光 → Gamma/饱和度 → 固定参考颜色校正 → 保边平滑。脚本不自动去畸变、移除阴影、替换背景或改变三维视角；这些操作需要额外标定或分割。

## 批量处理和实时复用

```bash
python -m mujoco_task.rgb_to_sim \
  --input /path/to/rgb_frames \
  --profile outputs/real_rgb_matched/profile.json \
  --output outputs/rgb_frames_processed
```

递归处理常见图像格式，保持相对子目录。输入应为 8 位 sRGB 图像；不处理视频文件、Bayer RAW、深度图或透明图像。带非 sRGB ICC 配置的图片需先转换到 sRGB；脚本不自动解释 ICC。输出目录应在输入目录之外。中途读到不支持的图片会报错并留下已生成结果，重跑请换输出目录。

实时相机可直接调用：

```python
from pathlib import Path
from mujoco_task.rgb_to_sim import convert_rgb, load_profile

profile = load_profile(Path("outputs/real_rgb_matched/profile.json"))
# rgb_frame 必须是 H×W×3、uint8、RGB。OpenCV 的 BGR 帧需先转换通道顺序。
processed_rgb = convert_rgb(rgb_frame, profile)
# 再按现有 VLA 的图像输入方式传入 processed_rgb。
```

每帧复用同一 profile，避免自动颜色统计随物体移动而跳变。相机自身的自动曝光/白平衡仍可能引入变化，采集与评估时应保持设置一致。脚本不会修改已有评估或控制流程。

## 验证范围

测试覆盖 RGB 通道顺序、恒等变换、噪声抑制与边缘保留、已知颜色偏移、裁剪比例、参数有效性、profile 固定复用和原文件不变。尚无用户真机图像，所以未标定真实相机参数，也未验证转换后的 VLA 成功率。

本机还用现有 EGL seed 1 图像实际跑通了参考匹配和对比图输出，产物在 `outputs/rgb_to_sim_demo_20260928/`。这组默认参数的 OSMesa 参考像素 MAE 从 **2.4593 上升到 2.5250**（8 位 RGB），没有改善该示例的像素一致性。它验证的是脚本能够运行，不是“转换已修复 EGL”的证据；全局颜色和滤波不能恢复驱动的采样信息。真机效果仍需用真实输入标定与验证。

本机一次 640×480 处理约 0.15 秒，仅为单次 CPU 测量，不保证实时帧率。只有在明确对齐视野后，才能通过 `--crop` / `--size` 降低处理分辨率。
