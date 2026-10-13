# 真机推理：SFT 与 SFT + GRPO LoRA

入口：`python -m mujoco_task.real_robot`，以下命令都在 `starvla_mujoco` 目录执行。

- `serve`：在 GPU 机器上加载模型，通过 websocket 接收 `{image, state}`，返回一段物理关节目标。
- `run`：闭环控制。`--checkpoint` 在本机加载模型；`--server host:port` 改为请求 `serve`，此时不导入 torch 和模型代码，工控机只需 `numpy`、`pillow`、`websockets`、`msgpack`。
- `--robot sim` 用 MuJoCo 代替真机跑同一个循环，上真机前先用它确认部署链路正确。
- `--robot ur5` 通过 RTDE 控制真机左侧 UR5（`mujoco_task/ur5_robot.py`，连接方式与 `go_home.py` 相同），见下文“UR5 真机”。
- 其他机械臂：复制 `mujoco_task/robot_template.py`，实现 `observe/command/close`，再用 `--robot 模块:类名` 加载，`--robot-kwargs '{"robot_ip": "..."}'` 传入构造参数。

## 两种模型

`--checkpoint` 决定加载哪个模型：

| 模型 | `--checkpoint` |
|------|----------------|
| SFT | `mujoco_qwenfast_native/checkpoints/steps_9000_pytorch_model.pt` |
| SFT + LoRA（验证最优） | `outputs/mujoco_nb2_lr2e5/checkpoints/iteration_000075` |

LoRA 目录会读取 `grpo_metadata.json` 中的 SFT 路径，再叠加 `adapter/`；SFT 搬了位置就加 `--sft-checkpoint`。`--decoding auto` 和 `mujoco_task.evaluate` 一致：SFT 用原来的 greedy 解码，v2 LoRA 用保存的 FAST 语法约束和 `repetition_penalty=1.05`。两者对比时可以都指定 `--decoding constrained --repetition-penalty 1.05`。

## 有 state / 无 state 两种模型

state 只以文本形式拼在 prompt 里（`Robot state: joint_rad=[...]; ...`），由训练配置 `framework.use_state_prompt` 决定：

- 无 state 训练（`train.sh` 里 `use_state_prompt=false`）：`config.full.yaml` 已记录该设置，`serve` 自动只用图像 + 指令，不需要额外参数。
- `--no-state`：强制不用 state，与 `mujoco_task.evaluate --no-state` 相同。用于 `config.full.yaml` 没记录对、或做消融对比。

`serve` 启动时打印的元数据里 `use_state_prompt` 显示实际用法，客户端据此决定是否发送 state；无 state 时请求里只有图像。客户端仍然读取关节状态，用于安全过滤和日志。若客户端加了 `--no-state` 而服务端模型使用 state，会直接报错退出。

```bash
# 无 state 模型的服务端
USE_TF=0 USE_FLAX=0 STARVLA_DISABLE_DEEPSPEED=1 MUJOCO_GL=osmesa \
python -m mujoco_task.real_robot serve \
  --checkpoint /path/to/mujoco_qwenfast_nostate/checkpoints/steps_9000_pytorch_model.pt \
  --base-vlm /path/to/Qwen2.5-VL-3B-Instruct-Action \
  --fast-tokenizer playground/Pretrained_models/fast --port 10093 --no-state
```

客户端命令不变。

## 先在仿真里验证

```bash
USE_TF=0 USE_FLAX=0 STARVLA_DISABLE_DEEPSPEED=1 MUJOCO_GL=osmesa \
python -m mujoco_task.real_robot run \
  --checkpoint outputs/mujoco_nb2_lr2e5/checkpoints/iteration_000075 \
  --base-vlm /home/czj/agent-vla-gpt/Qwen2.5-VL-3B-Instruct-Action \
  --fast-tokenizer playground/Pretrained_models/fast \
  --robot sim --sim-seed 0 --log outputs/real_robot_check/lora_seed0.jsonl
```

结束时打印 `{"steps", "failure", "success"}`。

单元测试（不需要 GPU）：

```bash
USE_TF=0 USE_FLAX=0 STARVLA_DISABLE_DEEPSPEED=1 MUJOCO_GL=osmesa \
python -m pytest -q tests/mujoco/test_real_robot.py tests/mujoco/test_ur5_robot.py
```

## 两台机器：GPU 服务器 + 机器人工控机

```bash
# GPU 服务器
USE_TF=0 USE_FLAX=0 STARVLA_DISABLE_DEEPSPEED=1 MUJOCO_GL=osmesa \
python -m mujoco_task.real_robot serve \
  --checkpoint mujoco_qwenfast_native/checkpoints/steps_9000_pytorch_model.pt \
  --base-vlm /path/to/Qwen2.5-VL-3B-Instruct-Action \
  --fast-tokenizer playground/Pretrained_models/fast --port 10093

# 工控机（需要本仓库的 mujoco_task/ 和 deployment/，依赖见 requirements-real-robot.txt）
python -m mujoco_task.real_robot run --server <gpu_ip>:10093 \
  --robot ur5 --robot-kwargs '{"robot_ip": "192.168.1.101", "camera": "realsense"}' \
  --no-send --log logs/dry.jsonl
```

服务器连接成功后会返回元数据：指令、图像大小、动作范围、限速等，客户端用这些参数构建安全过滤器。

## UR5 真机

策略控制的是仿真里的左臂（黑色夹爪，基座绕世界 Y 轴 -45°），对应真机左侧 UR5 `192.168.1.101`。仿真已按 `real_raw2` 三条真机轨迹对齐（见下文“仿真与真机对齐”）：运动学是 UR5 CB，基座坐标系就是 UR 控制器的基座坐标系，所以模型输出的关节角直接通过 `servoJ` 下发，`joint_offset` 保持 0。右臂在训练中一直不动，运行时不要控制它。

工控机安装：

```bash
python -m pip install -r requirements-real-robot.txt
```

上机步骤（都在 `starvla_mujoco` 目录执行）：

```bash
# 1. 查看当前关节角、与训练起始姿态的差距、TCP 位姿（只读）
python -m mujoco_task.ur5_robot status --ip 192.168.1.101

# 2. moveJ 到训练起始姿态（scene.xml 关键帧 home 的左臂关节角 = go_home.py 的姿态，夹爪全开），会先询问确认
python -m mujoco_task.ur5_robot go-start --ip 192.168.1.101 --gripper dm --gripper-stroke-m 0.07

# 3. 保存一帧相机图像和 128×128 的模型输入，与仿真画面对比视角
python -m mujoco_task.ur5_robot snapshot --camera realsense --out logs/camera.png

# 4. 只预测不下发（只连 RTDE 接收接口，机器人不会动）
python -m mujoco_task.real_robot run --server <gpu_ip>:10093 --robot ur5 \
  --robot-kwargs '{"robot_ip": "192.168.1.101", "camera": "realsense", "gripper": "dm", "gripper_stroke_m": 0.07}' \
  --no-send --max-steps 40 --log logs/ur5_dry.jsonl

# 5. 每段动作手动确认，限速调小
python -m mujoco_task.real_robot run --server <gpu_ip>:10093 --robot ur5 \
  --robot-kwargs '{"robot_ip": "192.168.1.101", "camera": "realsense", "gripper": "dm", "gripper_stroke_m": 0.07}' \
  --confirm --max-joint-delta 0.03 --log logs/ur5_confirm.jsonl
```

`--robot-kwargs` 可用参数（`UR5Robot`）：

| 参数 | 默认 | 说明 |
|------|------|------|
| `robot_ip` | `192.168.1.101` | 左侧 UR5 |
| `camera` | `0` | `realsense` 或 `realsense:<序列号>` 用 pyrealsense2 读 RealSense 彩色流（深度不用）；其他值为 OpenCV 序号或设备路径，如 `/dev/video2` |
| `camera_width` / `camera_height` | `640` / `480` | 采集分辨率，之后由 `real_robot` 取中心正方形缩放到 128 |
| `rgb_profile` | 无 | `mujoco_task.rgb_to_sim` 生成的 `profile.json`，对每帧做固定颜色校正 |
| `gripper` | `none` | `dm` 通过 `../ur5_control` 控制 DM4310 CAN 夹爪（后台线程以 50 Hz 持续发 MIT 命令）；`robotiq` 通过 URCap 端口 63352 控制 Robotiq 2F；`none` 不控制夹爪（读数等于上次命令） |
| `gripper_serial` | `/dev/ttyACM0` | DM4310 的 USB-CAN 串口 |
| `gripper_stroke_m` | `0.085` | 夹爪全开（0 %）时的指尖距离，**需实测**；开度按电机角度线性换算，100 % 为 0。仿真每指 0–0.035 m，即总开度 0–0.07 m |
| `rtde_frequency` / `ur_cap_port` | `200` / `50002` | 与 `go_home.py` 相同 |
| `lookahead_time` / `gain` | `0.1` / `300` | `servoJ` 参数 |
| `max_tracking_error` | `0.3` | 目标与实测关节角差距超过此值（rad）立即 `servoStop` 并报错 |
| `start_tolerance` | `0.15` | 启动时离训练起始姿态超过此值（rad）拒绝运行；`check_start: false` 可跳过 |
| `joint_offset` | `[0,0,0,0,0,0]` | 真机关节角 = 仿真关节角 + 偏移。真机底座相对仿真绕自身轴转了 180° 时用 `[3.141593,0,0,0,0,0]`；`status/go-start --joint-offset` 要用同一组值 |

控制方式：每个控制周期调用一次 `servoJ(q, t=1/control_hz)`；模型推理期间不发送新命令，机械臂停在上一个目标。每次观测和下发前都检查急停、保护性停止和 RTDE 脚本是否在运行，任何异常或 Ctrl+C 都会 `servoStop` 并断开连接。

注意：

- 训练时每步关节最多变化 0.12 rad，20 Hz 下约 2.4 rad/s，对真机偏快。先用 `--max-joint-delta 0.03` 之类的小值，确认动作方向正确后再逐步放开。
- 夹爪位置与开度按线性换算，没有标定；`gripper: none` 时只能验证手臂运动，无法完成抓取。
- `gripper: dm` 要求目录结构为 `<上层>/ur5_control/` 与 `<上层>/starvla_mujoco/mujoco_task/`（与本仓库一致），会自动把上层目录加入 `sys.path`。启动时电机会先失能 1 秒再使能，夹爪会短暂松开。
- 第一次运行前用 `status` 确认关节角没有多转一圈（例如 wrist_3 差 2π），否则 `start_tolerance` 会拦下。
- `go-start` 发现任一关节要转超过 `--max-move`（默认 1.5 rad）会拒绝执行：moveJ 所有关节同时插值，大幅转动可能横扫整个工作空间。先用示教器点动到附近，确认路径无障碍时才加 `--allow-large-move`。

## 仿真与真机对齐（`real_raw2`）

用三条真机轨迹（RGB + 对齐深度 + 关节角）和 `real_robot.md` 里的测量值标定，改动集中在 `scene/scene.xml`、`scene/layout.json`、`env.py`、`scene/block_task.py`：

| 项目 | 现在的仿真 | 依据 |
|------|------------|------|
| 机械臂 | UR5 CB 连杆参数，基座坐标系 = UR 控制器基座系（原来差绕 z 180°） | 同一组关节角下仿真法兰与控制器 TCP 读数差 ≤ 1.3 mm |
| 起始姿态 | 左臂 = `go_home.py` 姿态 `[-1.0004, -1.4391, -2.3295, -1.441, 0.932, 0.738]`，夹爪全开；右臂为其镜像 | 三条轨迹的第一帧 |
| 相机 | `policy_cam`：标定出的 D435I 位姿，桌面上方 22.5 cm，fovy 42.74°，对应 640×480 的中心 480×480 | 机械臂渲染深度与真实深度匹配 + 桌面水平 |
| 夹爪 | 按 DM 夹爪尺寸建模，TCP 在法兰下 0.159 m，两指间距 = 2 × 手指位置（0–0.07 m） | 抓住方块时方块中心相对法兰的位置 |
| 桌子 | 远桌面在基座下 0.2295 m、前沿 y = -0.825 m、左沿 x = -0.45 m；近桌（相机所在）高 4 mm，中间有黑缝 | 深度点云；“红块右沿距左沿 42 cm”与之吻合 |
| 立柱 / 右臂 | 立柱中心在左基座右侧 0.17 m，右基座在 0.34 m；去掉了真机没有的相机立杆 | 深度点云 |
| 方块 | 绿 `(-0.43, 0.065)`、红 `(-0.308, 0.065)`（世界系），间距 12.2 cm；重置时整体 x ±6 cm、y ±2 cm，各自 ±5 mm | 三条轨迹的方块位置（绿块 x 范围 10 cm） |
| 任务 | 绿块放到红块右侧 9.5 cm，抬升 8 cm，姿态保持起始姿态的工具朝向 | 真机演示：放置点距红块 8.4–10.3 cm，抬升 7–9 cm |

检查对齐（在有 MuJoCo 的机器上，无显示器时加 `MUJOCO_GL=egl`）：

```bash
# 真机帧 | 同关节角的仿真画面 | 叠加
python -m mujoco_task.real_overlay --episode data/real_raw/episode_0000 --frames 0 300 600 --out logs/overlay.png
# 按真机轨迹的抓取点摆放方块，跑仿真专家，存成与 ur5_record.py 相同的格式；--episodes N 另加 N 条随机重置
python -m mujoco_task.sim_vs_real collect --real data/real_raw/episode_0000 data/real_raw/episode_0001 --out data/sim_raw
# 按 开始/抓取/松开/结束 对齐两条轨迹：side_by_side.mp4、keyframes.png、trajectory.png、summary.json
python -m mujoco_task.sim_vs_real compare --real data/real_raw/episode_0000 --sim data/sim_raw/episode_0000 \
  --out logs/compare_0000
```

对齐后旧数据和旧权重都不再适用（相机、起始姿态、运动学都变了），需要重新生成仿真数据并重新训练。

## 键盘遥操与采集（`ur5_record.py`）

用于采集真机数据（标定帧和演示轨迹）以对齐仿真。控制方式沿用 `ur5_control/keyboard_ur5_teleop.py`：pynput 全局读键（任意窗口有焦点都生效，需要 X11 桌面会话，SSH 下不可用），按住即动、松开即停。每个周期的 `servoL` 目标 = 实测 TCP + EMA 平滑后的步长，目标不会越跑越远。

```bash
python -m mujoco_task.ur5_record --ip 192.168.1.101 --gripper dm --gripper-stroke-m 0.07 --out data/real_raw
```

| 键 | 作用 |
|----|------|
| W/S、A/D、Q/E | 世界 X、Y、Z 正/负（`go_home.py` 的世界系，倾角按 IP 自动取 -45°/+45°，可用 `--world-tilt-deg` 覆盖） |
| U/O、I/K、J/L | 绕 TCP 自身轴 roll、pitch、yaw |
| 空格、`[`/`]` | 夹爪全开↔闭合；每次 1 % 更闭合/更张开 |
| R、X | 开始/停止并保存一条轨迹；丢弃当前轨迹 |
| P | 保存一帧标定数据（RGB、深度、关节角、TCP），按之前让机械臂静止 |
| ESC | 退出（正在录的轨迹会保存） |

- 默认 20 Hz（与训练频率一致），平移 0.15 m/s、旋转 0.24 rad/s，与原 teleop 在 30 Hz 下的速度相同；用 `--speed`、`--turn-speed` 调整。
- 奇异保护：按 UR5 DH 参数计算雅可比条件数（常规姿态约 8–14），≥ 25 提示，≥ 35 时只发送能让条件数变小的运动，可以反向退出，不会被锁死。
- TCP 读数无效（位置模长 ≤ 1 cm）时跳过该帧运动指令。
- 预览窗口右半边是彩色深度图（近红远蓝，无数据为黑，`--depth-view-max-m` 调色标范围）。`depth.png` 是 16 位原始深度（乘 `meta.json` 里的 `depth_scale_m` 得到米），普通看图软件打开几乎全黑是正常的；标定帧另存一张 `depth_vis.png` 供人查看。
- 每条轨迹保存为 `episode_XXXX/{data.npz, meta.json, rgb/, depth/}`，各字段含义写在 `meta.json` 的 `fields` 中，数据均为 UR 控制器约定（不加 `joint_offset`）。

## 输入输出约定

必须和仿真训练一致，否则不会报错，只会悄悄失败：

- 图像：一路固定相机，RGB `uint8`。先取中心正方形，再缩放到 128×128；`--no-center-crop` 改为整幅缩放。相机需保持 `real_raw2` 采集时的安装位置（仿真 `policy_cam` 就是按它标定的）；挪动过相机要重新采标定数据。
- 状态 21 维：关节角 6（rad），关节速度 6（rad/s），两指开度 2（m，每指 0–0.035），上一次下发的命令 7。上一次命令由脚本记录；开始时取当前姿态和两指均值。
- 关节顺序：`shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3`，零位和正方向要与仿真模型一致。
- 输出：每次 8 步绝对目标（6 个关节 rad + 1 个共同手指目标 m），默认执行 `--execute 8` 步，以 20 Hz 下发。推理期间机器人保持上一个目标；仿真中推理时物理是暂停的，这一点与真机不同。

## 安全措施

- 每步关节变化不超过 0.12 rad（与仿真相同，可用 `--max-joint-delta` 调小）。
- 关节目标限制在训练集动作范围 ± `--range-margin`（默认 0.05 rad），手指限制在 0–0.035 m。仿真本身只有 ±2π 的关节限位，对真机没有保护作用。
- 模型输出非法 FAST 时立即停止循环，不再下发新目标。
- 建议上真机的顺序：先用 `--no-send` 只预测、不下发，检查日志里的 `predicted` 是否合理；再用 `--confirm` 每段动作手动确认；最后才连续运行。急停按钮始终由人控制。

## 限制

模型只用 MuJoCo 渲染图像训练，没有真实图像数据。直接迁移到真机预计有明显的视觉差距（光照、纹理、相机内参），成功率无法用仿真结果推断。`ur5_robot.py` 覆盖 UR5 + DM4310 / Robotiq 2F 夹爪 + OpenCV / RealSense 相机；其他夹爪或相机需要在其中添加对应驱动。
