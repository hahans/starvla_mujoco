# 带四类子任务标签的 HDF5 → LeRobot v2.1

本数据用于训练接收子任务指令的底层动作策略。Planner 的图文 SFT 数据仍是 LlamaFactory ShareGPT，不能用动作 Parquet 直接替代。

## 本次输出

- 已确认 HDF5：`outputs/planner_verified/confirmed/hdf5/`。
- 对应 LeRobot：`data/block_lerobot_subtasks/block_train/`。
- 归一化：`data/block_lerobot_subtasks/normalization.json`。
- 已确认 Planner 图文数据：`outputs/planner_verified/confirmed/dataset_llamafactory/`，登记名 `planner_train`。

这四份输出对应 **600 条 train 轨迹、190,800 个动作帧**；图文 Planner 默认抽样后为 22,800 条样本。
原始 HDF5、原总任务版 `data/block_lerobot/` 和先前的候选标注均保留。
本次只复核了 600 条 train，未将原来的 20 条 val、10 条 test 冒充为已确认子任务数据，也没有随机拆帧构造验证集。

## 转换命令

```bash
cd /home/czj/disk1/RLinf-Wan2/starvla_mujoco
/home/czj/anaconda3/envs/hdp/bin/python -m mujoco_task.convert \
  --source outputs/planner_verified/confirmed/hdf5 \
  --output data/block_lerobot_subtasks \
  --task-mode subtask
```

输出目录必须不存在。已生成后重跑请更换 `--output`。
快速检查可加 `--limit-per-split 2`，并使用单独目录。
默认 `--task-mode auto` 会检测 HDF5 的 `planner` 分组；推荐显式传 `subtask`，缺少标签时立即报错。
显式 `--task-mode global` 才会按总任务转换。
转换器支持只有 train 的带标签 manifest；归一化仍只使用 train。

## 字段映射与任务语义

| 带标签 HDF5 | LeRobot |
|---|---|
| `image[t]` | `observation.images.primary` 无损 RGB 视频的第 t 帧 |
| `proprio[t]` | `observation.state`，21 维 |
| `action[t]` | `action`，7 维绝对关节/夹爪目标 |
| `time[t]` | `timestamp` |
| `planner/subtask_index[t]` | **`task_index`，按帧切换，不再固定为 0** |
| `planner/subtask_boundary[t]` | `subtask_boundary`，每个子任务第一帧为 true，包括整条轨迹第一帧 |
| `planner/subgoal[t]` | `meta/tasks.jsonl` 中对应 `task_index` 的文本 |
| `planner/subtask_id[t]` | `meta/subtasks.json` 中对应 `task_index` 的固定 ID |
| HDF5 总任务 instruction | `meta/subtasks.json` 的 `global_task`，以及 `meta/source.json` 的 contract |

固定映射：

| task_index | subtask_id | 语义 | 本次所有轨迹对应帧（左闭右开） |
|---|---|---|---|
| 0 | approach | 黑爪接近绿块 | `[0, 105)` |
| 1 | grasp_lift | 抓取绿块并抬起 | `[105, 161)` |
| 2 | transfer | 搬至红块右侧上方 | `[161, 205)` |
| 3 | place_release | 放下、松爪、撤离并等待稳定 | `[205, 318)` |

转换按 HDF5 标签读取，不把上表的帧号硬编码进转换器。
整条轨迹保留为一个 episode，原图像、动作、状态和时间戳不变；`terminal_image` 没有对应动作，不追加成训练帧。
阶段 ID/文本、帧数、顺序、边界标志、复核状态及文件 SHA256 不一致时拒绝转换。

StarVLA 的本地 `TaskDataset` 已适配每帧 `task_index`，不会把所有帧错误地配成第一条子任务。
它在动作 chunk 遇到下一个子任务时重复当前子任务的最后一个目标，避免在当前指令下监督下一子任务的动作；写入 LeRobot 的原始动作并未修改。
若训练子任务策略，推理时也需由 planner 提供相同的子目标文本。

## 为什么可以确认此前的候选标签

`outputs/planner_verified/verification_summary.json` 记录了对全部 600 条轨迹的确定性物理回放：

- 每帧动作、机器人状态、时间、专家阶段及停止位置与原始记录对齐。
- 190,800 帧全部通过，最大动作误差和状态误差均为 0。
- 接近结束时夹爪到达绿块且张开；抓取抬升结束及搬运结束时存在双指接触、方块离桌；搬运末端达到目标上方。
- 最终方块已松开、落桌并稳定，红块和白色机械臂的移动量符合原采集成功标准，无异常接触和仿真警告。
- DeepSeek 给出的四类切换帧与回放验证的专家阶段一致。

确认的是“从该帧观察开始，示范执行哪个子任务”，依据是原始专家阶段加物理回放，**不是声称视觉模型从单张低分辨率图片一定能判断完成条件**。
原 Teacher 的置信度及 `needs_review` 保存在确认标注的 `teacher_annotation` 和 HDF5 属性中。
新的 `confidence: 1.0` 是确定性检查通过标记（`confidence_kind` 已明确），不是把 API 原概率改成 100%。

HDF5 副本新增 `planner/` 分组；副本 manifest 中的 SHA256 重新计算，并用 `source_sha256` 保存原文件指纹。
确认后的 LlamaFactory 数据没有待复核或缺失轨迹，使用 `planner_train`，不再使用旧目录下的 `planner_review_train`。

## 重做复核与生成

当代码或来源变化时，请使用新输出目录；旧复核报告绑定实现和源 manifest 指纹，原执行时的验证器快照保存在 `outputs/planner_verified/verify_planner_source.py`。

```bash
# 六个 CPU 进程，无 GPU、无额外 API 请求；只读回放原始 HDF5。
/home/czj/anaconda3/envs/hdp/bin/python -m mujoco_task.verify_planner \
  --output outputs/planner_verified_v2 --workers 6

# 全部回放通过后，生成带标签 HDF5 和确认后的 LlamaFactory 数据。
/home/czj/anaconda3/envs/hdp/bin/python -m mujoco_task.publish_verified_planner \
  --verification outputs/planner_verified_v2 \
  --destination outputs/planner_verified_v2/confirmed
```

测试：

```bash
/home/czj/anaconda3/envs/hdp/bin/python -m pytest -q \
  tests/test_planner_verification.py tests/test_planner_annotations.py tests/mujoco/test_data.py
```
