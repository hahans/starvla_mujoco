# 用 DeepSeek 标注四类 HRC Planner 子任务

**2026-10-04 更新**：600 条候选轨迹已通过完整物理回放复核，确认后的 HDF5、LlamaFactory 和保留子任务的 LeRobot 路径见 [README_SUBTASK_LEROBOT_ZH.md](README_SUBTASK_LEROBOT_ZH.md)。下文关于旧候选目录的统计作为历史记录保留。

入口：`mujoco_task/annotate_planner.py`。直接读取现有原始 HDF5，不需要 GPU、MuJoCo 渲染、VLA 权重或重新转换 LeRobot 数据。

## 固定四类

| ID | 给底层策略的子目标 | 采集阶段候选映射 |
|---|---|---|
| `approach` | 黑爪接近绿块 | prepare / approach / descend |
| `grasp_lift` | 抓住绿块并抬起 | grasp / lift |
| `transfer` | 把持有的绿块搬到红块右侧 | transfer |
| `place_release` | 放下绿块、松爪并撤离，等待稳定 | place / release / retreat / settle |

精确英文输出与解释在 `configs/mujoco/planner_subtasks.json`。英文沿用当前任务指令的语言，避免同一子任务出现多种自由表述。配置支持 3 或 4 类，每个原始 phase 必须且只能映射到一类；修改分类后使用新的输出目录。

原始数据不是完全没有标注：HDF5 的 `phase` 是采集控制器当帧要执行的动作阶段，且 `image[t]` 是执行 `action[t]` **之前**的观察。它适合提供边界候选，但不能直接证明物体已抓稳或子任务已成功完成。

脚本先合并 phase，再将按时间采样的 RGB 图像和候选边界交给 DeepSeek 核验。每条轨迹正常只请求一次，返回四个连续区间 `[start, end)`，允许修正切换边界。默认每 12 帧抽一张，另外补充候选边界附近和阶段中点的帧；不发送仿真坐标、动作、成功标记。模型只能选择四个 ID，不能增加细阶段。

## API 配置

截至 2026-10-03，[DeepSeek 官方 V4.1 Flash 发布说明](https://api-docs.deepseek.com/news/news260910/)指定模型名为 `deepseek-flash`，官方地址为 `https://api.deepseek.com`。[视觉接口](https://api-docs.deepseek.com/guides/vision/)接受 Chat Completions 的 `image_url` base64 输入。脚本用 PNG 保留原始 128×128 RGB，无需 OpenAI SDK，关闭 thinking，要求 JSON 输出。

依赖：Python 3.10+、`numpy h5py Pillow requests`；测试额外需要 `pytest`。本机的 `/home/czj/anaconda3/envs/hdp/bin/python` 已具备这些依赖。脚本的默认路径根据自身位置解析，不依赖历史文档里的 Desktop 路径。

```bash
cd /home/czj/disk1/RLinf-Wan2/starvla_mujoco
PYTHON=/home/czj/anaconda3/envs/hdp/bin/python

# 仅检查文件哈希、阶段、元数据和时间戳，零 API 调用。
"$PYTHON" -m mujoco_task.annotate_planner prepare

# 在终端隐藏输入密钥；不会写进配置、数据集或命令历史。
read -rsp 'DeepSeek API key: ' DEEPSEEK_API_KEY
export DEEPSEEK_API_KEY

# 先标注 3 条检查效果。它们会直接保存在完整任务的输出目录中。
"$PYTHON" -m mujoco_task.annotate_planner annotate --limit 3

# 继续全部 600 条 train，自动跳过已有合法结果。
"$PYTHON" -m mujoco_task.annotate_planner annotate

# 离线生成当前帧 → 当前子目标的训练集。
"$PYTHON" -m mujoco_task.annotate_planner export
```

如需独立验证、测试集，统一使用 `--splits train val test`：

```bash
"$PYTHON" -m mujoco_task.annotate_planner annotate --splits train val test
"$PYTHON" -m mujoco_task.annotate_planner export --splits train val test \
  --export-dir outputs/planner_annotations/dataset_all_splits
```

原 manifest 为 **600 train / 20 val / 10 test**。脚本保持轨迹所属 split，不随机拆帧、不将同一轨迹的相邻帧混入验证集。`--limit` 是所选轨迹总数上限，按 manifest 顺序选取，不是每个 split 的上限。默认只选 train；要训练时验证，务必另外导出 val。

可以指定 `--source`、`--output`、`--taxonomy`、`--base-url`、`--model`、`--api-key-env`。默认每分钟最多启动 30 次请求、读取超时 120 秒；超时、408、429、5xx 最多尝试 4 次，包含初次调用。401、402、403、错误模型或端点等配置问题立即停止。发生网络超时时，服务端可能已经执行并计费；重试无法保证只计费一次。

每条轨迹一轮请求不等于一条训练样本。默认每 8 帧导出一个当前图像样本，并排除切换边界前后 3 帧（含边界）。更密集的标注图像可用 `--image-stride 6`，但这会增加视觉 token 数并要求新输出目录。输出记录实际 API `usage`；预检只报告图片数和初次请求数，不把不确定的费用估计当作账单。

## 质量门槛与断点恢复

- 强制检查：四类顺序、完整覆盖、无重叠、整数帧边界、合法 ID、有限的 `[0,1]` 置信度、布尔复核标记。非法输出或截断响应不生成标签，也不回退到伪造的“API 通过”标签。
- 默认任一段置信度低于 `0.8`、模型标记 `needs_review`，或边界偏移超过 `8` 帧，**整条轨迹**不进入训练集，防止其不可靠历史污染后续样本。置信度是模型自报值，不是经过校准的准确率。
- `annotate` 有失败或待复核轨迹时返回退出码 1；`export` 仍导出通过的轨迹，并在有缺失或待复核时返回 1。查看报告后再决定是否使用部分数据。
- 原始响应先落盘，成功解析后单独保存结构化标注；重复运行跳过有效标注，崩溃恢复会复用已保存响应。源 manifest、分类、模型、端点、抽帧设置或提示词不匹配时拒绝混用缓存。
- 要对单条重新付费请求，先检查并移走它的 `annotations/<split>/seed_*.json` 和 `responses/<split>/seed_*.json`，然后重跑。若只有非法原始响应，移走对应 response 即可；其他成功结果不会重复请求。不要把测试 mock 响应拷进正式输出目录。
- 同一输出目录禁止多个进程同时操作。导出到 `.partial` 暂存目录后整体重命名；已有导出不覆盖。若导出中断，检查 `.partial` 后移走，或换 `--export-dir`。不同输出目录也应选择不同导出目录。

```text
outputs/planner_annotations/
  run.json                         # 模型、配置、提示词与源 manifest 指纹
  prepared.json                    # 原始候选区间、抽帧下标和全量检查报告
  responses/train/seed_0000000.json # 原始 API 响应；含 token usage
  annotations/train/seed_0000000.json
  annotation_summary.json          # 本次成功、缓存、待复核数量和失败轨迹
  dataset/
    train.jsonl                    # 只有通过门槛的轨迹
    images/train/seed_0000000/*.png
    taxonomy.json
    report.json                    # 导出数量、缺失轨迹和逐条复核理由
```

## 与 HRC 的对应关系

### LlamaFactory / Qwen2.5-VL 导出

使用现有 API 标注直接导出，无需密钥或重新请求 DeepSeek：

```bash
cd /home/czj/disk1/RLinf-Wan2/starvla_mujoco
/home/czj/anaconda3/envs/hdp/bin/python -m mujoco_task.annotate_planner export \
  --export-format llamafactory
```

默认新目录为 `outputs/planner_annotations/dataset_llamafactory/`；原 `dataset/` 保留。
再次导出时使用 `--export-dir` 指定新的目录。
格式遵循 [LlamaFactory Multimodal Image Dataset](https://github.com/hiyouga/LlamaFactory/blob/main/data/README.md#multimodal-image-dataset-1)：

```json
[
  {
    "conversations": [
      {"from": "human", "value": "<image>\nTask: ...\nPrevious subgoals: [...]\nAllowed subgoals: [...]\n..."},
      {"from": "gpt", "value": "{\"subgoal\": \"move the black gripper to the green block\"}"}
    ],
    "images": ["images/train/seed_0000000/000000.png"]
  }
]
```

每个 JSON 文件是一个数组，每个样本恰好有一个 `<image>` 和一张 RGB 图片。
`images` 使用相对于数据集根目录的路径，完整复制目录即可迁移到其他机器。
该格式用于 **Qwen2.5-VL**，纯文本 Qwen2.5 不支持这些图像输入。

自动生成的 `dataset_info.json` 只注册非空数据：

- `planner_train` → `train.json`：通过原有质量筛选的训练样本。
- `planner_review_train` → `review_train.json`：待复核的候选训练样本。
- 显式导出 `--splits train val test` 时，同样分别生成对应的 val/test 文件。

两类文件互斥，不应为了增加样本数把同一份文件重复注册使用。
LlamaFactory 的 `formatting` 为 `sharegpt`，列映射为 `messages: conversations`、`images: images`，角色标签为 `from/value`、`human/gpt`。
原标签、当前帧、历史及四类子目标保持不变。
复核原因、置信度、源轨迹和样本行号在独立的 `samples_metadata.jsonl` 中，不进入训练对话。

**本次数据现状（2026-10-03）**：600 条轨迹均已得到 API 返回，但全部含待复核片段，因此旧 `train.jsonl` 是空文件。
新导出得到 22,800 条候选样本，位于 `review_train.json`；`train.json` 为空，未注册 `planner_train`。
这次只转换格式，没有降低质量阈值，也没有把待复核状态改成已通过。
退出码 1 表示仍有待复核轨迹；是否成功写入请同时查看日志和 `report.json`。

如果决定使用这份候选标签先训练，在现有 LlamaFactory SFT YAML 中设置以下字段：

```yaml
model_name_or_path: Qwen/Qwen2.5-VL-7B-Instruct  # 或自己的 Qwen2.5-VL 权重路径
template: qwen2_vl
dataset_dir: /home/czj/disk1/RLinf-Wan2/starvla_mujoco/outputs/planner_annotations/dataset_llamafactory
media_dir: /home/czj/disk1/RLinf-Wan2/starvla_mujoco/outputs/planner_annotations/dataset_llamafactory
dataset: planner_review_train
val_size: 0.0
```

其余 LoRA、batch size、学习率和输出目录沿用自己的训练设置。
当前只有 train 轨迹，不能把相邻帧随机拆成训练和验证数据。若需要评估，应先标注原来的 val 轨迹，再独立导出并配置 `eval_dataset`。
若后续导出有通过质量筛选的 `planner_train`，可改为该名称。
WebUI 中同样先设置数据路径，再选择登记的数据集名和 `qwen2_vl` 模板。
`dataset_info.json` 已放在数据目录内，不需要覆盖 LlamaFactory 自带的注册文件。

### Planner 输入输出

参考 `../Rlinf4/Rlinf4/rlinf/models/embodiment/vlm_planner.py` 的 `plan(image, task_description, subgoal_history)` 和 `hrc_model.py` 中的子目标切换历史。导出 JSONL 包含：

```json
{
  "images": ["images/train/seed_0000000/000112.png"],
  "messages": [
    {"role": "user", "content": "<image>\nTask: ...\nPrevious subgoals: [...]\nAllowed subgoals: [...]\n..."},
    {"role": "assistant", "content": "{\"subgoal\": \"grasp and lift the green block with the black gripper\"}"}
  ],
  "task_description": "use the black gripper to move the green block to the right of the red block",
  "subgoal_history": ["move the black gripper to the green block"],
  "subgoal": "grasp and lift the green block with the black gripper",
  "subtask_id": "grasp_lift",
  "source": {"split": "train", "seed": 0, "frame_index": 112}
}
```

示例字段作结构说明，具体帧的标签以实际 API 核验结果为准。`images` 相对于该 JSONL 所在目录；训练加载器需只把图像、`messages` 用作模型输入。`source` 只供追溯，不应加入提示词。

Teacher 标注时可以回看整条轨迹，**student 输入始终只有当前 RGB、总任务和此前已切换的子目标**，不包含未来图像、边界时间、phase 或 Teacher 推理理由。相同子目标跨多个帧重复是有意设计，planner 应持续选择当前子目标直到视觉条件支持切换。

这次提供的是标注和数据导出，不是 VLM 训练入口，也没有改动现有 HRC 运行代码：

- JSON 的 `subgoal` 字段可被现有 `VLMPlanner.extract_subgoal` 解析。
- 当前 HRC `vlm_server.py` 仍使用要求坐标的旧提示词，且虽然构建了 `history_text`，实际没有拼入 prompt。部署新 planner 时应改用本模块 `planner_prompt()` 的文本和当前图像，并传入 HRC 的历史；否则会产生训练/推理差异。
- 本数据只监督子任务选择，不标注目标框、目标坐标和完成/失败检测。依赖这些坐标的 HRC 奖励不能直接据此训练。
- 现有 600 条均为成功示范且任务顺序固定，尚不能覆盖掉块、抓空、绕障碍或返回前一阶段的恢复规划。API 全轨迹审阅也不能消除 128×128 当前帧本身存在的视觉歧义；先检查边界样本和待复核轨迹。
- 如果底层 VLA 只用整句总任务训练过，还需要验证它是否能理解这些四类子目标；仅训练 planner 不保证分层执行立即有效。

## 离线测试

```bash
"$PYTHON" -m pytest -q tests/test_planner_annotations.py
"$PYTHON" -m ruff check mujoco_task/annotate_planner.py tests/test_planner_annotations.py
```

测试使用临时合成数据和 mock API，覆盖图像帧对齐、原始 split 保留、无未来历史、边界排除、低置信度过滤、非法响应拒绝、断点恢复、鉴权停止和限流重试。它不代表真实 API 标注质量已验证。
