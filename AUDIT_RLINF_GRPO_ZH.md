# 本地 Qwen FAST / RLinf GRPO 审计

审计对象是用户指定的 `../Rlinf4/Rlinf4/rlinf` 工作区，而不是另一份 RLinf 或网上版本。该工作区包含 HRC/subtask 修改。本次没有改动它的训练代码；修复发生在 `mujoco_task` 接入层。加载 SFT 后再注入 LoRA 的顺序保持不变。

## 核心计算的对照结果

| 环节 | 指定 RLinf 实现 | 当前 Qwen FAST 接入 | 结论 |
|---|---|---|---|
| 组内优势 | `algorithms/advantages.py::compute_grpo_advantages`，组均值、样本标准差、`1e-6` | 直接调用该函数的 task advantage 输出 | 没发现公式或分组错误。变长 episode 与 registry 的 embodied 路径数值一致 |
| 同初态分组 | LIBERO `reset_state_ids.repeat(group_size)` | 同 seed 重置独立 MuJoCo 实例，策略 RNG 不重置 | 语义相同；跨环境批次也保留组顺序，有回归测试 |
| 概率粒度 | `preprocess_loss_inputs` 可按 token/action/chunk 重排固定维数动作 | 使用实际生成的变长 FAST token，自回归 next-token 概率 | 有意适配。FAST 是整个 chunk 的 DCT/BPE 编码，不能直接把 token `reshape(-1, 7)` 当成 7 维动作 |
| PPO ratio / clip | `exp(new_logp-old_logp)`，最大化两项负优势损失 | 直接调用 `compute_grpo_actor_loss_fn`，最终进入相同 `compute_ppo_actor_loss` | 公式、正负优势及梯度对照通过；不是自己另写了一个 GRPO |
| 采样/回算数值偏差 | reasoning actor 在同时有 rollout 与重算 log-prob 时启用 `importance_sampling_fix`：PPO 分母仍是 rollout，优势乘 `clamp(exp(重算−采样), min=importance_sampling_clip)`，不因差值中止 | v2 调用同一约定，`importance_sampling_clip=1.0` | 中后段实测存在 BF16 差异。差异继续记入指标，更新不再因超过 0.5 退出 |
| 长度归一化 | 有 `loss_mask_sum` 时，`masked_mean_ratio` 校正 episode 有效步数 | 每 episode 的全部有效 FAST token 求均值，再对 episode 平均 | 固定每步 token 数的情形与 RLinf 数值及梯度一致。变长 FAST 按 token 数加权，不等同逐 chunk 等权，是应记录的目标差异 |
| 优化步数 | actor 用独立 Generator 打乱 rollout，按 `global_batch_size=16384` 个 chunk-step 做一次 Adam；`micro_batch_size=32` 只是一次前向的条数。LIBERO OpenVLA-OFT 学习率 `2e-5`，`kl_beta=0`，clip 下限 0.2 | 当前 buffer 128 条，`minibatch_size=32`，打乱后 4 步 Adam。学习率 `2e-5`，`kl_beta=0`，`clip_ratio=0.2`。变长 FAST 仍按轨迹内 token 均值、再对 minibatch 内轨迹平均；一次前向仍是一个 chunk | 步数按轨迹计，不是按 chunk-step。学习率和 KL 与该 LIBERO 示例对齐 |
| 采集规模 | OpenVLA-OFT 的 LIBERO-Object 配置：`group_size=8`、`num_group_envs=8`、`rollout_epoch=16`，buffer 1024。每一轮结束调用 `update_reset_state_ids()` 换下一批初态；同组用 `repeat(group_size)` 共用初态。优势按 `group_size` 分组 | v2：`groups_per_iteration=8`、`group_size=8`、`rollout_epoch=2`，buffer 128。每一轮重新抽组 seed，同组 8 条共用 | 并行初态和组大小与该示例相同，采集轮数是 2 而不是 16。场景库仍是 seed 0–20，有放回抽样。没有照搬 `reset()` 不传 id 时按环境独立随机、从而拆开同组初态的那条路径 |
| SFT reference / KL | embodied 示例一般 `kl_beta=0`；公共 `kl_penalty` 提供有界 k3，reasoning worker 可用 reference | v2 现为 `kl_beta=0`。LoRA 仍可关闭 adapter 得到冻结 SFT；full 用独立冻结副本 | 与 embodied 示例一样不加 KL。零优势且 `kl_beta=0` 时不调用 Adam |
| 奖励 | 由具体环境提供，示例可按组奖励过滤 | 旧配置给抓取/抬升/接近目标额外奖励；v2 改为任务成功 sparse reward | 任务目标差异。部分进展不等同成功；新奖励是否更好仍需训练验证 |
| 分布式执行 | Ray + rollout/actor workers + FSDP | 单进程、单卡、多个环境批量 VLA 推理 | 有意的最小接入。不是原生 RLinf Runner；A100 full 也不自动获得 FSDP/offload |

对照测试位于 `tests/mujoco/test_grpo_rlinf_parity.py`，会执行本地真实 RLinf 的 `calculate_adv_and_returns` / `policy_loss`，而不只与一份重复公式比较。`tests/mujoco/test_grpo_batch.py` 覆盖组顺序、padding、EOS 后截断、跨批采集和 chunk 内渲染。

## 已确认的问题及修复

### 1. 训练和原 SFT 评估使用不同的重复惩罚

原 SFT greedy 评估继承基础模型 `generation_config.json` 的 `repetition_penalty=1.05`；旧 GRPO 生成显式设成 `1.0`。这是行为策略与部署解码的差异。旧训练内部的采样/回算都为 1.0，不能把它描述成已经证实的 PPO old/new 概率错配。

v2 显式用 1.05，并在 teacher forcing 的每个 causal prefix 上复现 Hugging Face 重复惩罚，然后再做温度和 FAST support 处理。单测直接对比 `RepetitionPenaltyLogitsProcessor`，包括正/负 logits、prompt 中已出现的动作 token 和未来 token 不得泄露的情形。

之前试加格式约束时直接把重复惩罚 1.0 的新解码设为默认，破坏了历史基线；已撤回。`evaluate --decoding auto` 对旧 SFT/LoRA 使用历史方式，对有 v2 配置的 checkpoint 恢复训练解码参数。

### 2. 零优势时错误地一起跳过 KL

旧 `update_actor` 在整批优势为 0 时立即返回。这会同时丢掉启用的 SFT KL，尤其是 sparse reward 下整组失败时。现在只有“无策略梯度且无有效 KL 梯度”才不更新。回归测试构造偏离 SFT 的策略、零优势 batch，验证更新后 KL 减小。

**这不是旧 40 轮下降的已证实原因**：已有 42 轮日志全部 `updated=true`，没有整批零优势。该修复防止新配置触发问题，不能拿它解释未发生的历史情况。

### 3. 无 KL 数值保护、缺少 PPO 诊断

旧 k3 直接 `exp(ref_logp-logp)`，极端差异可以溢出；已有日志没有证明发生过溢出。现使用与 RLinf 一致的差值界限 `[-20,20]`、KL 上限 10，并用 `expm1` 减少接近零时的相消误差。

旧代码丢弃 RLinf 返回的 ratio/clip 指标，难以发现策略是否真的进入裁剪区间。现在记录这些指标及更新前的采样/回算最大误差；超出配置阈值则在 optimizer step 前报错。GPU 审计同时检查已训练 adapter、不同图像和不同 prompt 长度，避免只验证同一张图重复组成的 batch。

### 4. FAST 非法长度和缺少固定验证选择

旧训练前 40 轮的 160 条轨迹中，38 条因非法 FAST 提前终止（23.75%），主要是 55/57/58 个系数而不是 56。新增字节级 UTF-8 DFA 同时约束生成和损失回算。该约束会改变行为分布，因此不能只在评估时打开就宣称权重提升。

旧训练没有固定验证集选择，直接取第 40 轮不能代表最优。v2 保存 SFT baseline、定期比较固定 val seeds、记录最佳 checkpoint，并早停。相同成绩保留原最佳；未超过 SFT 时不把 LoRA 当作最佳模型。

### 5. BF16 回算误差可能造成更新前的裁剪

初始状态的 GPU 一致性很好，但将历史 SFT 评估的已执行动作重放到第 0/80/160/240 控制步，再采样实际 iteration 40 adapter，得到缓存生成与逐条完整前向之间最大 log-prob 误差 `0.13336`、平均 `0.001886`。约 0.45% token 的 ratio 在没有参数更新时已超出 v2 的 10% 区间。这与仅看初始状态得出的判断不同，不能继续声称所有状态都只有 `1e-4` 误差。

旧配置的 clip 区间是 20%，本次样本的最大误差还不足以越过这个旧阈值，而且审计使用的是新解码规则。因此这证明了需要补充数值误差处理，不能直接证明旧 50%→30% 就是由提前裁剪造成的。

v2 在首个 update epoch、任何 optimizer step 之前固定回算 log-prob。PPO 分母保持原始采样 log-prob。优势乘以与 RLinf actor worker 相同的无梯度权重 `clamp(exp(回算 − 采样), min=importance_sampling_clip)`，示例与本配置的下界都是 1.0，没有上限，也不再因为绝对误差超过 0.5 中止更新。因此首轮 PPO ratio 仍包含 BF16 缓存前向和完整前向的差异，可能进入 clip；更新会继续。尚未测定这个数值差异对历史成功率的贡献。

## 已有实验与解释边界

所有下列 train 对照均只作诊断，不用于选最终模型。历史两次报告使用 seeds 0–9、execute 8、max_steps 500，但旧 SFT 报告未记录加载精度，不能视为严格受控实验。

| 实验 | SFT | iteration 40 LoRA | 非法 FAST | 说明 |
|---|---:|---:|---|---|
| 历史评估，原解码 | 5/10 | 3/10 | LoRA 一条非法长度 | 原用户报告对应文件 |
| 同条件 BF16、10 环境、长度约束、重复惩罚 1.0 | 1/10 | 4/10 | 两者 0 | `outputs/grpo_v2_train10_comparison/summary.json`；说明新解码也可能损害 SFT |
| 同条件 BF16、10 环境、长度约束、重复惩罚 1.05 | 4/10 | 3/10 | 两者 0 | `outputs/grpo_v2_train10_rep105_comparison/summary.json`；LoRA 并未超过同条件 SFT |

重复惩罚 1.05 的后续对照已完成，单独保存，不覆盖旧结果。仅修正解码没有让第 40 轮超过 SFT，因此不能把旧权重直接称为改进后的模型。

十个 episode 很少，而且不同 batch/精度下的数值变化可能在闭环中累积。上述结果能证明解码设置对任务有明显影响；不能证明所有退化只由它导致，也不能证明重新训练后的 v2 已提升成功率。应使用同条件 SFT baseline、固定 val 模型选择，再独立报告 test。完整新训练尚未完成。

2026-09-25 的真实 3090 GPU 审计：v2 四环境 batch 完成采样、LoRA 反传、reference 不变及保存重载，采样/回算最大误差 `3.4332e-5`，测试峰值 allocated 显存约 7.62 GiB。再加载实际 iteration 40 adapter，用四张不同初态图像、不同 prompt 长度采样，回算最大误差 `9.9182e-5`、平均误差 `3.9403e-7`。这些初始观察的检查没有显示显著 PPO 概率错配；不等于已验证所有长轨迹状态。对应 opt-in GPU 测试两项通过。

完整测试套件 63 项通过、3 项 opt-in/资产相关测试按条件跳过。v2 已完成两环境 × 8 控制步的训练入口短测：先验证 SFT、采样、零梯度跳过更新、固定验证、保存成功；best 保留 SFT。从该 checkpoint 恢复后执行 iteration 2，训练 seed 从 4000000 继续到 4000001，固定 val seeds 和最佳记录保留，stale checks 从 1 增至 2。最初结果在 `outputs/grpo_v2_audit_smoke`、`outputs/grpo_v2_audit_resumed`；增加旧概率重算后的最终复测在 `outputs/grpo_v2_corrected_smoke`、`outputs/grpo_v2_corrected_resumed`。8 步不是完整任务评估，不能用其成功率判断训练效果。

## 下一步执行

使用 [README_GRPO_V2_ZH.md](README_GRPO_V2_ZH.md) 中的修正配置，从 `steps_9000` 新开实验。先看 `replay_logprob_abs_max`、`clip_fraction`、`kl`、组内有效优势比例和固定 val 结果，再决定是否增加采集场景数、update epochs 或优化 minibatch。不要在尚未对齐解码和验证基线时继续放大学习率或全参数更新。
