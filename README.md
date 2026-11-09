# Qwen3.5 数学推理后训练：SFT 与 On-Policy Distillation

在单卡资源约束下，以 Qwen3.5-0.8B 为学生、Qwen3.5-4B 为教师，比较教师轨迹 SFT、直接 OPD，以及 SFT 后继续 OPD 对数学推理的影响。所有模型统一采用 **non-thinking** 模式；测试题和配置固定后再做比较。

![测试正确率](figures/test_accuracy.png)

## 结果

| 组别 | 初始化与训练 | GSM8K（200） | MATH-500（200） |
|---|---|---:|---:|
| Base | 官方 Qwen3.5-0.8B | 45.0% | 38.0% |
| SFT | Base → 教师解答 SFT | 51.5% | 39.5% |
| Direct OPD | Base → OPD | **54.5%** | 36.0% |
| SFT → OPD | SFT → OPD | **54.5%** | 37.5% |
| OPD 1024 | Base → OPD，生成长度上限 1024 | 52.5% | 37.5% |

GSM8K 上，直接 OPD 相对 Base 提高 **9.5 个百分点**；基于题目的配对 bootstrap 95% 区间为 **[2.0, 17.0] 个百分点**。SFT 后追加 OPD 相对 SFT 为 **+3.0 个百分点，区间 [-4.0, 10.0]**：本次结果不足以确认追加 OPD 带来稳定收益。MATH-500 没有观察到明确提升，且各组 **38.5%–46.5%** 的生成未在 2048 token 内结束，解释时需要谨慎。

区间只反映这 200 道题的抽样不确定性；每组仅一次训练和一次采样评测，不覆盖训练随机性或多次生成波动。详细数值、验证集选点和局限见 [实验报告](docs/实验报告.md)。

![截断率](figures/test_truncation.png)

## 实验设计

```text
GSM8K 训练题 500 ── Qwen3.5-4B 生成解答 ── 答案验证 ── 456 条 ── 长度筛选 ── 455 条 SFT
       │                                                      │
       └── 学生在线采样，教师对学生生成序列给出分布指导 ── OPD ─┴── SFT → OPD

GSM8K 验证题 100：选择检查点；GSM8K 测试题 200 + MATH-500 测试题 200：最终比较
```

- 教师生成只在训练题上进行；最终答案通过 `math-verify` 与标准答案比较。答案正确不保证每一步推理均正确。
- 使用 TRL `DistillationTrainer`，学生自行生成轨迹、教师提供指导；LoRA rank 16、alpha 32，作用于 q/v 投影。主要 OPD 轨迹上限 512，另做 1024 消融。
- SFT 比较两个学习率及多个检查点，采用验证集最高准确率选点；两条 OPD 路线各训练 250 步，从 50、125、250 步中分别选点。1024 消融沿用直接 OPD 选出的步数。
- 测试均按同一提示词、采样参数和最大输出 2048 token 运行；答案解析失败计错。未结束生成单独统计为截断，但如果已能解析正确答案仍可计对。

## 数据和文件

| 位置 | 内容 |
|---|---|
| `data/v2/train.jsonl` | GSM8K 训练题 500 条，用于教师生成及 OPD 提示词 |
| `data/v2/teacher_raw.jsonl` / `teacher_sft.jsonl` | 教师原始生成 500 条 / 最终答案验证通过的 456 条 |
| `data/v2/val.jsonl` | GSM8K 验证题 100 条，用于选择超参数和检查点 |
| `data/v2/gsm8k_test.jsonl`、`math500_test.jsonl` | 两个固定测试子集，各 200 条 |
| `data/v2/manifest.json` | 数据源 revision、抽样种子、文件哈希和去重说明 |
| `scripts/` | 教师生成、训练、评测、结果汇总与绘图脚本 |
| `adapters/` | 四个实际测试使用的 LoRA 权重 |
| `results/v2/` | 逐题预测、验证集输出、汇总表和环境快照 |
| `figures/` | 可由逐题结果重新生成的实验图 |

数据来自 [GSM8K](https://huggingface.co/datasets/openai/gsm8k) 与 [MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500)。项目检查了训练题与验证/测试题的规范化精确重复；没有完成近重复检查，也无法排除基础模型预训练污染。

## 复现

训练时使用单张 48 GB vGPU、Python 3.12、PyTorch 2.8.0+cu128。先按 PyTorch 官方方法安装匹配 CUDA 的 PyTorch，再安装 `requirements.txt`；精确环境见 `results/v2/environment-freeze.txt`。模型默认从 Hugging Face 加载，可通过环境变量指定本地目录；模型版本写在训练和评测脚本中。

```bash
pip install -r requirements.txt
export STUDENT_MODEL=Qwen/Qwen3.5-0.8B
export TEACHER_MODEL=Qwen/Qwen3.5-4B

# 教师生成并验证训练答案；覆盖已保存的数据，请先备份
python scripts/make_teacher_sft.py
# SFT 两个候选；对检查点做验证集评测后选出低学习率第 25 步
python scripts/train.py sft --train-data data/v2/teacher_sft.jsonl --output runs/sft-low --steps 125 --save-steps 25 --lr 0.00005
python scripts/train.py sft --train-data data/v2/teacher_sft.jsonl --output runs/sft-high --steps 125 --save-steps 25 --lr 0.0002
# 主实验和长度消融
python scripts/train.py opd --train-data data/v2/train.jsonl --output runs/direct-opd --steps 250 --save-steps 25 --max-completion 512
python scripts/train.py opd --train-data data/v2/train.jsonl --start-adapter runs/sft-low/checkpoint-25 --output runs/sft-opd --steps 250 --save-steps 25 --max-completion 512
python scripts/train.py opd --train-data data/v2/train.jsonl --output runs/direct-opd-1024 --steps 250 --save-steps 25 --max-completion 1024

# 示例：固定评测设置；其他组替换 adapter 和数据路径，Base 省略 --adapter
python scripts/evaluate.py --adapter adapters/direct-opd --data data/v2/gsm8k_test.jsonl --output runs/direct-opd-gsm8k-test.jsonl --max-new-tokens 2048 --batch-size 8
# 下面两条直接读取仓库内的已保存结果，不需要 GPU
python scripts/report_v2.py
python scripts/plot_results.py
```

检查点评估使用相同脚本，SFT 验证最大生成 1024 token、OPD 验证为 2048 token，因此两阶段验证分数不能直接横向比较；最终测试均为 2048。采样生成受批次、硬件和软件版本影响，重跑数值可能变化。

## 参考

- [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) · [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B)
- [TRL On-policy Distillation 文档](https://huggingface.co/docs/trl/main/en/distillation_trainer)
- [On-Policy Distillation of Language Models](https://arxiv.org/abs/2306.13649)
