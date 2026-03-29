# 🤖 vocab_32k_gpt2

一个从零开始训练的 32K 词表 GPT2 项目，包含完整的三阶段流程：

1. 预训练（Pretrain）
2. 指令微调（SFT）
3. 偏好优化（DPO）

本仓库作为本科毕业设计实践，基于 PyTorch + Hugging Face Transformers + Accelerate + DeepSpeed 实现，支持多卡训练与断点续训。

## ✨ 1. 项目特点

- 自定义 `vocab_32k_gpt2` 模型与 tokenizer（SentencePiece 32K 词表）
- 统一训练入口：`train.py` + `trainer.py`
- 支持两类训练模式：`pretrain`、`instruct`
- 支持 DPO 对齐训练：`dpo.py`
- 使用 `accelerate + deepspeed` 进行分布式训练
- 支持 checkpoint 自动恢复（通过 `accelerator.load_state(work_dir)`）

## 🗂️ 2. 仓库结构

```text
.
├── train.py                         # 预训练 / SFT 统一训练入口
├── trainer.py                       # 训练循环、日志、保存与恢复
├── dpo.py                           # DPO 训练脚本（TRL DPOTrainer）
├── requirements.txt                 # Python 依赖
├── logs/
│   └── pretrain_log.out             # 训练日志归档
├── docs/
│   └── STRUCTURE.md                 # 仓库结构与约定说明
├── scripts/
│   ├── launch/
│   │   ├── pre_train.sh             # 预训练启动命令
│   │   ├── sft.sh                   # SFT 启动命令
│   │   ├── sft4dpo.sh               # DPO 前的 SFT 启动命令
│   │   └── dpo.sh                   # DPO 启动命令
│   └── eval/
│       ├── test_base_ckpt.py        # 基座模型推理测试
│       └── test_sft_ckpt.py         # SFT 模型推理测试
├── configs/
│   ├── pretrain_config.yaml         # 预训练配置
│   ├── instruct_config.yaml         # SFT 配置
│   ├── dpo_instruct_config.yaml     # DPO 前 SFT 配置
│   ├── model_configs/vocab_32k_gpt2.json
│   └── accelerate_configs/*.yaml    # DeepSpeed/Accelerate 配置
├── dataset/
│   ├── dataset.py                   # 数据读取与预处理（pretrain/instruct）
│   ├── data_iter.py                 # 可迭代数据集工具
│   ├── validation.py                # 训练期间验证样例
│   └── legacy/
│       └── sft_dataset.py           # 历史版本数据处理脚本（归档）
├── models/
│   ├── configuration_vocab_32k_gpt2.py
│   ├── modeling_vocab_32k_gpt2.py
│   └── tokenization_vocab_32k_gpt2.py
└── utils/
    └── spm_to_hf_tknizer.py
```

## ⚙️ 3. 环境准备

### 🐍 3.1 Python 与 CUDA

建议环境：

- Python 3.10+
- CUDA 11.8+（按你的 PyTorch 版本匹配）
- Linux 或 WSL2（`.sh` 脚本默认是 Linux 风格）

### 📦 3.2 安装依赖

```bash
pip install -r requirements.txt
```

`dpo.py` 使用了 `trl.DPOTrainer`，如果你的环境未安装 `trl`，请额外安装：

```bash
pip install trl
```

## 🧠 4. 模型与分词器说明

- 词表：`configs/tokenizer_models/vocab_32k_gpt2.model`（SentencePiece 32K）
- 模型配置：`configs/model_configs/vocab_32k_gpt2.json`
- `models/configuration_vocab_32k_gpt2.py` 默认结构与 GPT2-base 接近：
  - `n_layer=12`
  - `n_head=12`
  - `n_embd=768`
  - `n_positions=1024`

Token ID（配置文件中）：

- `bos_token_id = 1`
- `eos_token_id = 2`
- `pad_token_id = 3`
- `vocab_size = 32000`

## 🧾 5. 数据格式

本项目的数据入口由 `dataset/dataset.py` 控制，按 `config.data.mode` 分两类。

### 📚 5.1 预训练数据（`mode: pretrain`）

配置示例见 `configs/pretrain_config.yaml`：

- `data/SkyPile-150B/rawdata/2020-40_zh_*.jsonl`
- `data/openwebtext/openwebtext.jsonl`

当前预处理函数 `pretrain_transform` 默认使用字段 `text`，因此单条样本至少应包含：

```json
{"text": "你的预训练文本"}
```

### 🧑‍🏫 5.2 指令数据（`mode: instruct`）

配置示例见：

- `configs/instruct_config.yaml`（`data/sft_merge.jsonl`）
- `configs/dpo_instruct_config.yaml`（`data/DPO/mix_dpo_data4sft.jsonl`）

`instruct_transform` 依赖字段：

- `instruction`（必需）
- `output`（必需）
- `input`（可为空字符串）
- `history`（可为空列表）

推荐 JSONL 样例：

```json
{"instruction": "介绍一下你自己", "input": "", "output": "我是一个语言模型...", "history": []}
```

### ⚖️ 5.3 DPO 数据

`dpo.py` 读取：`data/DPO/mix_dpo_data.jsonl`

每条样本需要包含：

- `question`
- `response_j`（偏好更优）
- `response_k`（偏好较差）

样例：

```json
{"question": "如何学习深度学习？", "response_j": "建议先学线代和Python...", "response_k": "不知道"}
```

## 🚀 6. 训练流程（推荐）

建议顺序：Pretrain -> SFT -> SFT-for-DPO -> DPO

### 6.1 🔥 预训练

```bash
bash scripts/launch/pre_train.sh
```

等价核心命令：

```bash
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 \
accelerate launch \
  --config_file configs/accelerate_configs/ds_stage2.yaml \
  train.py \
  --train_config configs/pretrain_config.yaml \
  --model_config configs/model_configs/vocab_32k_gpt2.json
```

输出目录（默认）：`ckpt/vocab_32k_gpt2`

### 6.2 🛠️ 指令微调（SFT）

```bash
bash scripts/launch/sft.sh
```

对应配置：`configs/instruct_config.yaml`

- 默认从 `ckpt/vocab_32k_gpt2/` 加载预训练模型
- 输出到 `ckpt/vocab_32k_gpt2_instruction`

### 6.3 🧩 DPO 前的 SFT

```bash
bash scripts/launch/sft4dpo.sh
```

对应配置：`configs/dpo_instruct_config.yaml`

- 默认从 `ckpt/vocab_32k_gpt2_instruction/checkpoint_epoch4` 继续训练
- 输出到 `ckpt/vocab_32k_gpt2_sft4dpo`

### 6.4 🎯 DPO 训练

```bash
bash scripts/launch/dpo.sh
```

等价命令：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
accelerate launch \
  --config_file configs/accelerate_configs/ds_stage2.yaml \
  dpo.py
```

默认输出目录：`ckpt/vocab_32k_gpt2_dpo/`

## 🧷 7. 关键配置说明

### 7.1 📘 训练配置（`configs/*.yaml`）

常用字段：

- `data.mode`: `pretrain` 或 `instruct`
- `data.seq_length`: 序列长度
- `train.train_batch_size`: 每进程 batch size
- `train.gradient_accumulation_steps`: 梯度累积步数
- `train.num_training_steps`: 总训练步数
- `train.num_warmup_steps`: warmup 步数
- `train.lr`: 学习率
- `work_dir`: checkpoint 存储路径

### 7.2 🖥️ 分布式配置（`configs/accelerate_configs/*.yaml`）

可选：

- `ds_stage1.yaml`
- `ds_stage2.yaml`（脚本默认）
- `ds_stage3.yaml`
- `ds_stage3_offload.yaml`

显存不足时可尝试 `stage3` 或 `stage3_offload`。

## 💾 8. 日志、保存与断点续训

`trainer.py` 中：

- 每隔 `save_interval` 自动保存到 `work_dir/checkpoint_epochX`
- 启动时会尝试从 `work_dir` 自动恢复 `accelerator` 状态
- 使用 Weights & Biases 记录日志，默认 `WANDB_MODE=offline`

注意：`trainer.py` 当前写死了 `WANDB_API_KEY` 与 `offline` 模式。如果你需要线上同步，请按需修改环境变量设置。

## 🧪 9. 推理与快速测试

### 9.1 🧱 基座模型测试

```bash
python scripts/eval/test_base_ckpt.py
```

### 9.2 🗣️ SFT 模型测试

```bash
python scripts/eval/test_sft_ckpt.py
```

两者都会读取 `dataset/validation.py` 中的提示词并打印生成结果。

## ✅ 10. 可复现建议

- 固定随机种子（项目中已对部分流程设置 `seed`）
- 记录每次实验配置副本（建议保存一份 `configs/*.yaml`）
- 记录 GPU 数量、显存、CUDA 与依赖版本

## 🙏 11. 致谢

本项目部分实现参考了 Open-Llama/Hugging Face 生态工具链（Transformers、Accelerate、DeepSpeed、Datasets、TRL）。