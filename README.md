# 🤖 vocab_32k_gpt2_moe

一个从零开始训练的 32K 词表 GPT2-MoE 项目，包含完整三阶段流程：

1. 预训练（Pretrain）
2. 指令微调（SFT）
3. 偏好优化（DPO）

本仓库基于 PyTorch + Hugging Face Transformers + Accelerate + DeepSpeed + TRL 实现，支持多卡训练和断点续训。

## ✨ 1. 项目特点

- 自定义 `vocab_32k_gpt2_moe` 模型与 SentencePiece 32K tokenizer
- 统一训练入口：`train.py` + `trainer.py`
- 两类训练模式：`pretrain`、`instruct`
- DPO 对齐训练脚本：`dpo.py`（`trl.DPOTrainer`）
- 使用 `accelerate + deepspeed` 进行分布式训练
- 支持通过 `accelerator.load_state(work_dir)` 自动恢复训练状态

## 🗂️ 2. 仓库结构

```text
.
├── train.py
├── trainer.py
├── dpo.py
├── requirements.txt
├── README.md
├── configs/
│   ├── pretrain_config.yaml
│   ├── instruct_config.yaml
│   ├── dpo_instruct_config.yaml
│   ├── accelerate_configs/
│   │   ├── ds_stage1.yaml
│   │   ├── ds_stage2.yaml
│   │   ├── ds_stage3.yaml
│   │   └── ds_stage3_offload.yaml
│   ├── model_configs/
│   │   └── vocab_32k_gpt2_moe.json
│   └── tokenizer_models/
│       ├── vocab_32k_gpt2_moe.model
│       └── vocab_32k_gpt2_moe.vocab
├── dataset/
│   ├── dataset.py
│   ├── data_iter.py
│   └── validation.py
├── models/
│   ├── configuration_vocab_32k_gpt2_moe.py
│   ├── modeling_vocab_32k_gpt2_moe.py
│   └── tokenization_vocab_32k_gpt2.py
└── scripts/
    ├── launch/
    │   ├── pre_train_moe.sh
    │   ├── sft.sh
    │   ├── sft4dpo.sh
    │   └── dpo.sh
    └── eval/
        ├── test_base_ckpt.py
        └── test_sft_ckpt.py
```

## ⚙️ 3. 环境准备

### 🐍 3.1 Python 与 CUDA

建议环境：

- Python 3.10+
- CUDA 11.8+（按 PyTorch 版本匹配）
- Linux 或 WSL2（`scripts/launch/*.sh` 为 Linux shell 风格）

### 📦 3.2 安装依赖

```bash
pip install -r requirements.txt
pip install trl
```

说明：`dpo.py` 依赖 `trl.DPOTrainer`，请确保已安装 `trl`。

## 🧠 4. 模型与分词器说明

- 词表模型：`configs/tokenizer_models/vocab_32k_gpt2_moe.model`
- 模型配置：`configs/model_configs/vocab_32k_gpt2_moe.json`
- Token ID：
  - `bos_token_id = 1`
  - `eos_token_id = 2`
  - `pad_token_id = 3`
  - `vocab_size = 32000`

MoE 相关默认参数（见 `configs/model_configs/vocab_32k_gpt2_moe.json`）：

- `n_layer = 12`
- `moe_layer_freq = 1`
- `n_routed_experts = 8`
- `num_experts_per_tok = 2`

## 🧾 5. 数据格式

数据处理入口在 `dataset/dataset.py`，由 `config.data.mode` 控制。

### 📚 5.1 预训练数据（`mode: pretrain`）

默认配置见 `configs/pretrain_config.yaml`：

- `data/skypile/2020-40_*.jsonl`
- `data/openwebtext/openwebtext.jsonl`

`pretrain_transform` 当前默认读取 `text` 字段，因此样本至少应包含：

```json
{"text": "你的预训练文本"}
```

### 🧑‍🏫 5.2 指令数据（`mode: instruct`）

默认配置见：

- `configs/instruct_config.yaml`（`data/sft_merge.jsonl`）
- `configs/dpo_instruct_config.yaml`（`data/DPO/mix_dpo_data4sft.jsonl`）

`instruct_transform` 依赖字段：

- `instruction`（必需）
- `output`（必需）
- `input`（可为空字符串）
- `history`（可为空列表）

示例：

```json
{"instruction": "介绍一下你自己", "input": "", "output": "我是一个语言模型...", "history": []}
```

### ⚖️ 5.3 DPO 数据

`dpo.py` 默认读取：`data/DPO/mix_dpo_data.jsonl`

每条样本需包含：

- `question`
- `response_j`（偏好更优）
- `response_k`（偏好较差）

示例：

```json
{"question": "如何学习深度学习？", "response_j": "建议先学线代和Python...", "response_k": "不知道"}
```

## 🚀 6. 训练流程（推荐）

推荐顺序：Pretrain -> SFT -> SFT-for-DPO -> DPO

### 🔥 6.1 预训练

```bash
bash scripts/launch/pre_train_moe.sh
```

等价核心命令：

```bash
CUDA_VISIBLE_DEVICES=0,1,4,5,6,7 \
accelerate launch \
  --config_file configs/accelerate_configs/ds_stage2.yaml \
  train.py \
  --train_config configs/pretrain_config.yaml \
  --model_config configs/model_configs/vocab_32k_gpt2_moe.json
```

默认输出目录：`ckpt/vocab_32k_gpt2_moe`

### 🛠️ 6.2 指令微调（SFT）

```bash
bash scripts/launch/sft.sh
```

对应配置：`configs/instruct_config.yaml`

- 默认从 `ckpt/vocab_32k_gpt2_moe/` 继续训练
- 输出到 `ckpt/vocab_32k_gpt2_moe_instruction`

### 🧩 6.3 DPO 前的 SFT

```bash
bash scripts/launch/sft4dpo.sh
```

对应配置：`configs/dpo_instruct_config.yaml`

- 默认从 `ckpt/vocab_32k_gpt2_moe_instruction/checkpoint_epoch4` 继续训练
- 输出到 `ckpt/vocab_32k_gpt2_moe_sft4dpo`

### 🎯 6.4 DPO 训练

```bash
bash scripts/launch/dpo.sh
```

等价命令：

```bash
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 \
accelerate launch \
  --config_file configs/accelerate_configs/ds_stage2.yaml \
  dpo.py
```

默认输出目录：`ckpt/vocab_32k_gpt2_moe_dpo/`

## 🧷 7. 关键配置说明

### 📘 7.1 训练配置（`configs/*.yaml`）

常用字段：

- `data.mode`: `pretrain` 或 `instruct`
- `data.seq_length`: 序列长度
- `train.train_batch_size`: 每进程 batch size
- `train.gradient_accumulation_steps`: 梯度累积步数
- `train.num_training_steps`: 总训练步数
- `train.num_warmup_steps`: warmup 步数
- `train.lr`: 学习率
- `train.ckpt`: 初始化模型路径（可为空）
- `work_dir`: checkpoint 存储路径

### 🖥️ 7.2 分布式配置（`configs/accelerate_configs/*.yaml`）

可选：

- `ds_stage1.yaml`
- `ds_stage2.yaml`（脚本默认）
- `ds_stage3.yaml`
- `ds_stage3_offload.yaml`

显存不足时可尝试 `stage3` 或 `stage3_offload`。

## 💾 8. 日志、保存与断点续训

`trainer.py` 中：

- 每隔 `save_interval` 保存状态到 `work_dir/checkpoint_epochX`
- 启动时尝试 `accelerator.load_state(work_dir)` 自动恢复
- 默认使用 Weights & Biases，并设置了 `WANDB_MODE=offline`

注意：`trainer.py` 当前写死了 `WANDB_API_KEY` 和 `WANDB_MODE=offline`，如需线上同步建议改为环境变量注入。

## 🧪 9. 推理与快速测试

### 🧱 9.1 基座模型测试

```bash
python scripts/eval/test_base_ckpt.py
```

### 🗣️ 9.2 SFT 模型测试

```bash
python scripts/eval/test_sft_ckpt.py
```

以上脚本会读取 `dataset/validation.py` 中的提示词并打印生成结果。

## ✅ 10. 可复现建议

- 固定随机种子（项目已在部分流程中设置 `seed`）
- 记录实验配置副本（建议保存对应的 `configs/*.yaml`）
- 记录 GPU 数量、显存、CUDA 与依赖版本

## 📝 11. 补充说明

`dpo.py` 中 `ScriptArguments.model_name_or_path` 默认值为 `ckpt/vocab_32k_gpt2_moe_sft4dpo/checkpoint_epoch6`，但当前代码实际加载路径写死为同一路径。请根据你真实的 SFT-for-DPO 输出 checkpoint 调整该路径，避免找不到模型。
