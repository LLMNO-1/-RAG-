"""
BGE-M3 LoRA 微调脚本
====================
针对掌柜智库项目，基于 training_data.json 微调 BGE-M3 embedding 模型。

用法：
    python finetune_bge_m3.py

依赖（Windows 环境）：
    pip install FlagEmbedding peft torch transformers accelerate

显存需求：RTX 2060 6GB 即可，batch_size=4
训练时间：约 30-60 分钟（60 条样本，3 epochs）
"""
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# ============================================================
# 配置区 —— 按需修改
# ============================================================

# BGE-M3 基础模型路径
# 实际路径: D:\models\bge_m3（非 .env 里写的旧路径）
BASE_MODEL_PATH = r"D:\models\bge_m3"

# 训练数据路径
TRAIN_DATA_PATH = Path(__file__).parent / "training_data.json"

# 输出目录
OUTPUT_DIR = Path(__file__).parent / "finetuned_bge_m3"

# 训练超参
BATCH_SIZE = 2            # RTX 2060 6GB，9条文本/batch 时降到 2
NUM_EPOCHS = 3
LEARNING_RATE = 2e-4
WARMUP_STEPS = 50
MAX_SEQ_LENGTH = 512

# LoRA 配置
LORA_R = 8                # rank，越大效果越好但显存越多
LORA_ALPHA = 16
LORA_DROPOUT = 0.1

# 验证集比例
VAL_SPLIT = 0.15          # 60 × 0.15 ≈ 9 条用于验证

# ============================================================
# 步骤 1：加载训练数据并转换为标准格式
# ============================================================

def load_training_data(path):
    """加载 training_data.json，转换为 FlagEmbedding 训练格式"""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    
    print(f"加载了 {len(raw)} 条训练样本")
    
    # 统计各文档分布
    doc_counts = {}
    for d in raw:
        doc = d["doc_name"]
        doc_counts[doc] = doc_counts.get(doc, 0) + 1
    for doc, count in sorted(doc_counts.items()):
        print(f"  {doc}: {count} 条")
    
    # 转换为训练格式：{"query": str, "pos": [str], "neg": [str]}
    converted = []
    skipped = 0
    for d in raw:
        pos_content = d["positive"]["content"].strip()
        negs_with_type = [{"content": n["content"].strip(), "type": n.get("type", "other")} 
                         for n in d["negatives"] if n.get("content", "").strip()]
        
        if not pos_content or not negs_with_type:
            skipped += 1
            continue
        
        converted.append({
            "query": d["query"],
            "pos": [pos_content],
            "neg": negs_with_type,
            "doc_name": d["doc_name"],
            "match_score": d.get("match_score", 0),
        })
    
    print(f"有效样本: {len(converted)} (跳过 {skipped} 条空数据)")
    return converted


def split_train_val(data, val_ratio=0.15):
    """按文档分层划分训练集和验证集"""
    import random
    random.seed(42)
    
    # 按文档分组
    by_doc = {}
    for d in data:
        by_doc.setdefault(d["doc_name"], []).append(d)
    
    train, val = [], []
    for doc, items in by_doc.items():
        random.shuffle(items)
        n_val = max(1, int(len(items) * val_ratio))
        train.extend(items[:-n_val])
        val.extend(items[-n_val:])
    
    print(f"\n训练集: {len(train)} 条, 验证集: {len(val)} 条")
    return train, val


# ============================================================
# 步骤 2：加载模型
# ============================================================

def check_environment():
    """检查依赖和环境"""
    print("\n=== 环境检查 ===")
    errors = []
    
    # 检查 PyTorch
    try:
        import torch
        print(f"PyTorch: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"CUDA: 可用, 设备: {torch.cuda.get_device_name(0)}")
            print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            errors.append("CUDA 不可用，微调需要 GPU")
    except ImportError:
        errors.append("未安装 PyTorch: pip install torch")
    
    # 检查 FlagEmbedding
    try:
        import FlagEmbedding
        print(f"FlagEmbedding: 已安装")
    except ImportError:
        errors.append("未安装 FlagEmbedding: pip install FlagEmbedding")
    
    # 检查 peft
    try:
        import peft
        print(f"PEFT: 已安装")
    except ImportError:
        errors.append("未安装 peft: pip install peft")
    
    # 检查基础模型
    if not os.path.exists(BASE_MODEL_PATH):
        errors.append(f"基础模型路径不存在: {BASE_MODEL_PATH}")
    else:
        print(f"基础模型: {BASE_MODEL_PATH}")
    
    if errors:
        print("\n[FAIL] 环境检查失败:")
        for e in errors:
            print(f"  - {e}")
        return False
    
    print("[OK] 环境检查通过")
    return True


# ============================================================
# 步骤 3：训练
# ============================================================

def train():
    """执行 LoRA 微调（使用 HuggingFace + PEFT）"""
    import torch
    from torch.utils.data import Dataset, DataLoader
    from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
    from peft import get_peft_model, LoraConfig, TaskType
    from tqdm import tqdm
    
    # 准备输出路径
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 加载并划分数据
    data = load_training_data(TRAIN_DATA_PATH)
    train_data, val_data = split_train_val(data, VAL_SPLIT)
    
    print(f"\n{'='*60}")
    print(f"开始训练")
    print(f"  基础模型: {BASE_MODEL_PATH}")
    print(f"  训练集: {len(train_data)} 条, 验证集: {len(val_data)} 条")
    print(f"  LoRA: r={LORA_R}, alpha={LORA_ALPHA}")
    print(f"  Epochs: {NUM_EPOCHS}, Batch: {BATCH_SIZE}, LR: {LEARNING_RATE}")
    print(f"{'='*60}\n")
    
    # 加载 tokenizer 和模型
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    model = AutoModel.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    
    # 配置 LoRA
    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["query", "key", "value", "dense"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model = model.cuda()
    model.gradient_checkpointing_enable()  # 省显存
    
    # 准备训练数据：优先使用 hard_neg，不足时用其他负例补齐到 neg_count
    class ContrastiveDataset(Dataset):
        def __init__(self, data, tokenizer, max_len=256, neg_count=7):
            self.samples = []
            for d in data:
                query = d["query"]
                pos = d["pos"][0]
                # 分类负例
                hard = [n["content"] for n in d["neg"] if n.get("type") == "hard_neg"]
                others = [n["content"] for n in d["neg"] if n.get("type") != "hard_neg"]
                # hard_neg 优先，不足时从 others 补齐
                negs = hard + others
                negs = negs[:neg_count]
                self.samples.append((query, pos, negs))
            self.tokenizer = tokenizer
            self.max_len = max_len
        
        def __len__(self):
            return len(self.samples)
        
        def __getitem__(self, idx):
            query, pos, negs = self.samples[idx]
            all_texts = [query] + [pos] + negs
            encoded = self.tokenizer(
                all_texts, padding="max_length", truncation=True,
                max_length=self.max_len, return_tensors="pt"
            )
            return encoded
    
    train_dataset = ContrastiveDataset(train_data, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_loader) * NUM_EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, WARMUP_STEPS, total_steps)
    
    # 训练
    model.train()
    global_step = 0
    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        for batch in pbar:
            # Move to GPU
            batch = {k: v.cuda() for k, v in batch.items()}
            
            # 实际 batch size（最后一个 batch 可能不足）
            bs = batch["input_ids"].shape[0]
            n_texts = 1 + 1 + 7  # query + positive + 7 negatives
            
            # Reshape: [bs * n_texts, seq_len]
            input_ids = batch["input_ids"].view(bs * n_texts, -1)
            attention_mask = batch["attention_mask"].view(bs * n_texts, -1)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            # CLS pooling（BGE-M3 原生方式）
            embeddings = outputs.last_hidden_state[:, 0, :]  # [bs*n_texts, dim]
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
            # Reshape back: [bs, n_texts, dim]
            embeddings = embeddings.view(bs, n_texts, -1)
            
            # Query: [bs, dim], Positive: [bs, dim], Negatives: [bs, neg_count, dim]
            q_emb = embeddings[:, 0, :]
            p_emb = embeddings[:, 1, :]
            n_emb = embeddings[:, 2:, :]
            
            # InfoNCE loss
            pos_scores = (q_emb * p_emb).sum(dim=1) / 0.05  # temperature=0.05
            neg_scores = torch.bmm(n_emb, q_emb.unsqueeze(-1)).squeeze(-1) / 0.05
            
            logits = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1)
            labels = torch.zeros(bs, dtype=torch.long, device=logits.device)
            loss = torch.nn.functional.cross_entropy(logits, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
            global_step += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_loss = epoch_loss / len(train_loader)
        print(f"  Epoch {epoch+1} 平均 loss: {avg_loss:.4f}")
    
    # 保存 LoRA adapter
    lora_path = OUTPUT_DIR / "adapter"
    model.save_pretrained(str(lora_path))
    tokenizer.save_pretrained(str(lora_path))
    print(f"\n[OK] 训练完成！LoRA adapter 保存在: {lora_path}")


# ============================================================
# 步骤 4：验证
# ============================================================

def validate():
    """在验证集上评测微调前后的模型"""
    print(f"\n{'='*60}")
    print("模型验证")
    print(f"{'='*60}")
    
    import torch
    import numpy as np
    
    def load_bgem3(model_path):
        """加载 BGE-M3 模型"""
        from FlagEmbedding.inference.embedder import BGEM3FlagModel
        model = BGEM3FlagModel(
            model_path,
            devices=["cuda:0"],
            use_fp16=True,
        )
        print(f"    [BGEM3FlagModel] {model_path}")
        return model
    
    val_path = OUTPUT_DIR / "val_data.json"
    if not val_path.exists():
        print("[FAIL] 验证数据不存在，跳过验证")
        return
    
    with open(val_path, encoding="utf-8") as f:
        val_data = json.load(f)
    
    print(f"验证集: {len(val_data)} 条")
    
    # 加载原始模型
    print("\n加载原始模型...")
    original_model = None
    try:
        original_model = load_bgem3(BASE_MODEL_PATH)
        print(f"  原始模型加载成功")
    except Exception as e:
        print(f"  [WARN] 加载原始模型失败: {e}，仅评测微调后模型")
    
    # 加载微调后模型（需要 merged 目录，因为 BGEM3FlagModel 不支持纯 PEFT adapter）
    print("加载微调后模型...")
    merged_path = OUTPUT_DIR / "merged"
    adapter_path = OUTPUT_DIR / "adapter"
    
    if merged_path.exists():
        try:
            finetuned_model = load_bgem3(str(merged_path))
            print(f"  微调后模型加载成功 (merged)")
        except Exception as e:
            print(f"  [WARN] 加载 merged 模型失败: {e}")
            return
    elif adapter_path.exists():
        print(f"  [WARN] 未找到 merged 模型，请先运行 --skip-train --skip-validate 导出模型")
        return
    else:
        print(f"  [FAIL] 未找到任何微调模型，请先运行训练")
        return
    
    def evaluate_model(model, data, name):
        """评估模型分离度：正例相似度 vs 负例相似度"""
        queries = [d["query"] for d in data]
        positives = [d["pos"][0] for d in data]
        
        # 编码（BGEM3FlagModel API: encode_queries + encode）
        q_result = model.encode_queries(queries)
        p_result = model.encode(positives, batch_size=8)
        
        q_dense = torch.from_numpy(np.array(q_result["dense_vecs"])).float()
        p_dense = torch.from_numpy(np.array(p_result["dense_vecs"])).float()
        
        # 归一化
        q_dense = torch.nn.functional.normalize(q_dense, p=2, dim=1)
        p_dense = torch.nn.functional.normalize(p_dense, p=2, dim=1)
        
        # 正例相似度 (query[i] vs positive[i])
        pos_sims = (q_dense * p_dense).sum(dim=1).tolist()
        avg_pos = sum(pos_sims) / len(pos_sims)
        
        # 负例相似度 (query[i] vs negatives[i])
        all_neg_sims = []
        for i, d in enumerate(data):
            negs = d["neg"][:5]
            if negs:
                n_result = model.encode(negs, batch_size=8)
                n_dense = torch.from_numpy(np.array(n_result["dense_vecs"])).float()
                n_dense = torch.nn.functional.normalize(n_dense, p=2, dim=1)
                q_i = q_dense[i].unsqueeze(0)
                sims = (q_i @ n_dense.T).squeeze(0).tolist()
                all_neg_sims.extend(sims if isinstance(sims, list) else [sims])
        
        avg_neg = sum(all_neg_sims) / len(all_neg_sims) if all_neg_sims else 0
        separation = avg_pos - avg_neg
        
        print(f"\n  [{name}]")
        print(f"    正例平均相似度:  {avg_pos:.4f}")
        print(f"    负例平均相似度:  {avg_neg:.4f}")
        print(f"    分离度 (pos-neg): {separation:.4f}  <- 越大越好")
        
        return avg_pos, avg_neg, separation
    
    # 评测微调后模型
    finetuned_result = evaluate_model(finetuned_model, val_data, "微调后")
    
    # 评测原始模型
    if original_model:
        original_result = evaluate_model(original_model, val_data, "原始模型")
        
        pos_delta = finetuned_result[0] - original_result[0]
        sep_delta = finetuned_result[2] - original_result[2]
        
        print(f"\n  {'-' * 40}")
        print(f"  正例相似度变化: {pos_delta:+.4f}  (微调后 - 原始)")
        print(f"  分离度变化:     {sep_delta:+.4f}  (微调后 - 原始)")
        if sep_delta > 0.01:
            print(f"  [OK] 微调有效！分离度提升了 {sep_delta:.4f}")
        elif sep_delta > -0.01:
            print(f"  ⚡ 分离度持平，可尝试更多 epochs 或降低学习率")
        else:
            print(f"  [WARN] 分离度下降，建议调整超参数（降低 LR 或减少 epochs）")


# ============================================================
# 步骤 5：导出模型
# ============================================================

def export_model():
    """将 LoRA 权重合并到基础模型，导出为可直接使用的完整模型"""
    print(f"\n{'='*60}")
    print("导出模型")
    print(f"{'='*60}")
    
    merged_path = OUTPUT_DIR / "merged"
    adapter_path = OUTPUT_DIR / "adapter"
    
    if not adapter_path.exists():
        print(f"[FAIL] LoRA adapter 不存在: {adapter_path}")
        print(f"   请先运行训练")
        return
    
    try:
        from peft import PeftModel
        from transformers import AutoModel, AutoTokenizer
        import torch
        
        print(f"加载基础模型...")
        base_model = AutoModel.from_pretrained(
            BASE_MODEL_PATH,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
        
        print(f"加载 LoRA 权重...")
        lora_model = PeftModel.from_pretrained(base_model, str(adapter_path))
        
        print(f"合并权重...")
        merged_model = lora_model.merge_and_unload()
        
        print(f"保存到 {merged_path}...")
        merged_path.mkdir(parents=True, exist_ok=True)
        merged_model.save_pretrained(str(merged_path))
        tokenizer.save_pretrained(str(merged_path))
        
        print(f"[OK] 模型已导出到: {merged_path}")
        print(f"\n使用方法：将 .env 中的 BGE_M3_PATH 改为:")
        print(f"  BGE_M3_PATH={merged_path}")
        
    except Exception as e:
        print(f"[FAIL] 导出失败: {e}")
        import traceback
        traceback.print_exc()


# ============================================================
# 主入口
# ============================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="BGE-M3 LoRA 微调")
    parser.add_argument("--skip-train", action="store_true", help="跳过训练，仅验证和导出")
    parser.add_argument("--skip-validate", action="store_true", help="跳过验证")
    parser.add_argument("--skip-export", action="store_true", help="跳过导出")
    args = parser.parse_args()
    
    print("=" * 60)
    print("掌柜智库 — BGE-M3 LoRA 微调")
    print(f"启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    if not check_environment():
        sys.exit(1)
    
    if not args.skip_train:
        train()
    else:
        print("\n[SKIP] 跳过训练")
    
    if not args.skip_export:
        export_model()
    else:
        print("\n[SKIP] 跳过导出")
    
    if not args.skip_validate:
        validate()
    else:
        print("\n[SKIP] 跳过验证")
    
    print(f"\n{'='*60}")
    print("全部完成！")
    print(f"{'='*60}")
