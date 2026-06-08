"""
Hard Negative 构建脚本
=====================
不依赖 Milvus，直接用 BGE-M3 模型计算相似度，
为每条 query 找到"最相似但不是正确答案"的 chunk 作为 hard negative。

用法：
    python eval/build_hard_negatives.py

输出：更新 eval/training_data.json，为每条样本的 negatives 增加 hard_neg 类型
"""

import json
import os
from pathlib import Path

# ============================================================
# 配置
# ============================================================

BASE_MODEL_PATH = r"D:\models\bge_m3"
TRAIN_DATA_PATH = Path(__file__).parent / "training_data.json"
TOP_K_HARD = 3  # 每条样本取几个 hard negative

# ============================================================
# 步骤 1：加载数据
# ============================================================

def load_data():
    with open(TRAIN_DATA_PATH, encoding="utf-8") as f:
        data = json.load(f)
    print(f"加载 {len(data)} 条训练样本")
    return data


# ============================================================
# 步骤 2：加载 BGE-M3 并编码
# ============================================================

def load_model():
    """加载 BGE-M3 模型"""
    from FlagEmbedding.inference.embedder import BGEM3FlagModel
    
    model = BGEM3FlagModel(
        BASE_MODEL_PATH,
        devices=["cuda:0"],
        use_fp16=True,
    )
    print(f"[FlagEmbedding] 模型加载成功: {BASE_MODEL_PATH}")
    return model


def compute_similarities(model, query, passages):
    """计算 query 与多个 passage 的余弦相似度
    
    Returns:
        list of (index, similarity_score)
    """
    import torch
    import numpy as np
    
    # 编码 query
    q_result = model.encode_queries([query])
    q_dense = torch.from_numpy(np.array(q_result["dense_vecs"])).float()
    
    # 编码 passages
    p_result = model.encode(passages, batch_size=8)
    p_dense = torch.from_numpy(np.array(p_result["dense_vecs"])).float()
    
    # 归一化
    q_dense = torch.nn.functional.normalize(q_dense, p=2, dim=1)
    p_dense = torch.nn.functional.normalize(p_dense, p=2, dim=1)
    
    # 相似度
    sims = (q_dense @ p_dense.T).squeeze(0)
    return [(i, sims[i].item()) for i in range(len(passages))]


# ============================================================
# 步骤 3：构建 hard negatives
# ============================================================

def build_hard_negatives(data, model):
    """为每条样本找到 hard negatives"""
    
    # 按文档分组
    by_doc = {}
    for i, d in enumerate(data):
        by_doc.setdefault(d["doc_name"], []).append(i)
    
    # 收集所有 chunk（去重）
    all_content_set = {}
    chunk_id_to_content = {}
    for d in data:
        cid = d["positive"]["chunk_id"]
        content = d["positive"]["content"]
        if cid not in all_content_set:
            all_content_set[cid] = content
        chunk_id_to_content[cid] = content
    # Also add negative chunks
    for d in data:
        for n in d["negatives"]:
            cid = n["chunk_id"]
            if cid not in all_content_set:
                all_content_set[cid] = n["content"]
    
    # 按文档组织 chunks
    doc_chunks = {}
    for cid, content in all_content_set.items():
        doc_name = cid.rsplit("_", 1)[0]  # "hak180产品安全手册_5" → "hak180产品安全手册"
        doc_chunks.setdefault(doc_name, {})[cid] = content
    
    print(f"\n按文档分组:")
    for doc_name, chunks in doc_chunks.items():
        print(f"  {doc_name}: {len(chunks)} unique chunks")
    
    # 为每条样本找 hard negative
    total_added = 0
    for i, d in enumerate(data):
        doc_name = d["doc_name"]
        pos_chunk_id = d["positive"]["chunk_id"]
        
        # 获取同文档的所有 chunk（排除正例）
        same_doc_chunks = doc_chunks.get(doc_name, {})
        candidates = {cid: content for cid, content in same_doc_chunks.items() 
                      if cid != pos_chunk_id}
        
        if len(candidates) == 0:
            continue
        
        # 编码并计算相似度
        cand_ids = list(candidates.keys())
        cand_texts = [candidates[cid] for cid in cand_ids]
        
        try:
            sims = compute_similarities(model, d["query"], cand_texts)
            sims.sort(key=lambda x: x[1], reverse=True)
            
            # 取 top-K 作为 hard negative
            hard_neg_ids = cand_ids[sims[0][0]] if sims else None
            hard_negs = []
            for idx, score in sims[:TOP_K_HARD]:
                hard_negs.append({
                    "content": candidates[cand_ids[idx]],
                    "chunk_id": cand_ids[idx],
                    "doc_name": doc_name,
                    "type": "hard_neg",
                    "similarity": round(score, 4),
                })
            
            # 追加到 negatives
            d["negatives"].extend(hard_negs)
            total_added += len(hard_negs)
            
            if (i + 1) % 10 == 0:
                print(f"  进度: {i+1}/{len(data)}")
                
        except Exception as e:
            print(f"  ⚠ 样本 {i} ({d['query'][:40]}...) 失败: {e}")
    
    print(f"\n共添加 {total_added} 个 hard negatives")
    return data


# ============================================================
# 步骤 4：保存
# ============================================================

def save_data(data):
    # 备份原文件
    backup_path = Path(str(TRAIN_DATA_PATH) + ".bak")
    import shutil
    shutil.copy(TRAIN_DATA_PATH, backup_path)
    print(f"已备份原数据到: {backup_path}")
    
    with open(TRAIN_DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    # 统计
    total_neg = sum(len(d["negatives"]) for d in data)
    hard_count = sum(sum(1 for n in d["negatives"] if n.get("type") == "hard_neg") for d in data)
    same_count = sum(sum(1 for n in d["negatives"] if n.get("type") == "same_doc") for d in data)
    cross_count = sum(sum(1 for n in d["negatives"] if n.get("type") == "cross_doc") for d in data)
    
    print(f"\n更新完成！")
    print(f"  总样本数: {len(data)}")
    print(f"  负例分布: {same_count} same_doc + {cross_count} cross_doc + {hard_count} hard_neg = {total_neg}")
    
    file_size = os.path.getsize(TRAIN_DATA_PATH) / 1024
    print(f"  文件大小: {file_size:.0f} KB")


# ============================================================
# 主入口
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Hard Negative 构建")
    print(f"模型: {BASE_MODEL_PATH}")
    print("=" * 60)
    
    data = load_data()
    model = load_model()
    data = build_hard_negatives(data, model)
    save_data(data)
    
    print(f"\n下一步：重新运行微调")
    print(f"  python eval\\finetune_bge_m3.py")
