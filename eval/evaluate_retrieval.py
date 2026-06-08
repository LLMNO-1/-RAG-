"""
检索质量评测脚本
=================
对比微调前后 BGE-M3 模型在全部 60 条 QA 上的检索表现。

用法：
    python eval/evaluate_retrieval.py

输出：
    eval/evaluation_report.csv  — 每条 query 的详细对比
    eval/evaluation_summary.txt — 汇总指标
"""

import json
import csv
import os
from pathlib import Path
import numpy as np
from collections import defaultdict

# ============================================================
# 配置
# ============================================================

ORIGINAL_MODEL_PATH = r"D:\models\bge_m3"
FINETUNED_MODEL_PATH = Path(__file__).parent / "finetuned_bge_m3" / "merged"
TRAIN_DATA_PATH = Path(__file__).parent / "training_data.json"
QA_DIR = Path(__file__).parent

REPORT_CSV = Path(__file__).parent / "evaluation_report.csv"
SUMMARY_TXT = Path(__file__).parent / "evaluation_summary.txt"

TOP_K_VALUES = [1, 3, 5, 10]


# ============================================================
# 步骤 1：加载模型
# ============================================================

def load_model(model_path, name):
    from FlagEmbedding.inference.embedder import BGEM3FlagModel
    model = BGEM3FlagModel(str(model_path), devices=["cuda:0"], use_fp16=True)
    print(f"  [{name}] 加载完成: {model_path}")
    return model


# ============================================================
# 步骤 2：加载评测数据
# ============================================================

def load_eval_data():
    """从 training_data.json 提取全部 60 条 QA，构建评测集"""
    with open(TRAIN_DATA_PATH, encoding="utf-8") as f:
        data = json.load(f)
    
    eval_samples = []
    for d in data:
        # 正例 chunk
        pos_chunk_id = d["positive"]["chunk_id"]
        pos_content = d["positive"]["content"]
        
        # 收集该文档的所有 chunk（从正例和负例中去重）
        doc_name = d["doc_name"]
        
        eval_samples.append({
            "query": d["query"],
            "doc_name": doc_name,
            "pos_chunk_id": pos_chunk_id,
            "pos_content": pos_content,
        })
    
    # 按文档分组 chunks
    doc_chunks = defaultdict(dict)
    for d in data:
        # 正例
        cid = d["positive"]["chunk_id"]
        doc_chunks[d["doc_name"]][cid] = d["positive"]["content"]
        # 负例
        for n in d["negatives"]:
            cid = n["chunk_id"]
            doc_chunks[n["doc_name"]][cid] = n["content"]
    
    print(f"  评测集: {len(eval_samples)} 条 query")
    print(f"  文档数: {len(doc_chunks)}")
    for doc, chunks in doc_chunks.items():
        print(f"    {doc}: {len(chunks)} chunks")
    
    return eval_samples, doc_chunks


# ============================================================
# 步骤 3：检索评测
# ============================================================

def compute_metrics(model, eval_samples, doc_chunks, model_name):
    """对每条 query 在同文档 chunks 中检索，计算指标"""
    import torch
    
    results = []
    
    # 预编码所有 chunks
    all_chunk_ids = []
    all_chunk_texts = []
    chunk_to_doc = {}
    for doc_name, chunks in doc_chunks.items():
        for cid, content in chunks.items():
            all_chunk_ids.append(cid)
            all_chunk_texts.append(content)
            chunk_to_doc[cid] = doc_name
    
    print(f"\n  [{model_name}] 编码 {len(all_chunk_texts)} 个 chunks...")
    chunk_embeddings = model.encode(all_chunk_texts, batch_size=16)
    chunk_dense = torch.from_numpy(np.array(chunk_embeddings["dense_vecs"])).float()
    chunk_dense = torch.nn.functional.normalize(chunk_dense, p=2, dim=1)
    
    print(f"  [{model_name}] 评测 {len(eval_samples)} 条 query...")
    
    # 构建 chunk_id -> index 映射
    cid_to_idx = {cid: i for i, cid in enumerate(all_chunk_ids)}
    
    for sample in eval_samples:
        query = sample["query"]
        doc_name = sample["doc_name"]
        pos_id = sample["pos_chunk_id"]
        
        # 编码 query
        q_result = model.encode_queries([query])
        q_dense = torch.from_numpy(np.array(q_result["dense_vecs"])).float()
        q_dense = torch.nn.functional.normalize(q_dense, p=2, dim=1)
        
        # 只在这个文档的 chunks 里检索（模拟项目中的商品名限定检索）
        doc_chunk_ids = [cid for cid in all_chunk_ids if chunk_to_doc[cid] == doc_name]
        doc_indices = [cid_to_idx[cid] for cid in doc_chunk_ids]
        doc_dense = chunk_dense[doc_indices]
        
        # 相似度排序
        sims = (q_dense @ doc_dense.T).squeeze(0)
        sorted_indices = torch.argsort(sims, descending=True)
        
        # 排名结果
        ranked_ids = [doc_chunk_ids[i] for i in sorted_indices.tolist()]
        ranked_sims = [sims[i].item() for i in sorted_indices.tolist()]
        
        # 找到正例的排名
        try:
            pos_rank = ranked_ids.index(pos_id) + 1  # 1-indexed
        except ValueError:
            pos_rank = len(ranked_ids) + 1  # 没找到
        
        # 正例相似度
        pos_sim = sims[doc_chunk_ids.index(pos_id)].item() if pos_id in doc_chunk_ids else 0
        
        # 最高负例相似度 (= 排名第一但不是正例的相似度)
        top_wrong_sim = ranked_sims[0] if ranked_ids[0] != pos_id else (ranked_sims[1] if len(ranked_sims) > 1 else 0)
        
        results.append({
            "query": query[:80],
            "doc_name": doc_name,
            "pos_rank": pos_rank,
            "pos_sim": round(pos_sim, 4),
            "top_wrong_sim": round(top_wrong_sim, 4),
            "num_candidates": len(doc_chunk_ids),
            **{f"hit@{k}": 1 if pos_rank <= k else 0 for k in TOP_K_VALUES},
            **{f"recall@{k}": 1 if pos_rank <= k else 0 for k in TOP_K_VALUES},
            f"mrr": round(1.0 / pos_rank, 4) if pos_rank <= len(doc_chunk_ids) else 0,
        })
    
    return results


# ============================================================
# 步骤 4：汇总指标
# ============================================================

def summarize(original_results, finetuned_results, eval_samples):
    """计算汇总指标并输出对比"""
    
    def compute_summary(results):
        n = len(results)
        metrics = {}
        for k in TOP_K_VALUES:
            metrics[f"Hit@{k}"] = sum(r[f"hit@{k}"] for r in results) / n
        metrics["MRR"] = sum(r["mrr"] for r in results) / n
        metrics["Avg Rank"] = sum(r["pos_rank"] for r in results) / n
        metrics["Avg Pos Sim"] = sum(r["pos_sim"] for r in results) / n
        metrics["Avg Top-Wrong Sim"] = sum(r["top_wrong_sim"] for r in results) / n
        metrics["Separation"] = metrics["Avg Pos Sim"] - metrics["Avg Top-Wrong Sim"]
        return metrics
    
    orig = compute_summary(original_results)
    ft = compute_summary(finetuned_results)
    
    return orig, ft


# ============================================================
# 步骤 5：输出报告
# ============================================================

def write_report(original_results, finetuned_results, orig_metrics, ft_metrics):
    """写入 CSV 详细报告和 TXT 汇总"""
    
    # ── CSV 详细报告 ──
    with open(REPORT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        headers = ["query", "doc_name", "candidates",
                   "orig_rank", "ft_rank", "rank_delta",
                   "orig_pos_sim", "ft_pos_sim",
                   "orig_top_wrong", "ft_top_wrong",
                   "orig_hit@1", "ft_hit@1",
                   "orig_hit@3", "ft_hit@3",
                   "orig_hit@5", "ft_hit@5",
                   "orig_hit@10", "ft_hit@10",
                   "orig_mrr", "ft_mrr"]
        writer.writerow(headers)
        
        for i in range(len(original_results)):
            o = original_results[i]
            f = finetuned_results[i]
            writer.writerow([
                o["query"], o["doc_name"], o["num_candidates"],
                o["pos_rank"], f["pos_rank"], o["pos_rank"] - f["pos_rank"],
                o["pos_sim"], f["pos_sim"],
                o["top_wrong_sim"], f["top_wrong_sim"],
                o["hit@1"], f["hit@1"],
                o["hit@3"], f["hit@3"],
                o["hit@5"], f["hit@5"],
                o["hit@10"], f["hit@10"],
                o["mrr"], f["mrr"],
            ])
    
    # ── 按文档汇总 ──
    by_doc_orig = defaultdict(list)
    by_doc_ft = defaultdict(list)
    for i in range(len(original_results)):
        doc = original_results[i]["doc_name"]
        by_doc_orig[doc].append(original_results[i])
        by_doc_ft[doc].append(finetuned_results[i])
    
    # ── TXT 汇总报告 ──
    lines = []
    lines.append("=" * 70)
    lines.append("BGE-M3 微调评测报告")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"评测集: {len(original_results)} 条 query, 5 个文档")
    lines.append(f"原始模型: {ORIGINAL_MODEL_PATH}")
    lines.append(f"微调模型: {FINETUNED_MODEL_PATH}")
    lines.append("")
    
    # 总体对比
    lines.append("-" * 70)
    lines.append(f"{'指标':<20s} {'原始模型':>12s} {'微调后':>12s} {'变化':>12s}")
    lines.append("-" * 70)
    
    comparisons = [
        ("Hit@1", "hit@1"), ("Hit@3", "hit@3"), ("Hit@5", "hit@5"), ("Hit@10", "hit@10"),
        ("MRR", "mrr"), ("Avg Rank", "avg_rank"), ("Avg Pos Sim", "avg_pos_sim"),
        ("Separation", "separation"),
    ]
    
    for label, key in comparisons:
        if key == "avg_rank":
            o_val = orig_metrics["Avg Rank"]
            f_val = ft_metrics["Avg Rank"]
            delta = o_val - f_val  # 排名下降是好事
        elif key == "separation":
            o_val = orig_metrics["Separation"]
            f_val = ft_metrics["Separation"]
            delta = f_val - o_val
        else:
            o_val = orig_metrics[key.replace("_", " ").title().replace("Avg ", "Avg ")] if key != "mrr" else orig_metrics["MRR"]
            o_val = orig_metrics.get(f"{key}".replace("_", " ").title()) if key not in ["mrr", "avg_rank", "separation"] else o_val
            f_val = ft_metrics.get(f"{key}".replace("_", " ").title()) if key not in ["mrr", "avg_rank", "separation"] else ft_metrics["MRR"] if key == "mrr" else 0
        
        # 直接用 metric dict
        metric_map = {
            "hit@1": ("Hit@1", True), "hit@3": ("Hit@3", True), "hit@5": ("Hit@5", True), "hit@10": ("Hit@10", True),
            "mrr": ("MRR", True), "avg_rank": ("Avg Rank", False), 
            "avg_pos_sim": ("Avg Pos Sim", True), "separation": ("Separation", True),
        }
        
        mkey, higher_better = metric_map.get(key, (key, True))
        o_val = orig_metrics[mkey]
        f_val = ft_metrics[mkey]
        delta = f_val - o_val
        
        arrow = "[UP]" if delta > 0 else "[DOWN]" if delta < 0 else " - "
        direction = "higher=better" if higher_better else "lower=better"
        
        lines.append(f"{label:<20s} {o_val:>12.4f} {f_val:>12.4f} {delta:>+11.4f} {arrow}")
    
    lines.append("-" * 70)
    lines.append("")
    lines.append("指标说明:")
    lines.append("  Hit@K   = 正确答案排在前K名的概率。例: Hit@1=0.95 表示95%的query第一个结果就是对的")
    lines.append("  MRR     = Mean Reciprocal Rank, 平均倒数排名。排第1得1.0, 第2得0.5, 越接近1越好")
    lines.append("  Avg Rank= 正确答案的平均排名, 越低越好, 1=所有query都一次命中")
    lines.append("  Avg Pos Sim = query与正例chunk的余弦相似度, 反映模型对正确答案的\"信心\"")
    lines.append("  Separation  = 正例相似度 - 最高负例相似度。越大越好, 说明能区分\"对的\"和\"像的\"")
    lines.append("               负数表示最相似的chunk不是正确答案 (模型被\"忽悠\"了)")
    lines.append("")
    
    # 按文档对比
    lines.append("按文档对比 (Hit@5 / MRR):")
    lines.append("-" * 70)
    lines.append(f"{'文档':<35s} {'原始 H@5':>8s} {'微调 H@5':>8s} {'原始 MRR':>8s} {'微调 MRR':>8s}")
    lines.append("-" * 70)
    
    for doc in sorted(by_doc_orig.keys()):
        o_items = by_doc_orig[doc]
        f_items = by_doc_ft[doc]
        n = len(o_items)
        o_h5 = sum(r["hit@5"] for r in o_items) / n
        f_h5 = sum(r["hit@5"] for r in f_items) / n
        o_mrr = sum(r["mrr"] for r in o_items) / n
        f_mrr = sum(r["mrr"] for r in f_items) / n
        lines.append(f"{doc:<35s} {o_h5:>8.4f} {f_h5:>8.4f} {o_mrr:>8.4f} {f_mrr:>8.4f}")
    
    lines.append("-" * 70)
    lines.append("")
    
    # 排名改进统计
    improved = sum(1 for i in range(len(original_results)) 
                   if finetuned_results[i]["pos_rank"] < original_results[i]["pos_rank"])
    same = sum(1 for i in range(len(original_results))
               if finetuned_results[i]["pos_rank"] == original_results[i]["pos_rank"])
    worse = sum(1 for i in range(len(original_results))
                if finetuned_results[i]["pos_rank"] > original_results[i]["pos_rank"])
    
    lines.append(f"排名改进: {improved} 条 | 持平: {same} 条 | 变差: {worse} 条")
    lines.append(f"改进率: {improved}/{len(original_results)} = {improved/len(original_results)*100:.1f}%")
    lines.append("")
    lines.append(f"详细报告: {REPORT_CSV}")
    lines.append("=" * 70)
    
    with open(SUMMARY_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    
    print("\n".join(lines))


# ============================================================
# 主入口
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("BGE-M3 微调评测")
    print("=" * 70)
    
    # 加载模型
    print("\n[1/4] 加载模型...")
    original_model = load_model(ORIGINAL_MODEL_PATH, "原始")
    finetuned_model = load_model(str(FINETUNED_MODEL_PATH), "微调")
    
    # 加载数据
    print("\n[2/4] 加载评测数据...")
    eval_samples, doc_chunks = load_eval_data()
    
    # 评测
    print("\n[3/4] 评测原始模型...")
    original_results = compute_metrics(original_model, eval_samples, doc_chunks, "原始")
    print(f"  原始 Hit@1: {sum(r['hit@1'] for r in original_results)/len(original_results):.3f}")
    
    print("\n[4/4] 评测微调模型...")
    finetuned_results = compute_metrics(finetuned_model, eval_samples, doc_chunks, "微调")
    print(f"  微调 Hit@1: {sum(r['hit@1'] for r in finetuned_results)/len(finetuned_results):.3f}")
    
    # 汇总
    orig_metrics, ft_metrics = summarize(original_results, finetuned_results, eval_samples)
    
    # 输出
    print("\n" + "=" * 70)
    write_report(original_results, finetuned_results, orig_metrics, ft_metrics)
