# 掌柜智库 (Shopkeeper Brain) — 企业级 RAG 智能知识库系统

基于 **LangGraph 多节点管道** + **BGE-M3 微调嵌入** + **混合检索** 的企业级 RAG（检索增强生成）系统，
支持 PDF/Markdown 文档智能导入、精准语义检索、联网搜索增强，面向电子产品手册、技术文档等场景。

> **项目级别对标**：独立完成此项目，在二三线互联网 / 传统企业可对标 **P7 ~ P8（技术主管 / 架构师）**，
> 在大厂 AI 应用方向可对标 **P6+ ~ P7**。核心依据：BGE-M3 模型微调、多节点 LangGraph 管道编排、
> 混合检索 + Reranker 精排、MCP 协议集成、多轮对话指代消解、VLM 多模态理解。
>
> **薪资参考（2025~2026 行业水平）**：
>
> - 大厂 P6+：40W ~ 55W / 年
> - 大厂 P7：55W ~ 80W / 年
> - 二三线 P7 ~ P8：45W ~ 70W / 年

---

## 核心亮点

- **BGE-M3 嵌入模型微调**：60 条训练样本，Recall@1 从 47% 提升至 95%，MRR 从 0.61 提升至 0.97
- **多路混合检索**：稠密向量 + 稀疏向量双路召回 → RRF 融合 → BGE-Reranker 精排 → 动态断崖截断
- **MCP 协议联网搜索**：对接阿里云 DashScope WebSearch，检索结果与本地知识库融合
- **多轮对话 + 指代消解**：MongoDB 持久化历史，支持"这个"→具体商品名消解
- **VLM 图片理解**：导入时自动识别文档中图片并生成中文摘要，嵌入检索上下文
- **SSE 流式输出**：支持非流式（一次性返回）和流式（逐字推送）两种模式
- **端到端可观测性**：LangFuse 全链路 Trace（节点耗时、LLM Token 消耗、检索延迟），JSON 结构化日志，实时指标仪表盘

---

## 系统架构

```
┌─────────────────────────────────────────────────────┐
│                    前端 (chat.html / import.html /   │
│                          dashboard.html)              │
│                  FastAPI StaticFiles 托管             │
└──────────┬──────────────────────────┬───────────────┘
           │                          │
    查询服务 :8002               导入服务 :8001
    (query_router.py)          (import_router.py)
    + /metrics/* 端点           + /metrics/* 端点
           │                          │
     ┌─────┴──────┐          ┌────────┴────────┐
     │ LangGraph   │          │  LangGraph       │
     │ 查询管道    │          │  导入管道        │
     │             │          │                  │
     │ ①商品确认   │          │ ①入口节点        │
     │ ②混合检索   │          │ ②PDF→MD(MinerU) │
     │ ③HyDE检索   │          │ ③MD→图片(VLM)   │
     │ ④WebSearch  │          │ ④文档切分        │
     │ ⑤RRF融合    │          │ ⑤商品名识别(LLM) │
     │ ⑥Reranker   │          │ ⑥向量嵌入(BGE)  │
     │ ⑦答案生成   │          │ ⑦Milvus入库      │
     └─────┬──────┘          └────────┬────────┘
           │                          │
     ┌─────┼──────────────┬───────────┤
     │     │              │           │
  Milvus  MongoDB    DashScope    MinIO      LangFuse
 (向量库) (对话历史) (LLM/嵌入)  (对象存储)  (可观测性)
                                                :3000
```

---

## 技术栈

| 层次 | 技术 |
|---|---|
| **编排框架** | LangGraph（状态图驱动多节点管道） |
| **LLM 服务** | 阿里云 DashScope（qwen-flash / qwen3-vl-flash），OpenAI 兼容接口 |
| **嵌入模型** | BGE-M3（微调版），混合稠密+稀疏向量，FlagEmbedding + PyMilvus |
| **重排序** | BGE-Reranker-v2-m3（FlagEmbedding） |
| **向量数据库** | Milvus 3.x（混合检索 + COSINE/IP 度量） |
| **对话历史** | MongoDB（会话持久化、上下文注入） |
| **对象存储** | MinIO（原始文件持久化） |
| **PDF 解析** | MinerU（PDF → Markdown，支持表格、公式） |
| **Web 框架** | FastAPI + Uvicorn + SSE 流式推送 |
| **MCP 协议** | openai-agents + MCPServerStreamableHttp（DashScope WebSearch） |
| **可观测性** | LangFuse 全链路追踪 + JSON 结构化日志 + Chart.js 仪表盘 |
| **前端** | 原生 HTML/CSS/JS（无框架依赖，暗色主题） |

---

## 项目结构

```
shopkeeper_brain/
├── knowledge/                          # 核心知识库模块
│   ├── api/                            # FastAPI 路由
│   │   ├── import_router.py            # 导入服务（:8001）
│   │   ├── query_router.py             # 查询服务（:8002）
│   │   └── metrics_router.py           # 指标 API（/metrics/*）
│   ├── core/                           # 核心配置
│   │   ├── deps.py                     # 依赖注入
│   │   └── paths.py                    # 路径管理
│   ├── front/                          # 前端页面
│   │   ├── import.html                 # 文件上传页面
│   │   ├── chat.html                   # 对话查询页面
│   │   └── dashboard.html              # 可观测性仪表盘
│   ├── processor/                      # LangGraph 管道
│   │   ├── import_processor/           # 导入管道
│   │   │   ├── main_graph.py           # 图谱编排
│   │   │   ├── nodes/                  # 7 个节点
│   │   │   ├── state.py / config.py    # 状态与配置
│   │   │   └── exceptions.py           # 异常定义
│   │   └── query_processor/            # 查询管道
│   │       ├── main_graph.py           # 图谱编排
│   │       ├── nodes/                  # 8 个节点
│   │       ├── state.py / config.py    # 状态与配置
│   │       └── exceptions.py           # 异常定义
│   ├── prompts/                        # LLM 提示词模板
│   ├── schema/                         # Pydantic 数据模型
│   ├── service/                        # 业务服务层
│   ├── utils/                          # 工具函数
│   │   ├── client/                     # AI/存储客户端（单例+双重检查锁）
│   │   │   ├── ai_clients.py           # LLM/VLM/BGE 客户端
│   │   │   ├── storage_clients.py      # MinIO/Milvus 客户端
│   │   │   └── base.py                 # 客户端管理器基类
│   │   ├── embedding_util.py           # 嵌入工具
│   │   ├── milvus_util.py              # Milvus 混合搜索
│   │   ├── mongo_history_util.py       # MongoDB 对话历史
│   │   ├── sse_util.py                 # SSE 流式推送
│   │   ├── task_util.py                # 任务状态追踪
│   │   ├── trace_util.py               # LangFuse Trace 管理器
│   │   ├── metrics_util.py             # 指标聚合器
│   │   └── log_util.py                 # JSON 结构化日志
│   └── test/                           # 测试用例
├── eval/                               # 模型评估
│   ├── finetune_bge_m3.py              # LoRA 微调脚本
│   ├── build_hard_negatives.py         # 难负例构建
│   ├── evaluate_retrieval.py           # 检索评估
│   ├── evaluation_summary.txt          # 评估报告（95% Recall@1）
│   ├── finetuned_bge_m3/               # 微调后的模型
│   └── *_qa.csv                        # 按文档分拆的 QA 数据
└── 项目环境配置&服务部署指南.md          # 部署文档
```

---

## 导入管道（Import Pipeline）

文档导入经过 7 个 LangGraph 节点的状态图管道：

| 节点 | 功能 | 核心技术 |
|---|---|---|
| **entry_node** | 判断文件类型（PDF/MD），路由分支 | 文件扩展名检测 |
| **pdf_to_md_node** | PDF 转 Markdown | MinerU CLI（子进程调用） |
| **md_to_img_node** | 提取 MD 中引用的图片，VLM 生成摘要 | Qwen-VL（base64 编码图片） |
| **document_split_node** | 文档切片（基于 Markdown 标题层级） | 自定义 Markdown Splitter |
| **item_name_recognition_node** | LLM 提取商品名称型号 | Qwen-Flash + Prompt Engineering |
| **embedding_chunks_node** | BGE-M3 批量生成稠密+稀疏向量 | BGEM3FlagModel（CUDA） |
| **import_milvus_node** | 向量及元数据写入 Milvus | MilvusClient（HYBRID 集合） |

---

## 查询管道（Query Pipeline）

用户提问经过 8 个节点的状态图管道：

| 节点 | 功能 | 核心技术 |
|---|---|---|
| **item_name_confirmed_node** | 提取/确认商品名 + 问题改写 + 指代消解 | LLM + 对话历史 |
| **hybrid_vector_search_node** | 稠密+稀疏混合向量检索 | Milvus hybrid_search + 商品名过滤 |
| **hyde_vector_search_node** | HyDE（假设文档嵌入）增强检索 | LLM 生成假设文档 → BGE-M3 嵌入 → 检索 |
| **web_mcp_search_node** | MCP 协议联网搜索 | DashScope WebSearch |
| **rrf_merge_node** | RRF（倒数排序融合）多路结果 | Reciprocal Rank Fusion |
| **reranker_node** | BGE-Reranker 精排 + 动态断崖截断 | FlagReranker + sigmoid 归一化 |
| **answer_output_node** | 上下文组装 + LLM 生成答案 + SSE 推送 | Qwen-Flash + 对话历史注入 |
| (流式) **SSE 推送** | 逐字推送 + 进度事件 | SSEEvent.DELTA / PROGRESS / FINAL |

---

## BGE-M3 微调效果

| 指标 | 原始模型 | 微调后 | 提升 |
|---|---|---|---|
| **Recall@1** | 46.7% | **95.0%** | +48.3% |
| **Recall@3** | 66.7% | **98.3%** | +31.7% |
| **Recall@5** | 83.3% | **100.0%** | +16.7% |
| **MRR** | 0.614 | **0.971** | +58.1% |
| **平均排名** | 3.18 | **1.08** | -2.10 |

- 训练数据：60 条 QA + 180 条难负例
- 方法：LoRA 微调（target_modules=["query","key","value","dense"]）
- 60 条问题无退化，53% 进一步提升

---

## 快速开始

### 环境要求

- Python 3.12+
- CUDA GPU（推荐 RTX 2060 6GB 及以上）
- Docker（用于运行 Milvus、MinIO、MongoDB）

### 1. 安装依赖

```bash
cd knowledge
pip install -r requirements.txt
pip install FlagEmbedding transformers torch
```

### 2. 配置环境变量

复制并编辑 `knowledge/.env`：

```ini
# LLM API（阿里云 DashScope）
OPENAI_API_KEY=***
OPENAI_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_DEFAULT_MODEL=qwen-flash
VL_MODEL=qwen3-vl-flash

# BGE 模型路径
BGE_M3_PATH=D:\课程视频\...\eval\finetuned_bge_m3\merged
BGE_RERANKER_LARGE=D:\models\bge-reranker-v2-m3
BGE_DEVICE=cuda:0
BGE_FP16=True

# Milvus
MILVUS_URL=http://192.168.10.140:19530
CHUNKS_COLLECTION=kb_chunks_v1
ITEM_NAME_COLLECTION=kb_item_names_v1

# MongoDB
MONGO_URL=mongodb://admin:password@192.168.10.140:27017
MONGO_DB_NAME=kb001

# MinIO
MINIO_ENDPOINT=192.168.10.140:9000

# MCP 联网搜索
MCP_DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/api/v1/mcps/WebSearch/mcp

# LangFuse 可观测性
LANGFUSE_ENABLED=true
LANGFUSE_HOST=http://192.168.10.140:3000
LANGFUSE_PUBLIC_KEY=pk-xxxxx
LANGFUSE_SECRET_KEY=sk-xxxxx
```

### 3. 启动基础服务（Docker Compose）

```bash
# 启动所有基础设施容器（Milvus、MinIO、MongoDB、etcd、Attu、LangFuse）
docker compose up -d
```

### 容器访问地址总览

| 服务 | 地址 | 说明 |
|---|---|---|
| **Attu**（Milvus 管理） | `http://192.168.10.140:7000` | Milvus 图形化管理界面 |
| **MinIO Console** | `http://192.168.10.140:9001` | 对象存储管理 |
| **LangFuse** | `http://192.168.10.140:3000` | LLM 可观测性平台（Trace/Token/成本） |
| **导入服务** | `http://localhost:8001` | 文档上传页面 |
| **查询服务** | `http://localhost:8002` | 对话查询页面 |
| **指标仪表盘** | `http://localhost:8002/front/dashboard.html` | 实时请求统计和节点耗时 |
| **指标 API** | `http://localhost:8002/metrics/overview` | JSON 格式汇总指标 |

### 4. 启动应用

```bash
# 终端 1 — 导入服务
cd knowledge
python api/import_router.py
# → http://localhost:8001/front/import.html

# 终端 2 — 查询服务
python api/query_router.py
# → http://localhost:8002/front/chat.html
```

### 5. 使用流程

1. 打开导入页面，上传 PDF/Markdown 文档
2. 等待后台处理完成（状态轮询）
3. 打开对话页面，输入问题查询

---

## 关键设计

### 客户端单例模式

所有 AI 和存储客户端采用**双重检查锁**单例模式，避免重复初始化模型和数据库连接：

```python
@classmethod
def _get_or_create(cls, attr_name, lock, factory):
    instance = getattr(cls, attr_name, None)
    if instance is not None:
        return instance
    with lock:
        if getattr(cls, attr_name, None) is None:
            setattr(cls, attr_name, factory())
    return getattr(cls, attr_name)
```

### 动态 Top-K 断崖截断

Reranker 精排后不是固定取 Top-K，而是通过 sigmoid 归一化分数，在排序结果中寻找相邻文档间的**最大分数断崖**（gap ≥ 0.15），在断崖处截断，兼顾相关性和噪声过滤。

### HyDE（假设文档嵌入）

用户问题先让 LLM 生成一段"假设的技术文档片段"，再用 BGE-M3 对该片段做向量检索。这种方法对口语化、不规范的查询有显著提升效果。

### MCP 联网搜索降级

联网搜索采用 `MCPServerStreamableHttp` 协议对接 DashScope WebSearch，失败时优雅降级为仅本地检索，不影响核心功能。

---

## 后续规划

- [x] Docker Compose 一键部署
- [x] 端到端可观测性（LangFuse 全链路追踪 + 指标仪表盘）
- [ ] Agent 工具调用体系（ReAct + 自定义 Tool + MCP 网关）
- [ ] 知识图谱集成（实体关系抽取 + GraphRAG）
- [ ] 语义缓存（GPTCache / Redis 向量缓存，降低 LLM 调用成本）
- [ ] 管理后台（文档管理、索引监控）
- [ ] 私有化部署方案（vLLM + 模型量化 + 国产 GPU 适配）
