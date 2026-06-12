# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

**掌柜智库**（Shopkeeper Brain）：基于 LangGraph 多节点管道 + BGE-M3 微调嵌入 + 混合检索的企业级 RAG 系统，面向电子产品手册和技术文档场景。

- 语言：Python 3.12+
- 编排框架：LangGraph（状态图驱动多节点管道）
- Web 框架：FastAPI + Uvicorn（两个独立服务）
- 所有注释、日志、领域变量名使用中文；函数/类名使用英文

## 常用命令

```bash
# 安装依赖
cd knowledge && pip install -r requirements.txt
pip install FlagEmbedding transformers torch

# 启动基础设施
docker-compose up -d

# 启动导入服务（:8001）
cd knowledge && python api/import_router.py

# 启动查询服务（:8002）
cd knowledge && python api/query_router.py

# 运行测试（无 pytest，手动执行单个文件）
python knowledge/test/test_xxx.py

# BGE-M3 微调
python eval/finetune_bge_m3.py

# 检索质量评估
python eval/evaluate_retrieval.py

# 管道自测
python knowledge/processor/import_processor/main_graph.py
python knowledge/processor/query_processor/main_graph.py
```

**注意**：项目没有正式的构建系统、测试框架或 linter 配置。无 CI/CD。

## 架构

### 双服务架构

系统由两个独立的 FastAPI 服务组成，共享同一代码库：

- **导入服务** (`api/import_router.py`, `:8001`) — 文档上传与处理
- **查询服务** (`api/query_router.py`, `:8002`) — 用户提问与检索

### 导入管道（7 个 LangGraph 节点）

`processor/import_processor/main_graph.py` 编排状态图：

1. `entry_node` — 文件类型检测，路由 PDF/MD
2. `pdf_to_md_node` — MinerU CLI 子进程将 PDF 转 Markdown
3. `md_to_img_node` — 提取 MD 中图片，Qwen-VL 生成中文摘要
4. `document_split_node` — 按 Markdown 标题层级切分文档
5. `item_name_recognition_node` — LLM 提取商品名称型号
6. `embedding_chunks_node` — BGE-M3 生成稠密+稀疏向量（CUDA）
7. `import_milvus_node` — 向量及元数据写入 Milvus

PDF 文件经过全部 7 个节点，Markdown 文件跳过节点 2。

### 查询管道（8 个节点，含并行扇出）

`processor/query_processor/main_graph.py` 编排状态图：

1. `item_name_confirmed_node` — 商品名提取/确认 + 问题改写 + 指代消解
2. 条件分支：已有答案则跳到输出
3. **并行扇出**（3 路并发检索）：
   - `hybrid_vector_search_node` — Milvus 稠密+稀疏混合检索
   - `hyde_vector_search_node` — HyDE 假设文档嵌入检索
   - `web_mcp_search_node` — MCP 协议联网搜索（失败时优雅降级）
4. `rrf_merge_node` — RRF 倒数排序融合
5. `reranker_node` — BGE-Reranker 精排 + 动态断崖截断（sigmoid 归一化，gap ≥ 0.15）
6. `answer_output_node` — 上下文组装 + LLM 生成 + SSE 流式推送

### 核心模式

- **状态传播**：`TypedDict` 状态（`ImportGraphState` / `QueryGraphState`）流经所有节点
- **客户端单例**：`BaseClientManager._get_or_create()` 双重检查锁模式，避免重复初始化 GPU 模型和 DB 连接
- **依赖注入**：FastAPI `Depends()` + `core/deps.py` 中 `@cache` 工厂函数
- **配置管理**：`@dataclass` 配置对象，通过 `python-dotenv` 从 `.env` 加载
- **任务追踪**：内存字典追踪 task_id 的节点进度和耗时（`task_util.py`）
- **SSE 流式**：`queue.Queue` 桥接后台线程与异步 SSE 生成器（`sse_util.py`）

### 基础设施依赖

`docker-compose.yml` 运行：Milvus 2.5.5、MinIO、MongoDB 7.0、etcd、Attu。

## 关键配置文件

- `knowledge/.env` — 主配置（API 密钥、模型路径、DB 连接、集合名称），约 30 个环境变量
- `knowledge/processor/import_processor/config.py` — `ImportConfig` dataclass
- `knowledge/processor/query_processor/config.py` — `QueryConfig` dataclass（rerank top-k/gap、RRF 参数等）

## 注意事项

- BGE-M3 嵌入和 Reranker 模型需要 CUDA GPU（配置为 `cuda:0`）
- 任务追踪和 SSE 队列为内存状态，不支持多进程或分布式部署
- eval 脚本中存在硬编码的 Windows 路径（`D:\models\` 等）
- 没有依赖锁文件（`requirements.lock` / `poetry.lock`）
- 前端为原生 HTML/CSS/JS，由 FastAPI `StaticFiles` 托管
