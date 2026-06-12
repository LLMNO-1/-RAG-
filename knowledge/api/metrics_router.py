"""可观测性指标 API

提供指标查询端点，供仪表盘和前端消费。
"""
from fastapi import APIRouter, Query

from knowledge.utils.metrics_util import get_metrics_collector

router = APIRouter(prefix="/metrics", tags=["metrics"])


@router.get("/overview")
async def metrics_overview():
    """获取汇总仪表盘数据。"""
    collector = get_metrics_collector()
    return collector.get_overview()


@router.get("/nodes")
async def node_metrics():
    """获取各节点耗时统计。"""
    collector = get_metrics_collector()
    return collector.get_node_metrics()


@router.get("/recent")
async def recent_tasks(limit: int = Query(default=20, le=100)):
    """获取最近 N 条任务的摘要列表。"""
    collector = get_metrics_collector()
    return collector.get_recent_tasks(limit=limit)


@router.get("/trace/{task_id}")
async def task_trace(task_id: str):
    """获取指定任务的完整 Trace 摘要。"""
    collector = get_metrics_collector()
    result = collector.get_task_trace(task_id)
    if result["status"] == "unknown":
        return {"error": "任务不存在", "task_id": task_id}
    return result
