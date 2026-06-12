"""指标聚合器

基于内存中的任务追踪数据（task_util.py）聚合关键指标：
- 请求量 / 成功率 / 延迟分布
- 节点耗时统计
- 最近任务列表
"""
import time
from typing import Dict, List, Any
from statistics import mean, median, quantiles

from knowledge.utils.task_util import (
    _tasks_status,
    _tasks_duration,
    _tasks_running_list,
    _tasks_done_list,
    TASK_STATUS_COMPLETED,
    TASK_STATUS_FAILED,
    TASK_STATUS_PROCESSING,
)


class MetricsCollector:
    """从内存字典聚合指标。"""

    def __init__(self):
        self._start_time = time.time()

    def get_overview(self) -> Dict[str, Any]:
        """返回汇总仪表盘数据。"""
        total = len(_tasks_status)
        completed = sum(1 for s in _tasks_status.values() if s == TASK_STATUS_COMPLETED)
        failed = sum(1 for s in _tasks_status.values() if s == TASK_STATUS_FAILED)
        processing = sum(1 for s in _tasks_status.values() if s == TASK_STATUS_PROCESSING)

        # 收集所有任务的总耗时
        all_durations = []
        for durations_dict in _tasks_duration.values():
            total_dur = sum(durations_dict.values())
            if total_dur > 0:
                all_durations.append(total_dur)

        p50 = p95 = p99 = 0
        if len(all_durations) >= 2:
            sorted_durs = sorted(all_durations)
            p50 = sorted_durs[len(sorted_durs) // 2]
            p95 = sorted_durs[int(len(sorted_durs) * 0.95)]
            p99 = sorted_durs[int(len(sorted_durs) * 0.99)]
        elif len(all_durations) == 1:
            p50 = p95 = p99 = all_durations[0]

        return {
            "total_requests": total,
            "completed": completed,
            "failed": failed,
            "processing": processing,
            "success_rate": round(completed / total * 100, 1) if total > 0 else 0,
            "latency_p50_s": round(p50, 2),
            "latency_p95_s": round(p95, 2),
            "latency_p99_s": round(p99, 2),
            "uptime_s": round(time.time() - self._start_time),
        }

    def get_node_metrics(self) -> Dict[str, Dict[str, float]]:
        """返回各节点的耗时统计。

        Returns:
            {"node_name": {"count": N, "avg_s": X, "max_s": Y, "min_s": Z}, ...}
        """
        node_stats: Dict[str, List[float]] = {}
        for durations_dict in _tasks_duration.values():
            for node_name, duration in durations_dict.items():
                if node_name not in node_stats:
                    node_stats[node_name] = []
                node_stats[node_name].append(duration)

        result = {}
        for node_name, durs in node_stats.items():
            result[node_name] = {
                "count": len(durs),
                "avg_s": round(mean(durs), 2),
                "max_s": round(max(durs), 2),
                "min_s": round(min(durs), 2),
                "p50_s": round(median(durs), 2),
            }
        return result

    def get_recent_tasks(self, limit: int = 20) -> List[Dict[str, Any]]:
        """返回最近 N 条任务的摘要列表。"""
        tasks = []
        for task_id, status in list(_tasks_status.items())[-limit:]:
            durations = _tasks_duration.get(task_id, {})
            total_dur = sum(durations.values())
            tasks.append({
                "task_id": task_id,
                "status": status,
                "total_duration_s": round(total_dur, 2),
                "node_count": len(durations),
                "nodes": list(durations.keys()),
            })
        return tasks

    def get_task_trace(self, task_id: str) -> Dict[str, Any]:
        """返回单次任务的详细信息（Trace 摘要）。"""
        return {
            "task_id": task_id,
            "status": _tasks_status.get(task_id, "unknown"),
            "durations": _tasks_duration.get(task_id, {}),
            "done_list": [
                n for n in _tasks_done_list.get(task_id, [])
            ],
            "running_list": [
                n for n in _tasks_running_list.get(task_id, [])
            ],
        }


# 全局单例
_collector: MetricsCollector = None


def get_metrics_collector() -> MetricsCollector:
    global _collector
    if _collector is None:
        _collector = MetricsCollector()
    return _collector
