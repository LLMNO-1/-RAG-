"""查询流程节点基类

定义统一的节点接口规范，提供通用功能。
"""
from abc import ABC, abstractmethod
from typing import TypeVar, Optional, Dict, Any
import logging
import time

from knowledge.processor.query_processor.config import QueryConfig, get_config
from knowledge.processor.query_processor.exceptions import QueryProcessError
from knowledge.utils.task_util import add_running_task, add_done_task, add_node_duration, get_task_status, get_running_task_list, \
    get_done_task_list, get_node_durations
from knowledge.utils.sse_util import push_sse_event, SSEEvent
from knowledge.utils.trace_util import get_trace_manager, is_trace_enabled

T = TypeVar("T")  # 泛型状态类型


class BaseNode(ABC):
    """查询流程节点基类。

    所有节点类都应继承此基类，实现 process 方法。
    基类提供统一的日志、任务追踪和错误处理。

    Attributes:
        name: 节点名称，子类应覆盖。
        config: 配置对象。
        logger: 日志记录器。

    Example:
        >>> class MyNode(BaseNode):
        ...     name = "my_node"
        ...
        ...     def process(self, state):
        ...         # 实现具体逻辑
        ...         return state
        ...
        >>> # 作为 LangGraph 节点使用
        >>> node = MyNode()
        >>> workflow.add_node("my_node", node)
    """

    name: str = "base_node"

    def __init__(self, config: Optional[QueryConfig] = None):
        """初始化节点。

        Args:
            config: 配置对象，默认使用全局配置。
        """
        self.config = config or get_config()
        self.logger = logging.getLogger(f"query.{self.name}")

    def __call__(self, state: T) -> T:
        """节点执行入口。

        LangGraph 调用节点时会调用此方法。
        提供统一的日志输出、任务追踪、计时、Trace Span 和异常处理。

        Args:
            state: 图状态字典。

        Returns:
            更新后的状态字典。

        Raises:
            QueryProcessError: 节点执行失败时抛出。
        """
        is_stream = state.get('is_stream')
        task_id = state.get('task_id')
        trace_id = state.get('trace_id', '')
        start_time = time.time()

        # 创建 Trace Span（自动在 with 块结束时关闭）
        trace_mgr = get_trace_manager()
        span_ctx = trace_mgr.create_span(
            trace_id=trace_id,
            name=self.name,
            input_data={"state_keys": list(state.keys())},
        )

        with span_ctx:
            try:
                self.logger.info(f"--- {self.name} 开始 ---")

                if task_id:
                    add_running_task(task_id, self.name)
                    if is_stream:
                        self._push_progress(task_id)

                result = self.process(state)

                if task_id:
                    duration = round(time.time() - start_time, 2)
                    add_done_task(task_id, self.name)
                    add_node_duration(task_id, self.name, duration)
                    if is_stream:
                        self._push_progress(task_id)

                span_ctx.update(output={"state_keys": list(result.keys())})

                self.logger.info(f"--- {self.name} 完成 ({duration}s) ---")
                return result
            except Exception as e:
                if task_id:
                    duration = round(time.time() - start_time, 2)
                    add_node_duration(task_id, self.name, duration)
                self.logger.error(f"{self.name} 执行失败: {e}")
                raise QueryProcessError(
                    message=str(e),
                    node_name=self.name,
                    cause=e
                )

    @abstractmethod
    def process(self, state: T) -> T:
        """节点核心处理逻辑。

        子类必须实现此方法。

        Args:
            state: 图状态字典。

        Returns:
            更新后的状态字典。
        """
        pass

    def log_step(self, step_name: str, message: str = ""):
        """记录步骤日志。

        Args:
            step_name: 步骤名称。
            message: 附加信息。
        """
        log_msg = f"[{step_name}]"
        if message:
            log_msg += f" {message}"
        self.logger.info(log_msg)

    def _push_progress(self, task_id):
        """
        推送节点的进度(全量推所有进度) -- 含节点耗时数据
        Args:
            task_id: 任务id
        """
        push_sse_event(task_id=task_id,
                       event=SSEEvent.PROGRESS,
                       data={
                           "status": get_task_status(task_id),
                           "done_list": get_done_task_list(task_id),
                           "running_list": get_running_task_list(task_id),
                           "durations": get_node_durations(task_id),
                       }
                       )

    def setup_logging(level: int = logging.INFO):
        """配置查询流程日志（JSON 格式）。

        Args:
            level: 日志级别，默认 INFO。
        """
        from knowledge.utils.log_util import setup_json_logging
        setup_json_logging(level)
