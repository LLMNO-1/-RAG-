"""
LangFuse 追踪管理器

提供 Trace / Span / Generation 三层追踪模型：
- Trace：一次完整的查询或导入管道执行
- Span：管道中各个节点的执行
- Generation：LLM 调用（含 Token 消耗和成本）
"""
import logging
import os
import threading
import time
from contextlib import contextmanager
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# 惰性导入，避免 langfuse 未安装时启动失败
_langfuse = None


def _get_langfuse():
    """惰性获取 LangFuse 客户端。"""
    global _langfuse
    if _langfuse is None:
        try:
            from langfuse import Langfuse
            _langfuse = Langfuse(
                host=os.getenv("LANGFUSE_HOST", "http://localhost:3000"),
                public_key=os.getenv("LANGFUSE_PUBLIC_KEY", ""),
                secret_key=os.getenv("LANGFUSE_SECRET_KEY", ""),
            )
        except ImportError:
            logger.warning("langfuse 未安装，追踪功能不可用")
            return None
        except Exception as e:
            logger.warning(f"LangFuse 初始化失败: {e}")
            return None
    return _langfuse


def is_trace_enabled() -> bool:
    """检查是否启用了追踪。"""
    return os.getenv("LANGFUSE_ENABLED", "false").lower() == "true"


class TraceManager:
    """追踪管理器（单例）。

    封装 LangFuse SDK，提供便捷的 Trace/Span/Generation 创建方法。
    """

    _instance: Optional["TraceManager"] = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._client = None
        self._enabled = is_trace_enabled()
        if self._enabled:
            self._client = _get_langfuse()
        self._initialized = True

    def create_trace(
        self,
        name: str,
        user_id: str = "",
        session_id: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Any]:
        """创建顶级 Trace。

        Args:
            name: Trace 名称（如 "query" / "import"）。
            user_id: 用户标识。
            session_id: 会话 ID。
            metadata: 附加元数据（task_id、original_query 等）。

        Returns:
            LangFuse Trace 对象，或 None（追踪未启用时）。
        """
        if not self._enabled or self._client is None:
            return None
        try:
            trace = self._client.trace(
                name=name,
                user_id=user_id,
                session_id=session_id,
                metadata=metadata or {},
            )
            return trace
        except Exception as e:
            logger.warning(f"创建 Trace 失败: {e}")
            return None

    def create_span(
        self,
        trace_id: str,
        name: str,
        input_data: Optional[Dict[str, Any]] = None,
        output_data: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        parent_span_id: Optional[str] = None,
    ) -> Optional[Any]:
        """创建 Span 并返回上下文管理器兼容对象。

        用作上下文管理器:
            with trace_mgr.span(trace_id, "node_name") as span:
                ...

        Args:
            trace_id: 父 Trace ID。
            name: Span 名称。
            input_data: 输入数据摘要。
            output_data: 输出数据摘要。
            metadata: 附加元数据。
            parent_span_id: 父 Span ID（可选）。

        Returns:
            _SpanContext 对象，或 _NoopSpan（追踪未启用时）。
        """
        if not self._enabled or self._client is None or not trace_id:
            return _NoopSpan()
        return _SpanContext(
            client=self._client,
            trace_id=trace_id,
            name=name,
            input_data=input_data,
            output_data=output_data,
            metadata=metadata,
            parent_span_id=parent_span_id,
        )

    def create_generation(
        self,
        trace_id: str,
        name: str,
        model: str,
        input_text: str = "",
        output_text: str = "",
        usage: Optional[Dict[str, int]] = None,
        cost: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Any]:
        """创建 LLM Generation Span。

        Args:
            trace_id: 父 Trace ID。
            name: Generation 名称。
            model: 模型名称。
            input_text: 输入文本（截断至 2000 字符）。
            output_text: 输出文本（截断至 2000 字符）。
            usage: Token 用量 {"input": N, "output": M}。
            cost: 预估成本（元）。
            metadata: 附加元数据。

        Returns:
            LangFuse Generation 对象，或 None。
        """
        if not self._enabled or self._client is None or not trace_id:
            return None
        try:
            generation = self._client.generation(
                trace_id=trace_id,
                name=name,
                model=model,
                input=input_text[:2000] if input_text else "",
                output=output_text[:2000] if output_text else "",
                usage=usage or {},
                cost=cost,
                metadata=metadata or {},
            )
            return generation
        except Exception as e:
            logger.warning(f"创建 Generation 失败: {e}")
            return None

    def flush(self):
        """确保所有待发送数据已提交。"""
        if self._client:
            try:
                self._client.flush()
            except Exception:
                pass

    def shutdown(self):
        """关闭客户端（LangFuse 内部会自动 flush）。"""
        if self._client:
            try:
                self._client.shutdown()
            except Exception:
                pass


class _SpanContext:
    """Span 上下文管理器。

    支持 with 语法，自动记录开始/结束时间和异常。
    """

    def __init__(
        self,
        client,
        trace_id: str,
        name: str,
        input_data: Optional[Dict[str, Any]] = None,
        output_data: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        parent_span_id: Optional[str] = None,
    ):
        self._client = client
        self._trace_id = trace_id
        self._name = name
        self._input_data = input_data
        self._output_data = output_data
        self._metadata = metadata or {}
        self._parent_span_id = parent_span_id
        self._span = None
        self._start_time = None

    def __enter__(self):
        self._start_time = time.time()
        try:
            self._span = self._client.span(
                trace_id=self._trace_id,
                name=self._name,
                input=self._input_data,
                metadata=self._metadata,
                parent_observation_id=self._parent_span_id,
            )
        except Exception:
            pass
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration_ms = round((time.time() - self._start_time) * 1000) if self._start_time else 0
        if self._span:
            try:
                self._span.update(
                    output=self._output_data,
                    metadata={**self._metadata, "duration_ms": duration_ms},
                )
                if exc_type is not None:
                    self._span.update(
                        level="ERROR",
                        status_message=str(exc_val)[:500],
                    )
                self._span.end()
            except Exception:
                pass
        return False  # 不吞异常

    def update(self, **kwargs):
        """更新 Span 属性。"""
        if self._span:
            try:
                self._span.update(**kwargs)
            except Exception:
                pass

    @property
    def id(self) -> Optional[str]:
        """获取 Span ID。"""
        if self._span:
            try:
                return self._span.id
            except Exception:
                pass
        return None


class _NoopSpan:
    """空 Span，追踪未启用时的占位符。"""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def update(self, **kwargs):
        pass

    @property
    def id(self):
        return None


# 全局单例
_trace_manager: Optional[TraceManager] = None


def get_trace_manager() -> TraceManager:
    """获取 TraceManager 全局单例。"""
    global _trace_manager
    if _trace_manager is None:
        _trace_manager = TraceManager()
    return _trace_manager
