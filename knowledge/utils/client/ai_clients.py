import threading
import os
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np
from scipy.sparse import csr_array
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from openai import OpenAI
from FlagEmbedding import BGEM3FlagModel, FlagReranker
from knowledge.utils.client.base import BaseClientManager, logger

# 从当前模块文件位置向上找到 knowledge/.env（适配任意 CWD）
_env_path = Path(__file__).resolve().parent.parent.parent / '.env'
load_dotenv(dotenv_path=_env_path)


class BGEM3EmbeddingFunction:
    """
    pymilvus 3.0 移除了 pymilvus.model 模块，这里用 FlagEmbedding 原生
    BGEM3FlagModel 封装，保持与旧 BGEM3EmbeddingFunction 相同的接口。

    旧接口（已移除）：
        from pymilvus.model.hybrid import BGEM3EmbeddingFunction
        model = BGEM3EmbeddingFunction(model_name=path, device="cuda:0", use_fp16=True)
        result = model.encode_documents(docs)  # {'dense': ndarray, 'sparse': csr_array}
    """

    def __init__(self, model_name: str, device: str = "cpu", use_fp16: bool = True):
        # BGEM3FlagModel 用 devices（复数列表），BGEM3EmbeddingFunction 用 device（单数字符串）
        devices = [device] if device else ["cpu"]
        self._model = BGEM3FlagModel(
            model_name_or_path=model_name,
            use_fp16=use_fp16,
            devices=devices,
        )

    def encode_documents(self, documents: List[str]) -> Dict[str, Any]:
        """与旧 pymilvus BGEM3EmbeddingFunction.encode_documents 接口一致"""
        output = self._model.encode(
            documents,
            batch_size=8,
            max_length=8192,
            return_dense=True,
            return_sparse=True,
        )
        # 将 lexical_weights (List[Dict[int,float]]) 转换回 csr_array
        sparse_csr = self._lexical_weights_to_csr(
            output["lexical_weights"],
            vocab_size=self._model.model.config.vocab_size,
        )
        return {
            "dense": output["dense_vecs"],
            "sparse": sparse_csr,
        }

    @staticmethod
    def _lexical_weights_to_csr(
        weights_list: List[Dict[int, float]], vocab_size: int
    ) -> csr_array:
        """将 FlagEmbedding 的 lexical_weights 转为 scipy csr_array"""
        indptr = [0]
        indices = []
        data = []
        for weights in weights_list:
            for token_id, weight in weights.items():
                indices.append(token_id)
                data.append(weight)
            indptr.append(len(indices))
        return csr_array((data, indices, indptr), shape=(len(weights_list), vocab_size))


class AIClients(BaseClientManager):
    """AI 模型类客户端"""

    _openai_client: Optional[OpenAI] = None
    _openai_lock = threading.Lock()

    _openai_llm_response_text_client: Optional[ChatOpenAI] = None
    _openai_llm_response_text_lock = threading.Lock()

    _openai_llm_response_json_client: Optional[ChatOpenAI] = None
    _openai_llm_response_json_lock = threading.Lock()

    _bge_m3_client: Optional[BGEM3EmbeddingFunction] = None
    _bge_m3_lock = threading.Lock()

    _bge_m3_rerank_client: Optional[FlagReranker] = None
    _bge_m3_rerank_lock = threading.Lock()

    # ── VLM ──

    @classmethod
    def get_vlm_client(cls) -> OpenAI:
        return cls._get_or_create("_openai_client", cls._openai_lock, cls._create_vlm_client)

    @classmethod
    def _create_vlm_client(cls) -> OpenAI:
        try:
            api_key = cls._require_env("OPENAI_API_KEY")
            base_url = cls._require_env("OPENAI_API_BASE")

            client = OpenAI(api_key=api_key, base_url=base_url)
            logger.info(f"OpenAI 客户端初始化成功 (base_url={base_url})")

            return client

        except EnvironmentError:
            raise
        except Exception as e:
            logger.error(f"OpenAI 客户端创建失败: {e}")
            raise ConnectionError(f"OpenAI 连接失败: {e}") from e

    # ── LLM ──
    @classmethod
    def get_llm_client(cls, response_format: bool = True) -> ChatOpenAI:
        if response_format:
            return cls._get_or_create("_openai_llm_json_client", cls._openai_llm_response_json_lock,
                                      lambda: cls._create_llm_client(response_format))
        else:
            return cls._get_or_create("_openai_llm_text_client", cls._openai_llm_response_text_lock,
                                      lambda: cls._create_llm_client(response_format))

            # ── LLM ──

    @classmethod
    def _create_llm_client(cls, response_format) -> ChatOpenAI:
        try:
            api_key = cls._require_env("OPENAI_API_KEY")
            base_url = cls._require_env("OPENAI_API_BASE")
            model_name = cls._require_env('LLM_DEFAULT_MODEL')

            model_kwargs = {}
            if response_format:
                model_kwargs['response_format'] = {"type": "json_object"}

            llm_client = ChatOpenAI(
                model_name=model_name,
                temperature=0,
                openai_api_key=api_key,
                openai_api_base=base_url,
                model_kwargs=model_kwargs
            )
            logger.info(f"OpenAI LLM 客户端初始化成功")
            return llm_client

        except EnvironmentError:
            raise
        except Exception as e:
            raise ConnectionError(f"OpenAI 连接失败: {e}") from e

    # ── BGE-M3嵌入模型客户端 ──
    @classmethod
    def get_bge_m3_client(cls):
        return cls._get_or_create("_bge_m3_client", cls._bge_m3_lock, cls._create_bge_m3_client)

    @classmethod
    def _create_bge_m3_client(cls):
        """
        创建bge_m3 客户端
        Returns:
        """

        try:
            # 1. 获取环境变量
            model_name = cls._require_env('BGE_M3_PATH')
            device = cls._require_env('BGE_DEVICE')
            fp16_str = cls._require_env('BGE_FP16')

            fp16 = fp16_str.lower() in ("true", "1")
            # 2. 创建
            bge_m3_ef = BGEM3EmbeddingFunction(
                model_name=model_name,
                device=device,
                use_fp16=fp16
            )
            return bge_m3_ef
        except EnvironmentError as e:
            raise

        except Exception as e:
            raise ConnectionError(f"BGE_M3嵌入模型客户端创建失败: {e}") from e

    # ── BGE-M3重排序模型客户端 ──
    @classmethod
    def get_bge_m3_rerank_client(cls):
        return cls._get_or_create("_bge_m3_rerank_client",
                                  cls._bge_m3_rerank_lock,
                                  cls._create_bge_m3_rerank_client)

    @classmethod
    def _create_bge_m3_rerank_client(cls):
        """
        创建bge_m3 重排序模型客户端
        Returns:
        """

        try:
            # 1. 获取环境变量
            model_name_or_path = cls._require_env('BGE_RERANKER_LARGE')
            device = cls._require_env('BGE_DEVICE')
            fp16_str = cls._require_env('BGE_FP16')
            fp16 = fp16_str.lower() in ("true", "1")

            # 兼容新版 transformers：prepare_for_model 被移除，monkey-patch 回来
            from transformers import XLMRobertaTokenizer
            if not hasattr(XLMRobertaTokenizer, 'prepare_for_model'):
                def _prepare_for_model(self, ids, pair_ids=None, add_special_tokens=True,
                                       padding=False, truncation=False,
                                       max_length=None, stride=0,
                                       return_tensors=None, **kwargs):
                    return {"input_ids": ids}
                XLMRobertaTokenizer.prepare_for_model = _prepare_for_model

            # 2. 创建
            reranker = FlagReranker(
                model_name_or_path=model_name_or_path,
                device=device,
                use_fp16=fp16
            )

            return reranker
        except EnvironmentError as e:
            raise

        except Exception as e:
            raise ConnectionError(f"BGE-M3重排序模型客户端创建失败: {e}") from e


if __name__ == '__main__':
    # llm_client: ChatOpenAI = AIClients.get_llm_client()
    #
    # llm_response = llm_client.invoke("请您给我讲一个笑话，要求输出格式是一个json")
    #
    # llm_result = llm_response.content
    #
    # import json
    #
    # result = json.loads(llm_result)
    #
    # print(result)
    #
    print(AIClients.get_bge_m3_rerank_client())
