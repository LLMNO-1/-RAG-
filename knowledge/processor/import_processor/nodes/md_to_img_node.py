import re, time, base64
from collections import deque
from dataclasses import dataclass
from logging import Logger
from typing import List, Dict, Tuple, Set, Optional, Deque
from pathlib import Path

from openai import OpenAI

from knowledge.processor.import_processor.base import BaseNode, setup_logging, T
from knowledge.processor.import_processor.state import ImportGraphState
from knowledge.processor.import_processor.exceptions import StateFieldError, FileProcessingError
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.storage_clients import StorageClients


@dataclass  # 未来直接实例化（不需要重写__init__方法 __repr__方法）
class ImageContext:
    """
    一张图片上下文信息

    """
    head: str  # 上文标题内容
    pre_text: str  # 上文内容
    post_text: str  # 下文内容


@dataclass
class ImageInfo:
    """
    一张图片的完整信息
    图片的名字：作为存储图片摘要的字典容器key
    图片的地址：1.vlm要用【xx.png/xxx.png的内容】 2. minio要用
    图片上下文信息：作为VLM使用
    """
    name: str  # 图片的名字（全名）
    path: str  # 图片地址
    imag_context: ImageContext  # 图片上下文信息


class _MdFileHandler:
    """
    主要职责：
    1. 读取md内容、md_path、图片目录
    2. 备份新的md_content.(方便测试观察)
    """

    def __init__(self, logger: Logger, node_name: str):
        self.logger = logger
        self.node_name = node_name

    def validate_and_read_md(self, state) -> Tuple[str, Path, Path]:
        """
        核心逻辑：
        1. 读取md内容
        2. 读取md的路径
        3. 读取图片目录
        Args:
            state:  上一个节点更新后的state

        Returns:
            Tuple[str,Path,Path]

        """

        # 1. 从state中获取md_path
        md_path = state.get('md_path', '')

        # 2. 非空判断
        if not md_path:
            raise StateFieldError(node_name=self.node_name, field_name='md_path', expected_type=str)

        # 3. Path标准化
        md_path_obj = Path(md_path)

        # 4. 判断路径是否存在
        if not md_path_obj.exists():
            raise StateFieldError(node_name=self.node_name, field_name='md_path', expected_type=Path)

        # 5. 读取md_content
        try:
            with open(md_path_obj, 'r', encoding='utf-8') as f:
                md_content = f.read()
        except IOError as e:
            self.logger.error(f"MD文件:{md_path_obj.name} 打开失败")
            raise FileProcessingError(message="文件打开失败", node_name=self.node_name)

        # 6. 获取图片目录
        img_dir = md_path_obj.parent / "images"

        # 7. 返回
        return md_content, md_path_obj, img_dir

    def backup(self, md_path_obj: Path, new_md_content: str) -> str:
        self.logger.info("【step_5】备份新文件")

        new_file_path = md_path_obj.with_name(
            f"{md_path_obj.stem}_new{md_path_obj.suffix}"
        )
        try:
            with open(new_file_path, "w", encoding="utf-8") as f:
                f.write(new_md_content)
            self.logger.info(f"处理后的文件已备份至: {new_file_path}")
        except IOError as e:
            self.logger.error(f"写入新文件失败 {new_file_path}: {e}")
            raise FileProcessingError(
                f"文件写入失败: {e}", node_name="md_img_node"
            )
        return str(new_file_path)


class _ImageScanner:
    """
    主要职责：
    1. 根据图片目录，得到该目录下有效的图片文件
    2. 去到md文件中定位图片的位置
    3. 获取该图片在md中的上下文内容（给VLM模型提供上下文信息，帮助 模型识别结果更加准确）
    4. 最终组装所有图片的上下文内容（List）

    """

    def __init__(self, logger: Logger):
        self.logger = logger

    def scan_imgs_dir(self, img_dir_obj: Path, md_content: str, image_extensions: Set[str], img_content_length: int) -> \
            List[ImageInfo]:
        """
        核心逻辑：
        1. 扫描指定图片目录下的所有图片文件

        2. 遍历每一个图片文件去MD文件中获取到位置（上下文）
        2.1 上文信息（标题 + 上文内容）
        2.2 下文信息（下文内容）

        3. 将每一个（图片的上下文：ImageContext）放到最终封装每一个图片完整信息(ImageInfo)的容器中

        4. 将容器返回
        Args:
            img_dir_obj: 图片目录
            md_content:  md内容
            image_extensions: 允许的图片后缀格式
            img_content_length: 上下文的长度（各自最大不能超过200）
        Returns:
            List[ImageInfo]

        """
        img_info_list = []
        # 1. 遍历图片目录
        for img_path in img_dir_obj.iterdir():

            # 1.1 过滤掉子目录
            if not img_path.is_file():
                self.logger.error(f"{img_path}不是一个有效的文件")
                continue

            # 1.2  过滤掉不合法的图片文件
            if not img_path.suffix in image_extensions:
                self.logger.error(f"{img_path.suffix}不是允许的图片后缀格式")
                continue

            # 1.3 找该图片的上下文
            ctx = self._find_context(img_path.name, md_content, img_content_length)
            if not ctx:
                self.logger.info(f"MD中未找到该图片{img_path.name}引用")
                continue

            # 1.4 封装ImageInfo对象并且放到容器中
            img_info_list.append(ImageInfo(
                name=img_path.name,
                path=str(img_path),
                imag_context=ctx
            ))

        self.logger.info(f"MD中找到{len(img_info_list)}个有效的图片引用")

        # 2. 最终返回
        return img_info_list

    def _find_context(self, img_name: str, md_content: str, img_content_length: int) -> Optional[ImageContext]:
        """
        查找图片的上下文
        Args:
            img_name: 图片名
            md_content: MD内容
            img_content_length:上下文长度

        Returns:
          找到了--->ImageContext:图片的上下文信息
          没找到--->None
        """

        # 1. 预编译正则规则(主要目的：从MD（很多行）中抓取到当前这个图片)
        # ![](images\xxx.png "abc")
        # 正则在大模型应用中特别多
        # . 任意字符 * 0次或者多次  \[ \] \( \) ?非贪婪模式  escape（a.png）
        pattern = re.compile(r"!\[.*?\]\(.*?" + re.escape(img_name) + r".*?\)")

        # 2. 按行切割md_content
        md_lines = md_content.split("\n")

        # 3. 遍历每一行以及对应的行索引
        for md_idx, md_line in enumerate(md_lines):

            # 3.1 当前行不是当前图片
            if not pattern.search(md_line):
                continue

            # 3.2 当前行包含当前图片
            # 上文
            # 上文标题的索引作为起始索引(取不到)
            head, prev_index = self._find_heading_up(md_lines, md_idx)
            pre_lines = md_lines[prev_index + 1:md_idx]
            pre_context = self._extract_limited_context(pre_lines, img_content_length, direction="front")

            # 下文
            # 下文标题的索引作为结束索引
            next_index = self._find_heading_down(md_lines, md_idx)
            next_lines = md_lines[md_idx + 1:next_index]
            post_context = self._extract_limited_context(next_lines, img_content_length, direction="back")

            return ImageContext(
                head=head,
                pre_text=pre_context,
                post_text=post_context
            )
        return None

        # ==========================================================================
        # 【改进二】同一张图片在 MD 中多次引用时，只处理了第一次
        # ==========================================================================
        #
        # ▎问题：
        #   原代码在 for 循环中找到第一处匹配就直接 return 了。
        #   如果同一张图片在文档中被引用了多次（例如正文中引用一次、
        #   总结章节又引用一次），只有第一次的上下文会被提取。
        #   但 _update_md 会把所有引用都替换成同一个摘要，
        #   导致后面几处引用的上下文信息丢失。
        #
        # ▎改进思路：
        #   方案A：找到所有引用位置，分别提取各自的上下文，
        #          然后合并或选择最丰富的一组作为 VLM 的输入。
        #   方案B（更简单）：如果多处引用，优先选择上下文最完整的那一处。
        #
        # ▎改进代码（方案A - 收集所有引用位置，选上下文最丰富的）：
        #
        #   def _find_context(self, img_name, md_content, img_content_length):
        #       pattern = re.compile(
        #           r"!\[.*?\]\(.*?" + re.escape(img_name) + r".*?\)"
        #       )
        #       md_lines = md_content.split("\n")
        #
        #       # 收集所有匹配位置的索引
        #       match_indices = []
        #       for md_idx, md_line in enumerate(md_lines):
        #           if pattern.search(md_line):
        #               match_indices.append(md_idx)
        #
        #       if not match_indices:
        #           return None
        #
        #       # 为每个匹配位置提取上下文，选内容最丰富的一组
        #       best_context = None
        #       best_length = -1
        #
        #       for md_idx in match_indices:
        #           head, prev_index = self._find_heading_up(md_lines, md_idx)
        #           pre_lines = md_lines[prev_index + 1:md_idx]
        #           pre_context = self._extract_limited_context(
        #               pre_lines, img_content_length, direction="front"
        #           )
        #
        #           next_index = self._find_heading_down(md_lines, md_idx)
        #           next_lines = md_lines[md_idx + 1:next_index]
        #           post_context = self._extract_limited_context(
        #               next_lines, img_content_length, direction="back"
        #           )
        #
        #           # 计算这组上下文的总长度
        #           total_len = len(head) + len(pre_context) + len(post_context)
        #           if total_len > best_length:
        #               best_length = total_len
        #               best_context = ImageContext(
        #                   head=head,
        #                   pre_text=pre_context,
        #                   post_text=post_context
        #               )
        #
        #       return best_context
        #
        # ▎关键区别：
        #   原代码：找到第一处 → 直接 return（可能上下文很少）
        #   改进后：找到所有处 → 比较上下文长度 → 返回最丰富的那组
        #   例如：第一次引用在文档开头（上文很少），第二次引用在详细说明段落
        #         （上下文很丰富），改进后会选择第二次的上下文给 VLM
        # ==========================================================================

    def _find_heading_up(self, md_lines: List[str], from_idx: int) -> Tuple[str, int]:
        """

        Args:
            md_lines: 整个MD内容
            from_idx: 图片的索引

        Returns:
          当前图片最近的上文标题内容+索引
        """
        for i in range(from_idx - 1, -1, -1):
            if re.match(r"^#{1,6}\s+", md_lines[i]):
                return md_lines[i], i

        return "", -1

    def _find_heading_down(self, md_lines: List[str], from_idx: int) -> int:
        """

        Args:
            md_lines:  整个MD内容
            from_idx:   图片的索引

        Returns:
              当前图片最近的下文标题索引
        """
        for i in range(from_idx + 1, len(md_lines)):
            if re.match(r"^#{1,6}\s+", md_lines[i]):
                return i

        return len(md_lines)

    def _extract_limited_context(self, extracted_md_lines: List[str], img_content_length: int, direction: str) -> str:
        """
        职责：截取给定的上下文内容
        截取策略：不直接根据字符数暴力截取，采用段落方式截取。最后根据段落的字符数是否达到最大上下文长度选择留取。
        段落的规则：
        ①：自然而然的段落 获取切分后的内容 如果是""空字符串
        ②：人为设计其他图片作为段落（其它图片不要）
        Args:
            extracted_md_lines: 上（下）文
            img_content_length: 上下文长度
            direction: 方向(向上找)

        Returns:
            str:上（下）文的内容
        """
        current_paragraph = []
        paragraphs = []

        # 1. 遍历截取的行
        for line in extracted_md_lines:
            # 1.1 定义自然而然段落的规则
            is_blank_line = not line.strip()

            # 1.2 定义人为设计的图片段落规则
            is_other_image = re.match(
                r"^!\[.*?\]\(.*?\)$", line.strip()
            )

            # 1.3 当前行是空行或者其它图片行
            if is_blank_line or is_other_image:
                if current_paragraph:
                    paragraphs.append("\n".join(current_paragraph))
                    current_paragraph = []
                continue

            # 1.4  当前行不是空行也不是其它图片行
            current_paragraph.append(line)

        # 2. 处理最后的行
        if current_paragraph:
            paragraphs.append("\n".join(current_paragraph))

        # 反转(就近原则)
        if direction == "front":
            paragraphs.reverse()
        # 3. 遍历段落列表(判断长度，已经最终选择留下哪些段落)
        total = 0
        selected = []  # 最终收集到的段落
        for paragraph in paragraphs:
            if total + len(paragraph) > img_content_length and selected:
                break
            selected.append(paragraph)
            total += len(paragraph)

        # 反转（保证收集到的顺序和原文档中顺序一致，方便VLM参考）
        if direction == "front":
            selected.reverse()

        # 4. 将最终段落列表中的段落转成一个字符串
        return "\n\n".join(selected)


class _VLMSummarizer:
    """
    主要职责：
    主要根据每一张图片信息以及每一张图片的上下文信息，生成对应该图片的摘要信息
    """

    def __init__(self, logger: Logger, requests_per_minute: int):
        self.logger = logger
        self.requests_per_minute = requests_per_minute

    def _summary_all(self, document_name: str, img_info_list: List[ImageInfo], vl_model: str) -> Dict[str, str]:
        """
        职责：为所有图片生成摘要
        Args:
            document_name: 文档的名字
            img_info_list: 所有图片信息
            vl_model: vlm模型名字

        Returns:
            Dict[str,str]:{"img_name":"summary"}

        """
        summaries = {}
        request_timestamps: Deque[float] = deque()

        # 1. 获取VLM客户端
        try:
            vlm_client = AIClients.get_vlm_client()
        except Exception as e:
            for img_info in img_info_list:
                summaries[img_info.name] = "暂无摘要"
            return summaries

        # 2.调用VLM 为每一张图片生成摘要
        for img_info in img_info_list:
            # 测试一下
            self._enforce_rate_limit(request_timestamps, self.requests_per_minute)
            summaries[img_info.name] = self._summary_one(document_name, img_info, vlm_client, vl_model)

        self.logger.info(f"生成{len(summaries)}图片摘要")
        return summaries

    def _summary_one(self, document_name: str, img_info: ImageInfo, vlm_client: OpenAI, vl_model: str) -> str:
        """
        调用VLM模型为当前图片生成摘要信息
        Args:
            img_info: 当前图片信息
            vlm_client: vlm客户端
            vl_model: vlm模型名

        Returns:
         str:图片摘要

        """
        # 1. 构造VLM需要的上下文（标题名、上文内容、下文内容）
        parts = [p for p in
                 (img_info.imag_context.head, img_info.imag_context.pre_text, img_info.imag_context.post_text) if p]

        # 2. 构建最终的上下文
        final_context = "\n".join(parts) if parts else "暂无上下文"

        # 3. 根据图片地址获取到图片的内容（二进制字节流）---文本协议认识（base64编码）--->解码（‘utf-8’）--->字符串（文本协议能传输） ---- 根据收到字符串解码（二进制字节流 还原图片内容）
        try:
            with  open(img_info.path, 'rb') as f:
                img_data = base64.b64encode(f.read()).decode('utf-8')
        except IOError as e:
            self.logger.error(f"读取图片文件{img_info.path} 内容失败: {e}")
            return "暂无图片描述"

        # 4. 利用vlm客户端调用VLM模型
        try:
            resp = vlm_client.chat.completions.create(
                model=vl_model,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f"任务：为Markdown文档中的图片生成一个简短的中文标题。\n"
                                f"背景信息：\n"
                                f"  1. 所属文档标题：\"{document_name}\"\n"
                                f"  2. 图片上下文：{final_context}\n"
                                f"请结合图片内容和上述上下文信息，"
                                f"用中文简要总结这张图片的内容，"
                                f"生成一个精准的中文标题摘要（不要包含图片二字）。"
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_data}"
                            },
                        },
                    ],
                }],
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            self.logger.error(f"图片摘要生成失败 {img_info.path}: {e}")
            return "暂无图片描述"

    def _enforce_rate_limit(
            self, timestamps: Deque[float],
            max_requests: int,
            window: int = 60,
    ):
        now = time.time()
        while timestamps and now - timestamps[0] >= window:
            timestamps.popleft()

        if len(timestamps) >= max_requests:
            sleep_dur = window - (now - timestamps[0])
            if sleep_dur > 0:
                self.logger.info(
                    f"达到速率限制，暂停 {sleep_dur:.2f} 秒..."
                )
                time.sleep(sleep_dur)
            now = time.time()
            while timestamps and now - timestamps[0] >= window:
                timestamps.popleft()

        timestamps.append(now)

        # ==========================================================================
        # 【改进三】速率限制存在隐患
        # ==========================================================================
        #
        # ▎问题：
        #   1. sleep 醒来后只清理了一次过期记录，但如果 sleep 期间有其他线程/
        #      协程也发了请求，可能仍然超限（多线程场景不安全）。
        #   2. 没有区分 API 返回的 429（Too Many Requests）和真正的逻辑错误，
        #      API 端的服务端限流和客户端限流是两套机制，可能冲突。
        #   3. 如果 sleep_dur 计算出来是负数（极端时序问题），不会 sleep，
        #      但也不会报错，静默跳过，可能在高并发下打满 API 配额。
        #
        # ▎改进思路：
        #   使用 tenacity 库实现指数退避重试，配合客户端限流一起做，
        #   而不是自己手写 sleep + deque。tenacity 是 Python 生态中最流行的
        #   重试库，支持指数退避、抖动（jitter）、最大重试次数等策略。
        #
        # ▎改进代码：
        #
        #   # 文件顶部新增导入
        #   from tenacity import (
        #       retry,
        #       stop_after_attempt,
        #       wait_exponential,
        #       retry_if_exception_type,
        #   )
        #
        #   class _VLMSummarizer:
        #       def _summary_all(self, document_name, img_info_list, vl_model):
        #           summaries = {}
        #           vlm_client = AIClients.get_vlm_client()
        #
        #           for img_info in img_info_list:
        #               # 客户端侧限流（保留原来的滑动窗口）
        #               self._enforce_rate_limit(...)
        #
        #               # 服务端侧：用 tenacity 自动重试 429/5xx 错误
        #               summaries[img_info.name] = self._call_vlm_with_retry(
        #                   document_name, img_info, vlm_client, vl_model
        #               )
        #           return summaries
        #
        #       # 用装饰器声明重试策略：
        #       # - 最多重试 3 次
        #       # - 指数退避：2s → 4s → 8s（加随机抖动避免惊群效应）
        #       # - 只对 API 错误重试（网络超时、429限流、5xx服务端错误）
        #       @retry(
        #           stop=stop_after_attempt(3),
        #           wait=wait_exponential(multiplier=1, min=2, max=30),
        #           retry=retry_if_exception_type(
        #               (openai.RateLimitError, openai.APIError)
        #           ),
        #           before_sleep=lambda retry_state: self.logger.warning(
        #               f"VLM请求失败，{retry_state.next_action.sleep}秒后"
        #               f"第{retry_state.attempt_number}次重试..."
        #           ),
        #       )
        #       def _call_vlm_with_retry(self, document_name, img_info,
        #                                vlm_client, vl_model):
        #           """带自动重试的 VLM 调用"""
        #           # 构造请求参数（和原来的 _summary_one 一样）
        #           parts = [p for p in (
        #               img_info.imag_context.head,
        #               img_info.imag_context.pre_text,
        #               img_info.imag_context.post_text
        #           ) if p]
        #           final_context = "\n".join(parts) if parts else "暂无上下文"
        #
        #           with open(img_info.path, 'rb') as f:
        #               img_data = base64.b64encode(f.read()).decode('utf-8')
        #
        #           resp = vlm_client.chat.completions.create(
        #               model=vl_model,
        #               messages=[{
        #                   "role": "user",
        #                   "content": [
        #                       {"type": "text", "text": f"任务：..."},
        #                       {"type": "image_url", "image_url": {
        #                           "url": f"data:image/jpeg;base64,{img_data}"
        #                       }},
        #                   ],
        #               }],
        #           )
        #           return resp.choices[0].message.content.strip()
        #
        # ▎关键区别：
        #   原代码：API 失败 → 直接返回 "暂无图片描述"（永久丢失摘要）
        #   改进后：API 失败 → 等 2s 重试 → 还失败等 4s → 还失败等 8s →
        #           3 次都失败才兜底。大幅降低因网络抖动导致的摘要丢失
        #
        # ▎安装 tenacity：
        #   pip install tenacity
        # ==========================================================================


class _ImageUploader:
    """
    主要职责：
    1. 将本地图片上传到MinIO，得到该图片在MinIO中可访问的远程地址
    2. 替换md中的摘要和图片地址
    """

    def __init__(self, logger: Logger):
        self.logger = logger

    def upload_and_replace(self, object_dir_name: str, md_content: str, img_info_list: List[ImageInfo],
                           summaries: Dict[str, str],
                           minio_url: str, minio_bucket_name: str):
        """
        上传文件图片到minio并且更新md中的图片地址以及摘要
        Args:
            object_dir_name:  minio对象目录
            md_content:       md的内容
            img_info_list:    图片信息
            summaries:        图片摘要
            minio_url:        minio地址
            minio_bucket_name: 桶名

        Returns:
            更新后的md内容

        """

        # 1. 上传
        remote_urls = self._upload_all(object_dir_name, img_info_list, minio_url, minio_bucket_name)

        # 2. 更新
        md_content = self._update_md(md_content, summaries, remote_urls)

        return md_content

    def _upload_all(self, object_dir_name: str, img_info_list: List[ImageInfo], minio_url: str,
                    minio_bucket_name: str) -> Dict[str, str]:

        remote_urls = {}
        # 1. 得到MinIO客户端
        try:
            minio_client = StorageClients.get_minio_client()
        except Exception as e:
            for img_info in img_info_list:
                remote_urls[img_info.name] = img_info.path
            return remote_urls

        # 2. 遍历上传每一个
        for img_info in img_info_list:
            object_name = f"{object_dir_name}/{img_info.name}"
            try:
                # 2.1 上传图片到MinIO
                minio_client.fput_object(
                    minio_bucket_name, object_name, img_info.path)
                # 2.2 自己拼装路径
                # http://192.168.200.145:9000/桶名/对象名
                self.logger.info(f"成功将图片{img_info.name}上传到MinIO中")
                remote_urls[img_info.name] = f"{minio_url}/{minio_bucket_name}/{object_name}"
            except Exception as e:
                self.logger.warn(f"上传图片{img_info.name}到MinIO失败，用本地图片地址做兜底")
                remote_urls[img_info.name] = img_info.path

        self.logger.info(f"获取到远程的{len(remote_urls)}图片地址")
        return remote_urls

    def _update_md(self, md_content: str, summaries: Dict[str, str], remote_urls: Dict[str, str]) -> str:
        """
        更新MD中的图片描述和远程图片地址
        Args:
            md_content:  md内容
            summaries:   vlm生成的摘要
            remote_urls: minio生成的url

        Returns:
            新md

        """
        # 利用正则寻找(捕获组：()一个捕获组：group(0) 将整个匹配到的内容放进去 group(1)：图片的摘要 group(2):图片地址)
        pattern = re.compile(r"!\[(.*?)\]\((.*?)\)")

        def replacer(match: re.Match) -> str:
            """

            Args:
                match:

            Returns:
                ![摘要](远程图片地址)
            """

            for img_name, img_summary in summaries.items():
                origin_img_path = match.group(2)
                img_name_in_md = Path(origin_img_path).name
                if img_name == img_name_in_md:
                    return f"![{img_summary}]({remote_urls[img_name]})"
            return match.group(0)

        return pattern.sub(replacer, md_content)

        # ==========================================================================
        # 【改进一】_update_md 嵌套循环效率优化
        # ==========================================================================
        #
        # ▎问题：
        #   原代码中 replacer 函数每次被正则匹配到一张图片时，都要遍历整个
        #   summaries 字典（for img_name, img_summary in summaries.items()），
        #   时间复杂度是 O(匹配次数 × 图片数量)。
        #   如果一篇文档有 50 张图片、被引用 100 次，就要循环 5000 次。
        #
        # ▎改进思路：
        #   在正则替换之前，先预构建一个以"图片文件名"为 key 的查找字典，
        #   这样每次匹配只需要 O(1) 的字典查找就能定位到摘要和远程地址，
        #   总复杂度从 O(n×m) 降到 O(n+m)。
        #
        # ▎改进代码：
        #
        #   def _update_md(self, md_content: str, summaries: Dict[str, str],
        #                  remote_urls: Dict[str, str]) -> str:
        #       pattern = re.compile(r"!\[(.*?)\]\((.*?)\)")
        #
        #       # 预构建 O(1) 查找表：{图片文件名: (摘要, 远程URL)}
        #       # 例如: {"dial.png": ("万用表表盘刻度示意图",
        #       #                     "http://192.168.200.145:9000/.../dial.png")}
        #       lookup = {}
        #       for name, summary in summaries.items():
        #           lookup[name] = (summary, remote_urls.get(name, ""))
        #
        #       def replacer(match: re.Match) -> str:
        #           origin_img_path = match.group(2)
        #           img_name_in_md = Path(origin_img_path).name
        #
        #           # O(1) 字典查找，替代原来的 O(n) 遍历
        #           if img_name_in_md in lookup:
        #               summary, url = lookup[img_name_in_md]
        #               return f"![{summary}]({url})"
        #           return match.group(0)
        #
        #       return pattern.sub(replacer, md_content)
        #
        # ▎关键区别：
        #   原代码：每匹配一次 → 遍历 summaries（O(n)）→ 找到对应图片
        #   改进后：每匹配一次 → 字典查找 lookup[key]（O(1)）→ 直接拿到结果
        #   图片越多，改进效果越明显
        # ==========================================================================


class MarkDownToImgNode(BaseNode):
    """
    主要职责：
    1. 得到四个类的实例对象
    2. 分别调用四个实例对象的处理方法

    """

    def __init__(self):
        super().__init__()  # 显示调用父类的构造方法
        self._md_file_handler = _MdFileHandler(self.logger, self.name)
        self._img_scaner = _ImageScanner(self.logger)
        self._vlm_summarizer = _VLMSummarizer(self.logger, self.config.requests_per_minute)
        self._img_uploader = _ImageUploader(self.logger)

    name = "md_to_img_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """
        入口逻辑
        :param state:
        :return:
        """
        config = self.config
        # 1. 操作md_file_handler
        self.log_step("step1", "读取MD内容、路径以及图片的目录")
        md_content, md_path_obj, img_dir_obj = self._md_file_handler.validate_and_read_md(state)
        # 1.1 判断图片目录不存在
        if not img_dir_obj.exists():
            state['md_content'] = md_content
            return state

        # 2. 操作_img_scaner
        self.log_step("step2", "准备开始扫描图片目录")
        img_info_list: List[ImageInfo] = self._img_scaner.scan_imgs_dir(img_dir_obj,
                                                                        md_content,
                                                                        config.image_extensions,
                                                                        config.img_content_length)

        # 3. 操作_vlm_summarizer
        self.log_step("step3", "利用VLM提取摘要")
        summaries: Dict[str, str] = self._vlm_summarizer._summary_all(md_path_obj.stem, img_info_list,
                                                                      config.vl_model)

        # 4. 操作_img_uploader
        self.log_step("step4", "上传文件到MinIO,且更新MD")
        new_md_content = self._img_uploader.upload_and_replace(md_path_obj.stem, md_content, img_info_list,
                                                               summaries,
                                                               config.get_minio_base_url(),
                                                               config.minio_bucket)

        # 5. 备份调配
        self._md_file_handler.backup(md_path_obj, new_md_content)

        state['md_content'] = new_md_content

        return state
"""
● 三条改进已经全部以注释的形式写入了代码中。我来快速总结一下每条改进的位置和核 
  心要点：                                                                    
                 
  ---                                                                          
  改进一：_update_md 嵌套循环效率问题（第 595 行附近）                                                                                                        
  问题： replacer 函数每次匹配都要 for img_name in summaries.items()           
  遍历整个字典，O(n) 查找。                                                    
                                                                             
  改进： 预先构建 lookup = {图片名: (摘要, URL)} 字典，每次匹配 O(1) 查找。

  原来：每次匹配 × 遍历全部图片 = O(匹配数 × 图片数)
  改进：每次匹配 × 字典查找   = O(匹配数 × 1)

  ---
  改进二：同一张图片多次引用只处理第一次（第 226 行附近）

  问题： _find_context 在 for 循环中找到第一处匹配就
  return，后续引用被忽略。如果第一次引用处的上下文很少（比如在文档开头），VLM  
  拿到的信息就不充分。

  改进： 收集所有匹配位置的索引，为每处都提取上下文，最后选 len(head) +        
  len(pre_text) + len(post_text) 最大的那组返回给 VLM。

  ---
  改进三：速率限制隐患（第 513 行附近）

  问题：
  1. sleep 醒来后只清理了一次过期记录，多线程场景下不安全
  2. 客户端限流和 API 服务端返回的 429 是两套独立机制，可能冲突
  3. API 调用失败直接返回 "暂无图片描述"，网络抖动就永久丢失摘要

  改进： 用 tenacity 库的 @retry 装饰器实现指数退避重试（2s → 4s → 8s），区分  
  RateLimitError、APIError 等可重试异常和不可重试异常，3 次都失败才兜底。      


"""

if __name__ == '__main__':
    setup_logging()
    md_img_node = MarkDownToImgNode()
    init_state = {
        "md_path": r"D:\develop\develop\workspace\pycharm\BJ251208\shopkeeper_brain\knowledge\processor\import_processor\temp_dir\万用表的使用\hybrid_auto\万用表的使用.md"
    }
    md_img_node.process(init_state)

"""
  这个文件实现了一个 Markdown 文档中图片处理节点，属于一个基于 LangGraph         风格的图状态机流水线。核心流程是：                                           
                                                                                 读取MD文件 → 扫描图片 → VLM生成摘要 → 上传图片到MinIO → 替换MD中的图片引用                                                                                  
  整个文件采用了 "小类协作" 的设计，将职责拆分为 4 个内部类：

  ┌────────────────┬──────────────────────────────────────────────┐
  │      类名      │                     职责                     │
  ├────────────────┼──────────────────────────────────────────────┤
  │ _MdFileHandler │ 读取 MD 文件、获取图片目录、备份结果         │
  ├────────────────┼──────────────────────────────────────────────┤
  │ _ImageScanner  │ 扫描图片目录，提取每张图片在 MD 中的上下文   │
  ├────────────────┼──────────────────────────────────────────────┤
  │ _VLMSummarizer │ 调用 VLM（视觉语言模型）为图片生成摘要       │
  ├────────────────┼──────────────────────────────────────────────┤
  │ _ImageUploader │ 上传图片到 MinIO，替换 MD 中的图片地址和摘要 │
  └────────────────┴──────────────────────────────────────────────┘

  最终由 MarkDownToImgNode（继承 BaseNode）串联所有步骤。

  ---
  二、用一个具体例子走一遍全流程

  假设有如下文件结构：

  万用表的使用/
  ├── 万用表的使用.md          ← MD文件
  └── images/
      ├── dial.png             ← 表盘图片
      └── probe.png            ← 探针图片

  万用表的使用.md 内容如下：

  # 万用表的基本介绍

  万用表是电子测量中常用的工具，可以测量电压、电流和电阻。

  ## 表盘说明

  下面是万用表表盘的示意图：

  ![表盘](images/dial.png "表盘")

  表盘上有刻度线和数字标识。

  ## 探针说明

  探针分为红色和黑色两根：

  ![探针](images/probe.png "探针")

  使用时请注意正确的握持方式。

  # 使用步骤

  ...

  初始 state：
  state = {
      "md_path": "D:/.../万用表的使用/hybrid_auto/万用表的使用.md"
  }

  ---
  Step 1：_MdFileHandler.validate_and_read_md(state)

  md_content, md_path_obj, img_dir_obj =
  self._md_file_handler.validate_and_read_md(state)

  执行过程：

  1. 从 state 中取 md_path → "D:/.../万用表的使用.md"
  2. 校验非空、路径是否存在
  3. 用 open() 读取整个 MD 内容到 md_content
  4. 推导图片目录：md_path_obj.parent / "images" → D:/.../images/

  返回值：
  md_content   # 整个MD的文本内容
  md_path_obj  # Path("D:/.../万用表的使用.md")
  img_dir_obj  # Path("D:/.../images/")

  然后检查图片目录是否存在，如果不存在就直接返回（跳过后续步骤）。

  ---
  Step 2：_ImageScanner.scan_imgs_dir(...)

  img_info_list = self._img_scaner.scan_imgs_dir(
      img_dir_obj, md_content, {".png", ".jpg", ".jpeg"}, 200
  )

  执行过程（以 dial.png 为例）：

  2.1 遍历 images/ 目录

  for img_path in img_dir_obj.iterdir():

  - 过滤子目录（is_file()）
  - 过滤非法后缀（.suffix in image_extensions）

  2.2 调用 _find_context("dial.png", md_content, 200) 寻找上下文

  核心正则：
  pattern = re.compile(r"!\[.*?\]\(.*?" + re.escape("dial.png") + r".*?\)")    

  这个正则匹配 Markdown 中的图片语法：![表盘](images/dial.png "表盘")

  ▎ 知识点：re.escape() — 将特殊字符（如 .）转义，避免 dial.png 中的 .         
  ▎ 被解释为"任意字符"。

  逐行扫描：

  for md_idx, md_line in enumerate(md_lines):
      if not pattern.search(md_line):
          continue
      # 找到了！md_idx 指向 ![表盘](images/dial.png "表盘") 这一行

  向上找标题：_find_heading_up(md_lines, md_idx)

  for i in range(from_idx - 1, -1, -1):  # 从图片行往上倒着遍历
      if re.match(r"^#{1,6}\s+", md_lines[i]):  # 匹配 # 到 ###### 的标题      
          return md_lines[i], i

  ▎ 知识点：^#{1,6}\s+ — 匹配 Markdown 标题语法，{1,6} 表示 1~6 个 #。

  对 dial.png，向上找到 ## 表盘说明，返回 ("## 表盘说明", 标题行索引)。        

  向下找标题：_find_heading_down(md_lines, md_idx)

  同理向下找到 ## 探针说明，返回其索引。

  提取上文内容：

  pre_lines = md_lines[prev_index + 1 : md_idx]
  # → ["", "下面是万用表表盘的示意图：", ""]

  然后调用 _extract_limited_context(pre_lines, 200, direction="front")：       

  ▎ 知识点：段落式截取策略

  这个方法不暴力按字符数截取，而是按段落截取：

  1.
  分段规则：空行（is_blank_line）和其他图片行（is_other_image）作为段落分隔符  
  2. 就近原则：如果 direction="front"（上文），先 reverse()
  段落列表，从离图片最近的段落开始收集
  3. 长度控制：累加段落字符数，超过 img_content_length（200）就停止

  if direction == "front":
      paragraphs.reverse()  # 反转，离图片近的排前面

  for paragraph in paragraphs:
      if total + len(paragraph) > img_content_length and selected:
          break
      selected.append(paragraph)
      total += len(paragraph)

  if direction == "front":
      selected.reverse()  # 再反转回来，保证和原文顺序一致

  最终对 dial.png 返回的 ImageContext：

  ImageContext(
      head="## 表盘说明",
      pre_text="下面是万用表表盘的示意图：",
      post_text="表盘上有刻度线和数字标识。"
  )

  然后封装为 ImageInfo：

  ImageInfo(
      name="dial.png",
      path="D:/.../images/dial.png",
      imag_context=ImageContext(...)
  )

  对 probe.png 同理处理，最终返回：

  img_info_list = [ImageInfo(dial.png...), ImageInfo(probe.png...)]

  ---
  Step 3：_VLMSummarizer._summary_all(...)

  summaries = self._vlm_summarizer._summary_all("万用表的使用", img_info_list, 
  "qwen-vl-max")

  3.1 获取 VLM 客户端

  vlm_client = AIClients.get_vlm_client()  # 返回 OpenAI 兼容的客户端

  3.2 遍历每张图片生成摘要

  for img_info in img_info_list:
      self._enforce_rate_limit(request_timestamps, self.requests_per_minute)   
      summaries[img_info.name] = self._summary_one(...)

  3.3 _summary_one 单张图片摘要生成

  构造上下文：
  parts = [p for p in (head, pre_text, post_text) if p]
  final_context = "\n".join(parts)

  ▎ 知识点：列表推导式 + 真值过滤 — if p 会过滤掉空字符串 ""。

  图片 base64 编码：
  with open(img_info.path, 'rb') as f:
      img_data = base64.b64encode(f.read()).decode('utf-8')

  ▎ 知识点：二进制到文本的转换链
  ▎ - f.read() → 读取原始二进制字节
  ▎ - base64.b64encode(...) → 将二进制编码为 base64 字节
  ▎ - .decode('utf-8') → 将字节转为字符串（这样 JSON 才能传输）
  ▎
  ▎ 接收端收到字符串后反向操作：base64.decode(字符串) → 还原二进制 → 重建图片。

  调用 VLM：
  resp = vlm_client.chat.completions.create(
      model=vl_model,
      messages=[{
          "role": "user",
          "content": [
              {"type": "text", "text":
  f"任务：为Markdown文档中的图片生成一个简短的中文标题..."},
              {"type": "image_url", "image_url": {"url":
  f"data:image/jpeg;base64,{img_data}"}},
          ],
      }],
  )

  ▎ 知识点：多模态 API 调用 — content 是一个列表，包含 text 和 image_url       
  ▎ 两种类型。data:image/jpeg;base64,... 是 Data URI
  ▎ 方案，将图片数据直接内嵌到请求中。

  对 dial.png，VLM 可能返回： "万用表表盘刻度示意图"

  对 probe.png，VLM 可能返回： "万用表红黑探针实物图"

  3.4 速率限制 _enforce_rate_limit

  ▎ 知识点：滑动窗口限流

  def _enforce_rate_limit(self, timestamps: Deque[float], max_requests: int,   
  window: int = 60):

  使用 collections.deque 实现滑动窗口：

  1. 记录每次请求的时间戳到队列中
  2. 每次检查时，先清除窗口外（60秒前）的旧记录：while timestamps and now -    
  timestamps[0] >= window: timestamps.popleft()
  3. 如果队列长度 >= max_requests，计算需要等待的时间并 sleep
  4. 最后将当前时间戳加入队列

  ▎ 为什么用 deque 而不是 list？ — deque.popleft() 是 O(1)，而 list.pop(0) 是  
  ▎ O(n)。频繁从头部移除元素时 deque 性能更好。

  最终返回：
  summaries = {
      "dial.png": "万用表表盘刻度示意图",
      "probe.png": "万用表红黑探针实物图"
  }

  ---
  Step 4：_ImageUploader.upload_and_replace(...)

  new_md_content = self._img_uploader.upload_and_replace(
      "万用表的使用", md_content, img_info_list, summaries,
      "http://192.168.200.145:9000", "knowledge"
  )

  4.1 _upload_all — 上传图片到 MinIO

  minio_client.fput_object(minio_bucket_name, object_name, img_info.path)      
  remote_urls[img_info.name] = f"{minio_url}/{minio_bucket_name}/{object_name}"

  上传后拼装远程访问 URL：
  remote_urls = {
      "dial.png":
  "http://192.168.200.145:9000/knowledge/万用表的使用/dial.png",
      "probe.png":
  "http://192.168.200.145:9000/knowledge/万用表的使用/probe.png"
  }

  4.2 _update_md — 用正则替换 MD 中的图片引用

  pattern = re.compile(r"!\[(.*?)\]\((.*?)\)")

  ▎ 知识点：捕获组 (.*?)
  ▎ - group(0) — 整个匹配：![表盘](images/dial.png "表盘")
  ▎ - group(1) — 第一个捕获组：表盘（原摘要）
  ▎ - group(2) — 第二个捕获组：images/dial.png "表盘"（原路径）

  替换函数（闭包）：

  def replacer(match: re.Match) -> str:
      for img_name, img_summary in summaries.items():
          origin_img_path = match.group(2)
          img_name_in_md = Path(origin_img_path).name  # 取文件名部分
          if img_name == img_name_in_md:
              return f"![{img_summary}]({remote_urls[img_name]})"
      return match.group(0)  # 不匹配则原样返回

  return pattern.sub(replacer, md_content)

  ▎ 知识点：re.sub(pattern, replacer_func, text) — 当 replacer 
  ▎ 是一个函数时，每次匹配都会调用该函数，函数的返回值作为替换结果。这比简单的 
  ▎ 字符串替换灵活得多。

  替换结果：

  替换前：
  ![表盘](images/dial.png "表盘")

  替换后：
  ![万用表表盘刻度示意图](http://192.168.200.145:9000/knowledge/万用表的使用/di
  al.png)

  ---
  Step 5：_MdFileHandler.backup(...) — 备份处理后的 MD

  self._md_file_handler.backup(md_path_obj, new_md_content)

  生成文件名为 万用表的使用_new.md：

  new_file_path =
  md_path_obj.with_name(f"{md_path_obj.stem}_new{md_path_obj.suffix}")

  ▎ 知识点：Path 对象方法
  ▎ - .stem — 文件名不含扩展名："万用表的使用"
  ▎ - .suffix — 扩展名：".md"
  ▎ - .with_name(...) — 替换文件名，保留目录

  ---
  最终更新 state

  state['md_content'] = new_md_content  # 替换后的MD内容传给下一个节点
  return state

  ---
  三、关键知识点汇总

  ┌───────────────┬────────────────────────────────────────────────────────┐   
  │    知识点     │                      代码中的体现                      │   
  ├───────────────┼────────────────────────────────────────────────────────┤   
  │ @dataclass    │ ImageContext 和 ImageInfo 用装饰器自动生成             │   
  │               │ __init__、__repr__ 等方法                              │   
  ├───────────────┼────────────────────────────────────────────────────────┤   
  │ Path 路径操作 │ 用 pathlib.Path 代替字符串拼接路径，跨平台更安全       │   
  ├───────────────┼────────────────────────────────────────────────────────┤   
  │ 正则表达式    │ 图片匹配 !\[.*?\]\(.*?\)、标题匹配                     │   
  │               │ ^#{1,6}\s+、re.escape() 转义                           │   
  ├───────────────┼────────────────────────────────────────────────────────┤   
  │ re.sub +      │ _update_md 中用闭包函数做智能替换                      │   
  │ 函数替换      │                                                        │   
  ├───────────────┼────────────────────────────────────────────────────────┤   
  │ base64 编码   │ 将二进制图片编码为文本，通过 JSON API 传输给 VLM       │   
  ├───────────────┼────────────────────────────────────────────────────────┤   
  │ 多模态 API    │ OpenAI 兼容接口中 content 为列表，包含 text 和         │   
  │               │ image_url                                              │   
  ├───────────────┼────────────────────────────────────────────────────────┤   
  │ 滑动窗口限流  │ 用 deque 实现每分钟请求数限制                          │   
  ├───────────────┼────────────────────────────────────────────────────────┤   
  │ 模板方法模式  │ BaseNode.__call__                                      │   
  │               │ 定义流程（日志→执行→追踪），子类只需实现 process       │   
  ├───────────────┼────────────────────────────────────────────────────────┤   
  │ TypedDict     │ ImportGraphState                                       │   
  │               │ 用类型化字典定义状态结构，兼顾类型提示和灵活性         │   
  ├───────────────┼────────────────────────────────────────────────────────┤   
  │ 依赖注入      │ 各内部类通过构造函数注入 logger、config 等依赖         │   
  ├───────────────┼────────────────────────────────────────────────────────┤   
  │ 容错兜底      │ VLM 失败返回 "暂无摘要"、MinIO 失败用本地路径兜底      │   
  └───────────────┴────────────────────────────────────────────────────────┘ 
"""