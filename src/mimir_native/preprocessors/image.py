"""
Image Preprocessor - Bedrock First Implementation
图像预处理器 - Bedrock 优先实现

VLM (视觉描述): 使用 AWS Bedrock Amazon Nova Lite（中国 IP 兼容）
OCR: 使用 pytesseract (轻量本地) 或 Google Vision API
"""

import os
import logging
import base64
from pathlib import Path
from typing import Any, Optional, List, Dict
from datetime import datetime
import io

from .base import BasePreprocessor, RawContent, parse_date

logger = logging.getLogger(__name__)

# OCR 可选依赖（轻量本地）
try:
    import pytesseract
    from PIL import Image
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    logger.warning("pytesseract 未安装，OCR 功能不可用。建议: pip install pytesseract pillow")

# Bedrock 客户端
from .bedrock_client import BedrockClient, BedrockConfig, check_bedrock_configuration

# 备选 API 客户端（当 Bedrock 不可用时）
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


def detect_image_format(image_data: bytes) -> str:
    """检测图片格式"""
    # JPEG
    if image_data[:2] == b'\xff\xd8':
        return 'jpeg'
    # PNG
    elif image_data[:8] == b'\x89PNG\r\n\x1a\n':
        return 'png'
    # GIF
    elif image_data[:6] in (b'GIF87a', b'GIF89a'):
        return 'gif'
    # WebP
    elif image_data[:4] == b'RIFF' and image_data[8:12] == b'WEBP':
        return 'webp'
    # BMP
    elif image_data[:2] == b'BM':
        return 'bmp'
    # TIFF
    elif image_data[:4] in (b'II\x2a\x00', b'MM\x00\x2a'):
        return 'tiff'
    else:
        return 'jpeg'  # 默认


class ImageProcessor(BasePreprocessor):
    """
    图像预处理器 - Bedrock 优先实现（中国 IP 兼容）

    功能：
    - OCR 提取图像中的文本（pytesseract 轻量本地）
    - 视觉描述生成（AWS Bedrock Amazon Nova Lite，中国 IP 可用）

    环境变量：
    - AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY: AWS 凭证
    - AWS_REGION: AWS 区域（默认 us-east-1）
    - IMAGE_PROVIDER: 可选 'bedrock'(默认), 'openai'

    支持的格式：PNG, JPG, JPEG, GIF, BMP, TIFF, WebP
    """

    SUPPORTED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp'}

    def __init__(
        self,
        bedrock_client: BedrockClient = None,
        openai_client=None,
        provider: str = None,
        region: str = None
    ):
        """
        初始化图像处理器

        Args:
            bedrock_client: 预配置的 Bedrock 客户端（可选）
            openai_client: 预配置的 OpenAI 客户端（可选，备选）
            provider: 提供商 'bedrock'(默认) 或 'openai'
            region: AWS 区域（默认从环境变量读取或 us-east-1）
        """
        self.provider = provider or os.getenv('IMAGE_PROVIDER', 'bedrock').lower()
        self.region = region or os.getenv('AWS_REGION', 'us-east-1')

        self._bedrock_client = bedrock_client
        self._openai_client = openai_client

        # 延迟初始化客户端
        self._initialized = False

    def _ensure_initialized(self):
        """确保客户端已初始化"""
        if self._initialized:
            return

        if self.provider == 'bedrock':
            if self._bedrock_client is None:
                try:
                    config = BedrockConfig(region=self.region)
                    self._bedrock_client = BedrockClient(config)
                    logger.info(f"Bedrock 客户端初始化成功")
                except Exception as e:
                    logger.error(f"Bedrock 初始化失败: {e}")
                    # 尝试回退到 OpenAI
                    if OPENAI_AVAILABLE and os.getenv('OPENAI_API_KEY'):
                        logger.warning("回退到 OpenAI")
                        self.provider = 'openai'
                        self._openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
                    else:
                        raise

        elif self.provider == 'openai':
            if self._openai_client is None:
                if not OPENAI_AVAILABLE:
                    raise ImportError("OpenAI SDK 未安装。运行: pip install openai")
                api_key = os.getenv('OPENAI_API_KEY')
                if not api_key:
                    raise ValueError("OpenAI API Key 未配置")
                self._openai_client = OpenAI(api_key=api_key)

        else:
            raise ValueError(f"不支持的图像处理器: {self.provider}")

        self._initialized = True

    def supports(self, content_type: str) -> bool:
        """检查是否支持指定的内容类型"""
        return content_type.lower() in {'image', 'img', 'picture', 'photo'}

    def process(self, content: Any, metadata: dict = None) -> RawContent:
        """
        处理图像内容

        Args:
            content: 图像路径 (str/Path) 或图像内容 (bytes)
            metadata: 元数据字典

        Returns:
            RawContent: 标准化的内容对象
                [Image] {description}
                [OCR] {text}
        """
        metadata = metadata or {}

        try:
            # 确保客户端已初始化
            self._ensure_initialized()

            # 获取图像数据
            image_data, file_name = self._get_image_data(content, metadata)

            # OCR 提取文本（轻量本地）
            ocr_text = self._extract_ocr(image_data)

            # VLM 生成视觉描述（Bedrock Amazon Nova Lite，中国 IP 兼容）
            description = self._generate_vlm_description(image_data, metadata)

            # 组合输出
            text_parts = []
            if description:
                text_parts.append(f"[Image] {description}")
            if ocr_text:
                text_parts.append(f"[OCR] {ocr_text}")

            full_text = "\n".join(text_parts) if text_parts else "[Image] (无内容提取)"

            # 分块和摘要
            chunks = self._chunk_text(full_text) if len(full_text) > 1000 else [full_text]
            summary = description or (ocr_text[:200] if ocr_text else "图像内容")

            # 解析时间
            occurred_at = parse_date(metadata.get('created_date') or metadata.get('timestamp'))

            return RawContent(
                text=full_text,
                summary=summary[:200] if summary else None,
                chunks=chunks,
                metadata={
                    'file_name': file_name,
                    'has_ocr': bool(ocr_text),
                    'has_description': bool(description),
                    'ocr_text_length': len(ocr_text) if ocr_text else 0,
                    'vlm_provider': self.provider,
                    **metadata
                },
                occurred_at=occurred_at
            )

        except Exception as e:
            logger.error(f"图像处理失败: {e}")
            return RawContent(
                text=f"[图像处理错误] {str(e)}",
                summary="图像处理失败",
                chunks=[],
                metadata={'error': str(e), **metadata},
                occurred_at=None
            )

    def _get_image_data(self, content: Any, metadata: dict) -> tuple:
        """获取图像数据和文件名"""
        if isinstance(content, (str, Path)):
            image_path = str(content)
            with open(image_path, 'rb') as f:
                return f.read(), Path(image_path).name
        elif isinstance(content, bytes):
            return content, metadata.get('file_name', 'image.jpg')
        else:
            raise ValueError(f"不支持的图像内容类型: {type(content)}")

    def _extract_ocr(self, image_data: bytes) -> str:
        """使用 OCR 提取图像文本"""
        if not TESSERACT_AVAILABLE:
            logger.debug("OCR 不可用：pytesseract 未安装")
            return ""

        try:
            image = Image.open(io.BytesIO(image_data))
            if image.mode != 'RGB':
                image = image.convert('RGB')

            text = pytesseract.image_to_string(image, lang='chi_sim+eng')
            return '\n'.join(line for line in text.strip().split('\n') if line.strip())

        except Exception as e:
            logger.warning(f"OCR 提取失败: {e}")
            return ""

    def _generate_vlm_description(self, image_data: bytes, metadata: dict) -> str:
        """
        调用 VLM 生成视觉描述

        优先使用 Amazon Nova Lite（中国 IP 兼容），失败可回退到 OpenAI
        """
        prompt = metadata.get('prompt', "请详细描述这张图片的内容。如果有文字，请识别并描述文字内容。")
        image_format = detect_image_format(image_data)

        try:
            if self.provider == 'bedrock':
                # 使用 Amazon Nova Lite 替代 Claude（中国 IP 兼容）
                return self._bedrock_client.invoke_nova_vision(
                    prompt=prompt,
                    image_data=image_data,
                    image_format=image_format
                )
            elif self.provider == 'openai':
                return self._call_openai_vision(image_data, prompt)
            else:
                raise ValueError(f"不支持的 VLM 提供商: {self.provider}")

        except Exception as e:
            logger.error(f"VLM 调用失败 ({self.provider}): {e}")

            # 如果 Bedrock 失败且 OpenAI 可用，尝试回退
            if self.provider == 'bedrock' and self._openai_client:
                try:
                    logger.warning("Bedrock 失败，尝试 OpenAI 回退")
                    return self._call_openai_vision(image_data, prompt)
                except Exception as e2:
                    logger.error(f"OpenAI 回退也失败: {e2}")

            return ""

    def _call_openai_vision(self, image_data: bytes, prompt: str) -> str:
        """调用 OpenAI GPT-4V（备选）"""
        base64_image = base64.b64encode(image_data).decode('utf-8')

        response = self._openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                        }
                    ]
                }
            ],
            max_tokens=500
        )
        return response.choices[0].message.content


# 便捷函数
def process_image(
    image_path: str,
    provider: str = None,
    prompt: str = None
) -> RawContent:
    """
    便捷函数：处理图像

    默认使用 Bedrock Amazon Nova Lite（中国 IP 兼容），自动从 AWS 环境变量读取配置

    Args:
        image_path: 图像文件路径
        provider: 提供商 'bedrock'(默认) 或 'openai'
        prompt: 自定义提示词（可选）

    Returns:
        RawContent: 处理结果

    Example:
        >>> # 配置 AWS 凭证（~/.aws/credentials 或环境变量）
        >>> result = process_image("/path/to/image.jpg")
        >>> print(result.text)
        [Image] 图片中包含...
        [OCR] 提取的文字...
    """
    metadata = {'file_name': Path(image_path).name}
    if prompt:
        metadata['prompt'] = prompt

    processor = ImageProcessor(provider=provider)
    return processor.process(image_path, metadata)
