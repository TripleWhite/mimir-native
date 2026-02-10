"""
Mimir Memory V2 - Multimodal Preprocessors
多模态预处理层统一入口 - Bedrock 优先实现

提供统一的接口来处理多种内容类型：
- document: PDF, DOCX, TXT 文档
- image: 图片（OCR + Bedrock Claude 3.5 Sonnet 视觉描述）
- audio: 音频（⚠️ Bedrock 无 ASR，当前使用 OpenAI Whisper API 临时方案）
- conversation: 对话（时序修复）

Bedrock 配置（优先使用）：
- AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_REGION
- Claude 3.5 Sonnet: VLM / LLM
- Titan Embeddings: 文本嵌入

⚠️ ASR 说明：
AWS Bedrock 目前没有 ASR (语音转录) 模型，
AudioProcessor 当前使用 OpenAI Whisper API 作为临时方案。
需要用户决策最终 ASR 方案。

用法示例：
    from app.mimir_v2.preprocessors import (
        MultimodalPreprocessor,
        BedrockClient,
        check_bedrock_configuration
    )
    
    # 检查 Bedrock 配置
    status = check_bedrock_configuration()
    
    # 处理图像（使用 Bedrock Claude 3.5 Sonnet）
    preprocessor = MultimodalPreprocessor()
    result = preprocessor.process(
        content="/path/to/image.jpg",
        content_type="image"
    )
"""

import os
from typing import Optional
from .base import BasePreprocessor, RawContent, parse_date
from .document import DocumentProcessor
from .image import ImageProcessor
from .audio import AudioProcessor
from .conversation import ConversationProcessor
from .bedrock_client import BedrockClient, BedrockConfig, check_bedrock_configuration


class MultimodalPreprocessor:
    """
    多模态预处理器统一入口 - Bedrock 优先实现
    
    模型调用优先级：
    1. AWS Bedrock (优先)
       - Image VLM: Claude 3.5 Sonnet
       - Embedding: Titan Embeddings
    2. 其他供应商 (Bedrock 不支持时)
       - Audio ASR: OpenAI Whisper API (临时方案)
    
    ⚠️ ASR 说明：Bedrock 没有 ASR 模型，AudioProcessor 使用 OpenAI Whisper API
    """
    
    def __init__(
        self,
        bedrock_client: BedrockClient = None,
        openai_client=None,
        asr_client=None,
        region: str = None,
        enable_diarization: bool = False
    ):
        """
        初始化多模态预处理器
        
        Args:
            bedrock_client: 预配置的 Bedrock 客户端（可选）
            openai_client: 预配置的 OpenAI 客户端（可选，音频 ASR 用）
            asr_client: 预配置的 ASR 客户端（可选）
            region: AWS 区域（默认从环境变量读取）
            enable_diarization: 是否启用说话人分离
        """
        # 创建或复用 Bedrock 客户端
        if bedrock_client is None:
            try:
                config = BedrockConfig(region=region or os.getenv('AWS_REGION', 'us-east-1'))
                bedrock_client = BedrockClient(config)
            except Exception as e:
                print(f"警告: Bedrock 客户端初始化失败: {e}")
                bedrock_client = None
        
        self.processors = {
            'document': DocumentProcessor(),
            'image': ImageProcessor(bedrock_client=bedrock_client),
            'audio': AudioProcessor(
                asr_client=asr_client or openai_client,
                enable_diarization=enable_diarization
            ),
            'conversation': ConversationProcessor(),
        }
        
        # 别名映射
        self._type_aliases = {
            'pdf': 'document',
            'docx': 'document',
            'txt': 'document',
            'text': 'document',
            'img': 'image',
            'picture': 'image',
            'photo': 'image',
            'voice': 'audio',
            'speech': 'audio',
            'recording': 'audio',
            'chat': 'conversation',
            'dialogue': 'conversation',
            'message': 'conversation',
        }
        
        self._bedrock_client = bedrock_client
    
    def process(self, content, content_type: str, metadata: dict = None) -> RawContent:
        """
        处理内容
        
        Args:
            content: 输入内容
            content_type: 内容类型
            metadata: 可选的元数据字典
            
        Returns:
            RawContent: 标准化的内容对象
        """
        if metadata is None:
            metadata = {}
        
        normalized_type = self._normalize_type(content_type)
        processor = self.processors.get(normalized_type)
        
        if not processor:
            raise ValueError(
                f"Unknown content type: {content_type} (normalized: {normalized_type}). "
                f"Supported types: {list(self.processors.keys())}"
            )
        
        return processor.process(content, metadata)
    
    def supports(self, content_type: str) -> bool:
        """检查是否支持指定的内容类型"""
        normalized_type = self._normalize_type(content_type)
        return normalized_type in self.processors
    
    def _normalize_type(self, content_type: str) -> str:
        """标准化内容类型"""
        content_type_lower = content_type.lower()
        if content_type_lower in self._type_aliases:
            return self._type_aliases[content_type_lower]
        return content_type_lower
    
    def get_processor(self, content_type: str) -> BasePreprocessor:
        """获取指定类型的处理器"""
        normalized_type = self._normalize_type(content_type)
        processor = self.processors.get(normalized_type)
        if not processor:
            raise ValueError(f"Unknown content type: {content_type}")
        return processor
    
    def list_supported_types(self) -> list:
        """获取支持的内容类型列表"""
        types = []
        for main_type in self.processors.keys():
            types.append(main_type)
            for alias, target in self._type_aliases.items():
                if target == main_type:
                    types.append(alias)
        return sorted(set(types))
    
    @property
    def bedrock_client(self) -> Optional[BedrockClient]:
        """获取 Bedrock 客户端（用于直接调用）"""
        return self._bedrock_client


# 便捷函数
def preprocess_document(file_path: str, **kwargs) -> RawContent:
    """便捷函数：处理文档"""
    preprocessor = MultimodalPreprocessor()
    return preprocessor.process(
        content=file_path,
        content_type='document',
        metadata={'file_path': file_path, **kwargs}
    )


def preprocess_image(image_path: str, **kwargs) -> RawContent:
    """
    便捷函数：处理图像（使用 Bedrock Claude 3.5 Sonnet）
    
    自动使用 AWS Bedrock 进行视觉理解
    """
    preprocessor = MultimodalPreprocessor()
    return preprocessor.process(
        content=image_path,
        content_type='image',
        metadata={'image_path': image_path, **kwargs}
    )


def preprocess_audio(audio_path: str, **kwargs) -> RawContent:
    """
    便捷函数：处理音频
    
    ⚠️ 当前使用 OpenAI Whisper API（临时方案）
    AWS Bedrock 目前没有 ASR 模型
    """
    preprocessor = MultimodalPreprocessor()
    return preprocessor.process(
        content=audio_path,
        content_type='audio',
        metadata={'audio_path': audio_path, **kwargs}
    )


def preprocess_conversation(conversation: dict, **kwargs) -> RawContent:
    """便捷函数：处理对话"""
    preprocessor = MultimodalPreprocessor()
    return preprocessor.process(
        content=conversation,
        content_type='conversation',
        metadata=kwargs
    )


def check_configuration() -> dict:
    """
    检查所有配置状态
    
    Returns:
        dict: 配置状态
        
    Example:
        >>> status = check_configuration()
        >>> print(status)
        {
            'bedrock': {
                'available': True,
                'region': 'us-east-1',
                'models': {
                    'claude_3_5_sonnet': True,
                    'titan_embeddings': True
                }
            },
            'openai_asr': {
                'available': True,  # 临时方案
                'note': 'Bedrock 无 ASR 模型，使用 OpenAI Whisper API'
            }
        }
    """
    status = {
        'bedrock': check_bedrock_configuration(),
        'openai_asr': {
            'available': bool(os.getenv('OPENAI_API_KEY')),
            'note': 'Bedrock 无 ASR 模型，当前使用 OpenAI Whisper API 临时方案'
        }
    }
    return status


__all__ = [
    'MultimodalPreprocessor',
    'BasePreprocessor',
    'RawContent',
    'DocumentProcessor',
    'ImageProcessor',
    'AudioProcessor',
    'ConversationProcessor',
    'BedrockClient',
    'BedrockConfig',
    'parse_date',
    'check_bedrock_configuration',
    'check_configuration',
    'preprocess_document',
    'preprocess_image',
    'preprocess_audio',
    'preprocess_conversation',
]
