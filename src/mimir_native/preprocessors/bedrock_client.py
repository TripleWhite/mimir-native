"""
AWS Bedrock Client
Bedrock API 封装 - 使用中国 IP 兼容的模型

支持模型（全部兼容中国 IP）：
- Mistral 7B - LLM 文本生成
- Amazon Nova Micro - LLM 文本生成
- Amazon Nova Lite - VLM 图像理解
- Titan Embeddings - 文本嵌入

注意：
- Bedrock 目前**没有 ASR (语音转录)** 模型
- Claude 模型已禁用（对中国 IP 限制访问）
"""

import os
import json
import base64
import logging
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# 可选依赖
try:
    import boto3
    from botocore.exceptions import ClientError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False
    logger.warning("boto3 未安装，AWS Bedrock 功能不可用")


@dataclass
class BedrockConfig:
    """Bedrock 配置 - 使用中国 IP 兼容的模型"""
    region: str = "us-east-1"
    
    # Mistral 7B - 文本生成（中国 IP 可用）
    mistral_model_id: str = "mistral.mistral-7b-instruct-v0:2"
    
    # Amazon Nova Micro - 文本生成（中国 IP 可用）
    amazon_micro_model_id: str = "amazon.nova-micro-v1:0"
    
    # Amazon Nova Lite - 图像理解（中国 IP 可用）
    amazon_nova_lite_id: str = "amazon.nova-lite-v1:0"
    
    # Titan Embeddings - 文本嵌入（中国 IP 可用）
    titan_embedding_model_id: str = "amazon.titan-embed-text-v1"
    
    max_tokens: int = 1000
    temperature: float = 0.0


class BedrockClient:
    """
    AWS Bedrock 客户端 - 使用中国 IP 兼容的模型
    
    模型选择：
    - 文本生成：Mistral 7B 或 Amazon Nova Micro
    - 图像理解：Amazon Nova Lite
    - 文本嵌入：Titan Embeddings
    
    注意：Claude 模型已禁用（对中国 IP 限制访问）
    """
    
    def __init__(self, config: BedrockConfig = None, aws_access_key: str = None, aws_secret_key: str = None):
        """
        初始化 Bedrock 客户端
        
        Args:
            config: Bedrock 配置，默认使用 BedrockConfig()
            aws_access_key: AWS Access Key（可选，默认从环境变量读取）
            aws_secret_key: AWS Secret Key（可选，默认从环境变量读取）
        """
        if not BOTO3_AVAILABLE:
            raise ImportError("boto3 未安装。运行: pip install boto3")
        
        self.config = config or BedrockConfig()
        
        # 初始化 Bedrock Runtime 客户端
        try:
            if aws_access_key and aws_secret_key:
                self.client = boto3.client(
                    'bedrock-runtime',
                    region_name=self.config.region,
                    aws_access_key_id=aws_access_key,
                    aws_secret_access_key=aws_secret_key
                )
            else:
                # 从环境变量或 ~/.aws/credentials 读取
                self.client = boto3.client(
                    'bedrock-runtime',
                    region_name=self.config.region
                )
            logger.info(f"Bedrock 客户端初始化成功 (region: {self.config.region})")
        except Exception as e:
            logger.error(f"Bedrock 客户端初始化失败: {e}")
            raise
    
    def invoke_mistral(self, prompt: str, max_tokens: int = None, temperature: float = None) -> str:
        """
        调用 Mistral 7B（中国 IP 兼容）
        
        Args:
            prompt: 输入提示词
            max_tokens: 最大生成 token 数
            temperature: 温度参数
            
        Returns:
            str: 生成的文本
        """
        body = {
            'prompt': f"<s>[INST] {prompt} [/INST]",
            'max_tokens': max_tokens or self.config.max_tokens,
            'temperature': temperature if temperature is not None else self.config.temperature
        }
        
        try:
            response = self.client.invoke_model(
                modelId=self.config.mistral_model_id,
                body=json.dumps(body)
            )
            result = json.loads(response['body'].read())
            return result['outputs'][0]['text']
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_message = e.response['Error']['Message']
            logger.error(f"Bedrock Mistral 调用失败: {error_code} - {error_message}")
            raise
        except Exception as e:
            logger.error(f"Bedrock Mistral 调用失败: {e}")
            raise
    
    def invoke_amazon_micro(self, prompt: str, max_tokens: int = None, temperature: float = None) -> str:
        """
        调用 Amazon Nova Micro（中国 IP 兼容）
        
        Args:
            prompt: 输入提示词
            max_tokens: 最大生成 token 数
            temperature: 温度参数
            
        Returns:
            str: 生成的文本
        """
        body = {
            'inferenceConfig': {
                'max_new_tokens': max_tokens or self.config.max_tokens,
                'temperature': temperature if temperature is not None else 0.7
            },
            'messages': [
                {
                    'role': 'user',
                    'content': [{'text': prompt}]
                }
            ]
        }
        
        try:
            response = self.client.invoke_model(
                modelId=self.config.amazon_micro_model_id,
                body=json.dumps(body)
            )
            result = json.loads(response['body'].read())
            return result['output']['message']['content'][0]['text']
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_message = e.response['Error']['Message']
            logger.error(f"Bedrock Amazon Micro 调用失败: {error_code} - {error_message}")
            raise
        except Exception as e:
            logger.error(f"Bedrock Amazon Micro 调用失败: {e}")
            raise
    
    def invoke_nova_vision(self, prompt: str, image_data: bytes, image_format: str = "jpeg") -> str:
        """
        调用 Amazon Nova Lite 进行图像理解（中国 IP 兼容，替代 Claude Vision）
        
        Args:
            prompt: 关于图片的问题/提示
            image_data: 图片二进制数据
            image_format: 图片格式 (jpeg, png, webp, gif)
            
        Returns:
            str: 图片描述/分析结果
        """
        base64_image = base64.b64encode(image_data).decode('utf-8')
        
        body = {
            'inferenceConfig': {
                'max_new_tokens': 1000,
                'temperature': 0.0
            },
            'messages': [
                {
                    'role': 'user',
                    'content': [
                        {
                            'image': {
                                'format': image_format,
                                'source': {
                                    'bytes': base64_image
                                }
                            }
                        },
                        {
                            'text': prompt
                        }
                    ]
                }
            ]
        }
        
        try:
            response = self.client.invoke_model(
                modelId=self.config.amazon_nova_lite_id,
                body=json.dumps(body)
            )
            result = json.loads(response['body'].read())
            return result['output']['message']['content'][0]['text']
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_message = e.response['Error']['Message']
            logger.error(f"Bedrock Nova Vision 调用失败: {error_code} - {error_message}")
            raise
        except Exception as e:
            logger.error(f"Bedrock Nova Vision 调用失败: {e}")
            raise
    
    def embed(self, text: str, dimensions: int = 1536) -> List[float]:
        """
        使用 Titan Embeddings 生成文本嵌入（中国 IP 兼容）
        
        Args:
            text: 输入文本
            dimensions: 嵌入维度 (256, 512, 1024)
            
        Returns:
            List[float]: 嵌入向量
        """
        body = {
            "inputText": text,
            "dimensions": dimensions,
            "normalize": True
        }
        
        try:
            response = self.client.invoke_model(
                modelId=self.config.titan_embedding_model_id,
                body=json.dumps(body)
            )
            response_body = json.loads(response['body'].read())
            return response_body['embedding']
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_message = e.response['Error']['Message']
            logger.error(f"Bedrock Titan Embeddings 调用失败: {error_code} - {error_message}")
            raise
        except Exception as e:
            logger.error(f"Bedrock Titan Embeddings 调用失败: {e}")
            raise
    
    def batch_embed(self, texts: List[str], dimensions: int = 1536) -> List[List[float]]:
        """
        批量生成文本嵌入
        
        Args:
            texts: 文本列表
            dimensions: 嵌入维度
            
        Returns:
            List[List[float]]: 嵌入向量列表
        """
        embeddings = []
        for text in texts:
            try:
                embedding = self.embed(text, dimensions)
                embeddings.append(embedding)
            except Exception as e:
                logger.error(f"嵌入生成失败 for text: {text[:50]}... - {e}")
                embeddings.append(None)
        return embeddings
    
    def check_available_models(self) -> Dict[str, bool]:
        """
        检查可用的 Bedrock 模型
        
        Returns:
            Dict[str, bool]: 模型可用性状态
        """
        models = {
            'mistral_7b': False,
            'amazon_nova_micro': False,
            'amazon_nova_lite': False,
            'titan_embeddings': False,
        }
        
        # 检查 Mistral 7B
        try:
            response = self.client.get_foundation_model(
                modelIdentifier=self.config.mistral_model_id
            )
            models['mistral_7b'] = response['modelDetails']['modelLifecycle']['status'] == 'ACTIVE'
        except Exception as e:
            logger.warning(f"Mistral 7B 不可用: {e}")
        
        # 检查 Amazon Nova Micro
        try:
            response = self.client.get_foundation_model(
                modelIdentifier=self.config.amazon_micro_model_id
            )
            models['amazon_nova_micro'] = response['modelDetails']['modelLifecycle']['status'] == 'ACTIVE'
        except Exception as e:
            logger.warning(f"Amazon Nova Micro 不可用: {e}")
        
        # 检查 Amazon Nova Lite
        try:
            response = self.client.get_foundation_model(
                modelIdentifier=self.config.amazon_nova_lite_id
            )
            models['amazon_nova_lite'] = response['modelDetails']['modelLifecycle']['status'] == 'ACTIVE'
        except Exception as e:
            logger.warning(f"Amazon Nova Lite 不可用: {e}")
        
        # 检查 Titan Embeddings
        try:
            response = self.client.get_foundation_model(
                modelIdentifier=self.config.titan_embedding_model_id
            )
            models['titan_embeddings'] = response['modelDetails']['modelLifecycle']['status'] == 'ACTIVE'
        except Exception as e:
            logger.warning(f"Titan Embeddings 不可用: {e}")
        
        return models
    
    # =================================================================================
    # 已弃用的 Claude 方法（保留但抛出异常，防止误用）
    # =================================================================================
    
    def invoke_claude(self, prompt: str, max_tokens: int = None) -> str:
        """
        [已弃用] Claude 模型对中国 IP 限制访问
        请使用 invoke_mistral() 或 invoke_amazon_micro()
        """
        raise NotImplementedError(
            "Claude 模型已禁用（中国 IP 限制访问）。"
            "请使用 invoke_mistral() 或 invoke_amazon_micro()"
        )
    
    def invoke_claude_vision(self, prompt: str, image_data: bytes, image_format: str = "jpeg") -> str:
        """
        [已弃用] Claude 模型对中国 IP 限制访问
        请使用 invoke_nova_vision()
        """
        raise NotImplementedError(
            "Claude Vision 已禁用（中国 IP 限制访问）。"
            "请使用 invoke_nova_vision()"
        )


def check_bedrock_configuration() -> Dict[str, Any]:
    """
    检查 Bedrock 配置状态
    
    Returns:
        Dict: 配置状态信息
        
    Example:
        >>> status = check_bedrock_configuration()
        >>> print(status)
        {
            'available': True,
            'region': 'us-east-1',
            'models': {
                'mistral_7b': True,
                'amazon_nova_micro': True,
                'amazon_nova_lite': True,
                'titan_embeddings': True
            }
        }
    """
    status = {
        'available': False,
        'region': None,
        'error': None,
        'models': {}
    }
    
    if not BOTO3_AVAILABLE:
        status['error'] = "boto3 未安装"
        return status
    
    try:
        # 尝试创建客户端
        config = BedrockConfig()
        client = BedrockClient(config)
        
        status['available'] = True
        status['region'] = config.region
        status['models'] = client.check_available_models()
        
    except Exception as e:
        status['error'] = str(e)
    
    return status


# 便捷函数
def create_bedrock_client(region: str = None) -> BedrockClient:
    """
    便捷函数：创建 Bedrock 客户端
    
    Args:
        region: AWS 区域，默认 us-east-1
        
    Returns:
        BedrockClient: 配置好的客户端
    """
    config = BedrockConfig(region=region or "us-east-1")
    return BedrockClient(config)
