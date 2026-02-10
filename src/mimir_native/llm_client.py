"""
Mimir Memory V2 - LLM Client

AWS Bedrock 封装的专用 LLM 客户端，用于 Memory Agent
所有 LLM 调用使用 Mistral/Amazon 模型（Claude 对中国 IP 限制访问）
Embedding 使用 Qwen3-Embedding-8B（更快，国内可用）
"""

import os
import json
import hashlib
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
    # Claude 模型 - 已禁用（中国 IP 限制访问）
    # claude_model_id: str = "anthropic.claude-3-5-sonnet-20240620-v1:0"
    # claude_version: str = "bedrock-2023-05-31"

    # Mistral 7B - 文本生成（中国 IP 可用）
    mistral_model_id: str = "mistral.mistral-7b-instruct-v0:2"

    # Amazon Nova - 文本生成（中国 IP 可用）
    amazon_model_id: str = "amazon.nova-micro-v1:0"

    # Amazon Nova Lite - 图像理解（中国 IP 可用）
    amazon_nova_lite_id: str = "amazon.nova-lite-v1:0"

    # Titan Embeddings - 文本嵌入（中国 IP 可用）
    titan_embedding_model_id: str = "amazon.titan-embed-text-v1"

    max_tokens: int = 1000
    temperature: float = 0.0


class BedrockClient:
    """
    AWS Bedrock 客户端 - Memory Agent 专用

    功能：
    - 事实提取（使用 Mistral，非 Claude）
    - 冲突检测（使用 Mistral，非 Claude）
    - 嵌入生成（使用 Qwen3-Embedding-8B，更快）
    - 图像理解（使用 Amazon Nova）

    注意：所有 Claude 调用已替换为 Mistral/Amazon 模型（中国 IP 兼容）
    """

    def __init__(self, config: BedrockConfig = None, aws_access_key: str = None,
                 aws_secret_key: str = None):
        """
        初始化 Bedrock 客户端

        Args:
            config: Bedrock 配置，默认使用 BedrockConfig()
            aws_access_key: AWS Access Key（可选）
            aws_secret_key: AWS Secret Key（可选）
        """
        if not BOTO3_AVAILABLE:
            raise ImportError("boto3 未安装。运行: pip install boto3")

        self.config = config or BedrockConfig()
        self._embedding_cache = {}  # 简单的嵌入缓存

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
                self.client = boto3.client(
                    'bedrock-runtime',
                    region_name=self.config.region
                )
            logger.info(f"Bedrock 客户端初始化成功 (region: {self.config.region})")
        except Exception as e:
            logger.error(f"Bedrock 客户端初始化失败: {e}")
            self.client = None
        
        # 初始化 Silicon Flow Embedding 客户端（国内更快）
        try:
            from .siliconflow_client import SiliconFlowClient
            # 使用用户提供的 API key
            self._siliconflow = SiliconFlowClient(
                api_key="sk-wzxwqcbzhzcpgszxerezpoletnmdodkvjxgegihyayohonkc",
                embedding_model="Qwen/Qwen3-Embedding-8B",
                text_model="Qwen/Qwen2.5-14B-Instruct"
            )
            logger.info("Silicon Flow 客户端初始化成功 (Embedding + Qwen2.5-14B)")
        except Exception as e:
            logger.warning(f"Silicon Flow 客户端初始化失败: {e}，将使用 Bedrock")
            self._siliconflow = None

    def is_available(self) -> bool:
        """检查客户端是否可用"""
        return self.client is not None

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
        if not self.client:
            raise RuntimeError("Bedrock 客户端未初始化")

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

    def invoke_amazon(self, prompt: str, max_tokens: int = None, temperature: float = None) -> str:
        """
        调用 Amazon Nova Micro（中国 IP 兼容）

        Args:
            prompt: 输入提示词
            max_tokens: 最大生成 token 数
            temperature: 温度参数

        Returns:
            str: 生成的文本
        """
        if not self.client:
            raise RuntimeError("Bedrock 客户端未初始化")

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
                modelId=self.config.amazon_model_id,
                body=json.dumps(body)
            )
            result = json.loads(response['body'].read())
            return result['output']['message']['content'][0]['text']

        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_message = e.response['Error']['Message']
            logger.error(f"Bedrock Amazon 调用失败: {error_code} - {error_message}")
            raise
        except Exception as e:
            logger.error(f"Bedrock Amazon 调用失败: {e}")
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
        if not self.client:
            raise RuntimeError("Bedrock 客户端未初始化")

        import base64
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

    def extract_facts(self, text: str, context: dict = None) -> List[Dict[str, Any]]:
        """
        专用方法: 事实提取（优先使用 Silicon Flow Qwen，回退到 Mistral）

        从文本中提取结构化事实，保留时间信息

        Args:
            text: 输入文本
            context: 上下文信息（包含日期等）

        Returns:
            List[Dict]: 提取的事实列表
        """
        context = context or {}

        prompt = f"""从以下文本中提取所有事实陈述。

文本: {text}

上下文: {json.dumps(context, ensure_ascii=False)}

关键要求:
1. **提取所有有意义的陈述**，即使是简单的事实（如"某人身份"、"某人的喜好"）
2. **保持原文语言**：如果输入是英文，fact 字段必须是英文，不要翻译
3. **保留所有时间信息**：原始日期格式必须保留（如 "7 May 2023"）
4. 每个事实应该是自包含的，无需上下文就能理解
5. 识别涉及的实体（人名、地点、组织等）

事实类型: event | preference | relationship | work | personal_info | other

输出 JSON 数组格式:
[
  {{
    "fact": "事实陈述（保持原文语言，不要翻译）",
    "temporal_info": {{
      "absolute_time": "YYYY-MM-DD 格式或 null",
      "relative_time": "如 'yesterday' 或 null",
      "time_mentions": ["文中提到的所有时间，保持原始格式"]
    }},
    "entities": ["实体1", "实体2"],
    "fact_type": "personal_info",
    "confidence": 0.95
  }}
]

**重要**: 
- 不要过滤！即使是简单的陈述也要提取
- 不要翻译！保持输入文本的原始语言
- 只输出 JSON，不要其他文字。"""

        # 优先使用 Silicon Flow
        if self._siliconflow:
            try:
                return self._siliconflow.extract_facts(text, context)
            except Exception as e:
                logger.warning(f"Silicon Flow fact extraction 失败，回退到 Mistral: {e}")
        
        # 回退到 Bedrock Mistral
        try:
            response = self.invoke_mistral(prompt, max_tokens=1500, temperature=0.0)
            return self._parse_json_response(response)
        except Exception as e:
            logger.error(f"事实提取失败 (Mistral): {e}")
            # 降级方案：返回空列表
            return []

    def check_conflict(self, existing_fact: str, new_fact: str,
                       existing_time: str = None, new_time: str = None) -> Dict[str, Any]:
        """
        检查两个事实是否冲突（优先使用 Silicon Flow Qwen，回退到 Mistral）

        Args:
            existing_fact: 已有事实
            new_fact: 新事实
            existing_time: 已有事实的时间
            new_time: 新事实的时间

        Returns:
            Dict: 冲突检测结果
        """
        # 优先使用 Silicon Flow
        if self._siliconflow:
            try:
                return self._siliconflow.check_conflict(new_fact, existing_fact, new_time, existing_time)
            except Exception as e:
                logger.warning(f"Silicon Flow conflict check 失败，回退到 Mistral: {e}")
        
        # 回退到 Bedrock Mistral
        prompt = f"""判断以下两个事实是否冲突。

已有事实: {existing_fact}
已有事实时间: {existing_time or '未知'}

新事实: {new_fact}
新事实时间: {new_time or '未知'}

判断标准:
1. 如果两个事实描述的是同一事物的不同状态，且不能同时为真，则为冲突
2. 如果新事实是已有事实的补充或更新（且时间更近），则不是冲突，而是更新
3. 如果两个事实描述的是不同方面，则不是冲突

输出 JSON 格式:
{{
  "is_conflict": true/false,
  "resolution": "keep_existing/update/merge/new",
  "reason": "判断理由",
  "confidence": 0.9
}}

resolution 说明:
- keep_existing: 保留已有事实
- update: 用新事实更新（时间更近或更准确）
- merge: 合并两个事实
- new: 作为新事实保留

只输出 JSON，不要其他文字。"""

        try:
            response = self.invoke_mistral(prompt, max_tokens=500, temperature=0.0)
            return self._parse_json_response(response)
        except Exception as e:
            logger.error(f"冲突检测失败 (Mistral): {e}")
            return {
                "is_conflict": False,
                "resolution": "new",
                "reason": f"检测失败，默认作为新事实: {str(e)}",
                "confidence": 0.0
            }

    def invoke_qwen(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.0) -> str:
        """
        调用 Silicon Flow Qwen 进行文本生成
        
        Args:
            prompt: 输入提示词
            max_tokens: 最大生成 token 数
            temperature: 温度参数
            
        Returns:
            str: 生成的文本
            
        Raises:
            RuntimeError: 如果 Silicon Flow 不可用
        """
        if self._siliconflow:
            return self._siliconflow.generate(prompt, max_tokens, temperature)
        raise RuntimeError("Silicon Flow not available")

    def embed(self, text: str, dimensions: int = 1536) -> Optional[List[float]]:
        """
        生成文本嵌入向量（优先使用 Qwen3-Embedding-8B，更快）

        Args:
            text: 输入文本
            dimensions: 嵌入维度 (32-4096，默认1536)

        Returns:
            List[float] 或 None（失败时）
        """
        # 优先使用 Silicon Flow（国内更快，Qwen3-Embedding-8B）
        if self._siliconflow:
            try:
                return self._siliconflow.embed(text, dimensions)
            except Exception as e:
                logger.warning(f"Silicon Flow embed 失败，回退到 Bedrock: {e}")
        
        # 回退到 Bedrock Titan
        if not self.client:
            logger.warning("Bedrock 客户端不可用，无法生成嵌入")
            return None

        # 简单缓存
        cache_key = hashlib.md5(text.encode()).hexdigest() + f"_{dimensions}"
        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]

        # Titan v1 只接受 inputText 字段
        body = {
            "inputText": text[:8000]  # Titan 有输入长度限制
        }

        try:
            response = self.client.invoke_model(
                modelId=self.config.titan_embedding_model_id,
                body=json.dumps(body)
            )
            response_body = json.loads(response['body'].read())
            embedding = response_body['embedding']

            # 缓存结果
            self._embedding_cache[cache_key] = embedding
            return embedding

        except ClientError as e:
            logger.error(f"嵌入生成失败: {e}")
            return None
        except Exception as e:
            logger.error(f"嵌入生成失败: {e}")
            return None

    def batch_embed(self, texts: List[str], dimensions: int = 1536) -> List[Optional[List[float]]]:
        """
        批量生成文本嵌入 - 优先使用 Qwen（更快），回退到 Bedrock

        Args:
            texts: 文本列表
            dimensions: 嵌入维度

        Returns:
            List[List[float]]: 嵌入向量列表（失败时为 None）
        """
        if not texts:
            return []

        # 优先使用 Silicon Flow（国内更快，Qwen3-Embedding-8B）
        if self._siliconflow:
            try:
                return self._siliconflow.batch_embed(texts, dimensions)
            except Exception as e:
                logger.warning(f"Silicon Flow batch_embed 失败，回退到 Bedrock: {e}")

        # 回退到 Bedrock Titan
        if not self.client:
            logger.warning("Bedrock 客户端不可用，无法生成嵌入")
            return [None] * len(texts)

        # 过滤空文本
        valid_texts = []
        valid_indices = []
        for i, text in enumerate(texts):
            if text and text.strip():
                valid_texts.append(text[:8000])  # Titan 有输入长度限制
                valid_indices.append(i)

        if not valid_texts:
            return [None] * len(texts)

        # 检查缓存
        results = [None] * len(texts)
        uncached_texts = []
        uncached_indices = []

        for idx, text in zip(valid_indices, valid_texts):
            cache_key = hashlib.md5(text.encode()).hexdigest() + f"_{dimensions}"
            if cache_key in self._embedding_cache:
                results[idx] = self._embedding_cache[cache_key]
            else:
                uncached_texts.append(text)
                uncached_indices.append(idx)

        if not uncached_texts:
            return results

        try:
            # Titan v1 批量 API 调用
            # 注意：Titan v1 支持 inputText 为列表
            body = {
                "inputText": uncached_texts if len(uncached_texts) > 1 else uncached_texts[0]
            }

            response = self.client.invoke_model(
                modelId=self.config.titan_embedding_model_id,
                body=json.dumps(body)
            )
            response_body = json.loads(response['body'].read())

            # 解析返回的嵌入
            # Titan 返回格式: {"embedding": [...]} 单个或批量格式
            if len(uncached_texts) == 1:
                # 单个文本
                embedding = response_body['embedding']
                cache_key = hashlib.md5(uncached_texts[0].encode()).hexdigest() + f"_{dimensions}"
                self._embedding_cache[cache_key] = embedding
                results[uncached_indices[0]] = embedding
            else:
                # 批量返回 - Titan v1 可能返回不同格式
                # 尝试不同的响应格式
                embeddings_list = None

                if 'embeddings' in response_body:
                    embeddings_list = response_body['embeddings']
                elif 'results' in response_body:
                    embeddings_list = [r['embedding'] for r in response_body['results']]
                elif isinstance(response_body.get('embedding'), list):
                    # 如果是二维数组，说明是批量返回
                    if len(response_body['embedding']) > 0 and isinstance(response_body['embedding'][0], list):
                        embeddings_list = response_body['embedding']
                    else:
                        # 单个嵌入，包装成列表
                        embeddings_list = [response_body['embedding']]

                if embeddings_list and len(embeddings_list) == len(uncached_texts):
                    for idx, text, embedding in zip(uncached_indices, uncached_texts, embeddings_list):
                        cache_key = hashlib.md5(text.encode()).hexdigest() + f"_{dimensions}"
                        self._embedding_cache[cache_key] = embedding
                        results[idx] = embedding
                else:
                    # 批量 API 返回格式不符，降级为逐个调用
                    logger.warning(f"批量嵌入返回格式不符，降级为逐个调用")
                    for idx, text in zip(uncached_indices, uncached_texts):
                        try:
                            embedding = self.embed(text, dimensions)
                            results[idx] = embedding
                        except Exception as e:
                            logger.error(f"嵌入失败 for text: {text[:50]}... - {e}")

        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_message = e.response['Error']['Message']
            logger.error(f"批量嵌入 API 调用失败: {error_code} - {error_message}")
            # 降级为逐个调用
            logger.warning("批量 API 失败，降级为逐个调用 embed()")
            for idx, text in zip(uncached_indices, uncached_texts):
                try:
                    embedding = self.embed(text, dimensions)
                    results[idx] = embedding
                except Exception as e2:
                    logger.error(f"降级嵌入失败: {e2}")
        except Exception as e:
            logger.error(f"批量嵌入失败: {e}")
            # 降级为逐个调用
            for idx, text in zip(uncached_indices, uncached_texts):
                try:
                    embedding = self.embed(text, dimensions)
                    results[idx] = embedding
                except Exception as e2:
                    logger.error(f"降级嵌入失败: {e2}")

        return results

    def _parse_json_response(self, response: str) -> Any:
        """解析 LLM 返回的 JSON"""
        try:
            # 尝试直接解析
            return json.loads(response)
        except json.JSONDecodeError:
            # 尝试从 markdown 代码块中提取
            try:
                if '```json' in response:
                    json_str = response.split('```json')[1].split('```')[0]
                    return json.loads(json_str.strip())
                elif '```' in response:
                    json_str = response.split('```')[1].split('```')[0]
                    return json.loads(json_str.strip())
            except Exception:
                pass

            # 尝试提取 [] 或 {} 包裹的内容
            try:
                start = response.find('[')
                end = response.rfind(']')
                if start != -1 and end != -1:
                    return json.loads(response[start:end+1])

                start = response.find('{')
                end = response.rfind('}')
                if start != -1 and end != -1:
                    return json.loads(response[start:end+1])
            except Exception:
                pass

            logger.error(f"无法解析 JSON 响应: {response[:200]}")
            raise ValueError(f"Invalid JSON response: {response[:200]}")


def create_llm_client(region: str = None, **kwargs) -> BedrockClient:
    """
    便捷函数：创建 LLM 客户端

    Args:
        region: AWS 区域
        **kwargs: 其他配置

    Returns:
        BedrockClient: 配置好的客户端
    """
    config = BedrockConfig(region=region or "us-east-1")

    # 从环境变量读取凭证（如果有）
    aws_access_key = kwargs.get('aws_access_key') or os.environ.get('AWS_ACCESS_KEY_ID')
    aws_secret_key = kwargs.get('aws_secret_key') or os.environ.get('AWS_SECRET_ACCESS_KEY')

    return BedrockClient(config, aws_access_key, aws_secret_key)
