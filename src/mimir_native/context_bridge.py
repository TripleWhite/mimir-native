"""
Mimir Context Bridge - 跨平台记忆注入

核心价值：用户在任何 AI 平台使用时，Mimir 自动提供相关记忆

场景：
1. 用户打开 Claude → Mimir 提供项目相关上下文
2. 用户打开 Midjourney → Mimir 提供风格偏好
3. 用户写邮件 → Mimir 提供历史沟通记录
"""

import json
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class Platform(Enum):
    """支持的平台"""
    CLAUDE = "claude"
    CHATGPT = "chatgpt"
    MIDJOURNEY = "midjourney"
    GMAIL = "gmail"
    SLACK = "slack"
    GENERIC = "generic"


@dataclass
class ContextSnippet:
    """上下文片段"""
    content: str
    source: str  # 来源（哪次对话/文档）
    relevance_score: float
    timestamp: Optional[str] = None


class MimirContextBridge:
    """
    Mimir 上下文桥接器
    
    不再关注 benchmark 分数，而是专注于：
    - 理解用户当前意图
    - 从记忆中找出最相关的上下文
    - 生成适合当前平台的 prompt 增强
    """
    
    def __init__(self, mimir_memory, llm_client):
        self.memory = mimir_memory
        self.llm = llm_client
    
    def generate_context_injection(
        self,
        user_input: str,
        platform: Platform,
        user_id: str = 'default',
        max_context_length: int = 2000
    ) -> Dict[str, Any]:
        """
        生成上下文注入
        
        Args:
            user_input: 用户当前的输入/prompt
            platform: 使用的平台（决定 context 格式）
            user_id: 用户ID
            max_context_length: 最大上下文长度
            
        Returns:
            {
                'enhanced_prompt': 增强后的 prompt,
                'context_snippets': 相关记忆片段,
                'suggestions': 建议（可选）
            }
        """
        # 1. 理解用户意图
        intent = self._analyze_intent(user_input)
        logger.info(f"用户意图: {intent}")
        
        # 2. 检索相关记忆
        memories = self._retrieve_relevant_memories(
            query=user_input,
            intent=intent,
            user_id=user_id,
            top_k=10
        )
        
        # 3. 根据平台生成 context
        if platform == Platform.CLAUDE:
            return self._format_for_claude(user_input, memories, max_context_length)
        elif platform == Platform.MIDJOURNEY:
            return self._format_for_midjourney(user_input, memories, max_context_length)
        elif platform == Platform.GMAIL:
            return self._format_for_gmail(user_input, memories, max_context_length)
        else:
            return self._format_generic(user_input, memories, max_context_length)
    
    def _analyze_intent(self, user_input: str) -> Dict[str, Any]:
        """
        分析用户意图
        
        不只是关键词匹配，而是理解用户想做什么
        """
        prompt = f"""分析以下用户输入的意图：

用户输入：{user_input}

请提取：
1. 主要意图（coding/writing/design/research/communication/other）
2. 相关实体（项目名称、人名、主题等）
3. 需要的上下文类型（历史代码/过往对话/风格偏好/文档资料）

输出 JSON 格式：
{{
  "intent": "coding",
  "entities": ["项目A", "用户系统"],
  "context_type": "historical_code"
}}"""
        
        try:
            response = self.llm.invoke_mistral(prompt, max_tokens=300, temperature=0.0)
            return json.loads(response)
        except:
            return {"intent": "unknown", "entities": [], "context_type": "general"}
    
    def _retrieve_relevant_memories(
        self, 
        query: str, 
        intent: Dict,
        user_id: str,
        top_k: int = 10
    ) -> List[ContextSnippet]:
        """
        检索相关记忆
        
        基于意图和查询，找出最相关的记忆
        """
        # 构建增强查询
        enhanced_query = query
        if intent.get('entities'):
            enhanced_query += " " + " ".join(intent['entities'])
        
        # 多维度检索
        memories = self.memory.query(
            enhanced_query, 
            user_id=user_id, 
            top_k=top_k
        )
        
        snippets = []
        for m in memories:
            content = m.memory.content if hasattr(m, 'memory') else str(m)
            score = m.score if hasattr(m, 'score') else 0.5
            
            snippets.append(ContextSnippet(
                content=content,
                source="memory",
                relevance_score=score,
                timestamp=None
            ))
        
        # 按相关性排序
        snippets.sort(key=lambda x: x.relevance_score, reverse=True)
        return snippets
    
    def _format_for_claude(
        self, 
        user_input: str, 
        memories: List[ContextSnippet],
        max_length: int
    ) -> Dict[str, Any]:
        """
        为 Claude 格式化上下文
        
        Claude 特点：
        - 支持长上下文
        - 适合代码和技术内容
        - 可以用 XML tag 组织信息
        """
        # 选择最相关的记忆
        selected = []
        current_length = 0
        
        for snippet in memories:
            if current_length + len(snippet.content) < max_length:
                selected.append(snippet)
                current_length += len(snippet.content)
            else:
                break
        
        # 构建增强 prompt
        context_parts = []
        for i, s in enumerate(selected, 1):
            context_parts.append(f"[相关记忆 {i}]\n{s.content}\n")
        
        context_text = "\n".join(context_parts)
        
        enhanced_prompt = f"""以下是与当前任务相关的历史记忆：

<context>
{context_text}
</context>

用户当前输入：
{user_input}

请基于以上上下文回答。"""
        
        return {
            'enhanced_prompt': enhanced_prompt,
            'context_snippets': selected,
            'suggestions': []
        }
    
    def _format_for_midjourney(
        self, 
        user_input: str, 
        memories: List[ContextSnippet],
        max_length: int
    ) -> Dict[str, Any]:
        """
        为 Midjourney 格式化上下文
        
        Midjourney 特点：
        - 需要风格描述
        - 关键词重要
        - 不适合长文本
        """
        # 提取风格相关信息
        style_keywords = []
        for snippet in memories:
            # 简单提取风格关键词
            if any(kw in snippet.content.lower() for kw in ['style', 'color', 'lighting', 'mood']):
                # 提取关键描述
                words = snippet.content.split()
                style_keywords.extend(words[:10])  # 取前10个词
        
        style_text = ", ".join(list(set(style_keywords))[:20])  # 去重，限制数量
        
        enhanced_prompt = user_input
        if style_text:
            enhanced_prompt += f" --style {style_text}"
        
        return {
            'enhanced_prompt': enhanced_prompt,
            'context_snippets': memories[:3],
            'suggestions': [f"基于您之前的偏好，建议添加风格: {style_text}"] if style_text else []
        }
    
    def _format_for_gmail(
        self, 
        user_input: str, 
        memories: List[ContextSnippet],
        max_length: int
    ) -> Dict[str, Any]:
        """
        为 Gmail 格式化上下文
        
        邮件特点：
        - 需要历史沟通记录
        - 语气建议
        - 关键信息提醒
        """
        # 提取历史沟通要点
        key_points = []
        for snippet in memories[:5]:
            key_points.append(f"- {snippet.content[:100]}...")
        
        context_text = "\n".join(key_points)
        
        enhanced_prompt = f"""撰写邮件时请参考以下历史沟通要点：

历史记录：
{context_text}

当前邮件内容：
{user_input}

建议：
- 提及之前的讨论以建立连续性
- 保持与之前沟通一致的语气"""
        
        return {
            'enhanced_prompt': enhanced_prompt,
            'context_snippets': memories[:5],
            'suggestions': ["已为您加载相关历史沟通记录"]
        }
    
    def _format_generic(
        self, 
        user_input: str, 
        memories: List[ContextSnippet],
        max_length: int
    ) -> Dict[str, Any]:
        """通用格式"""
        selected = memories[:5]
        
        context_text = "\n\n".join([f"- {s.content[:200]}" for s in selected])
        
        enhanced_prompt = f"""相关背景信息：

{context_text}

---

{user_input}"""
        
        return {
            'enhanced_prompt': enhanced_prompt,
            'context_snippets': selected,
            'suggestions': []
        }


class MimirAutoComplete:
    """
    Mimir 自动补全
    
    在用户输入时，实时提供记忆相关的建议和补全
    """
    
    def __init__(self, mimir_memory):
        self.memory = mimir_memory
    
    def get_suggestions(
        self, 
        partial_input: str,
        user_id: str = 'default'
    ) -> List[Dict[str, str]]:
        """
        基于部分输入，提供记忆相关的建议
        
        例如：
        - 用户输入 "上次那个项目..." → 提示项目名称
        - 用户输入 "我们之前讨论过..." → 提示讨论主题
        """
        if len(partial_input) < 3:
            return []
        
        # 检索相关记忆
        memories = self.memory.query(partial_input, user_id=user_id, top_k=5)
        
        suggestions = []
        for m in memories:
            content = m.memory.content if hasattr(m, 'memory') else str(m)
            # 生成建议
            suggestions.append({
                'type': 'memory',
                'text': f"💭 {content[:50]}...",
                'full_content': content
            })
        
        return suggestions


# 使用示例
if __name__ == "__main__":
    from mimir_native import MimirMemory
    from mimir_native.llm_client import BedrockClient
    
    # 初始化
    mimir = MimirMemory(db_path='mimir.db')
    llm = BedrockClient()
    bridge = MimirContextBridge(mimir, llm)
    
    # 示例 1: Claude 编码场景
    result = bridge.generate_context_injection(
        user_input="帮我写用户登录功能",
        platform=Platform.CLAUDE,
        user_id='user_123'
    )
    print("Claude 增强 Prompt:")
    print(result['enhanced_prompt'])
    
    # 示例 2: Midjourney 设计场景
    result = bridge.generate_context_injection(
        user_input="a futuristic city",
        platform=Platform.MIDJOURNEY,
        user_id='user_123'
    )
    print("\nMidjourney 增强 Prompt:")
    print(result['enhanced_prompt'])
