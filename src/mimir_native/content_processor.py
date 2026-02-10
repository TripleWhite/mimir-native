"""
Processing 层时序绑定修复

核心问题：确保每条消息在提取时就绑定正确的 session_date
"""

import json
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass


@dataclass
class RawContent:
    """原始内容单元"""
    text: str
    source_type: str  # 'claude', 'wechat', 'pdf', etc.
    source_id: str
    created_at: datetime  # 内容的创建时间
    metadata: Dict[str, Any]
    
    # 对于对话类内容
    speaker: Optional[str] = None
    session_date: Optional[str] = None  # 对话发生的日期


class TemporalNormalizer:
    """
    时序标准化器 - 在 Processing 层强制替换相对时间
    """
    
    WEEKDAY_MAP = {
        'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3,
        'friday': 4, 'saturday': 5, 'sunday': 6
    }
    
    MONTH_MAP = {
        'january': 1, 'february': 2, 'march': 3, 'april': 4,
        'may': 5, 'june': 6, 'july': 7, 'august': 8,
        'september': 9, 'october': 10, 'november': 11, 'december': 12
    }
    
    def __init__(self):
        self.time_patterns = [
            # (pattern, offset_func, format_func)
            (r'\byesterday\b', lambda d: d - timedelta(days=1), 
             lambda d: d.strftime('%d %B %Y')),
            (r'\btoday\b', lambda d: d, 
             lambda d: d.strftime('%d %B %Y')),
            (r'\btomorrow\b', lambda d: d + timedelta(days=1), 
             lambda d: d.strftime('%d %B %Y')),
            (r'\blast week\b', lambda d: d - timedelta(days=7), 
             lambda d: d.strftime('%d %B %Y')),
            (r'\bnext week\b', lambda d: d + timedelta(days=7), 
             lambda d: d.strftime('%d %B %Y')),
            (r'\blast year\b', lambda d: d.replace(year=d.year-1), 
             lambda d: str(d.year-1)),
            (r'\bnext year\b', lambda d: d.replace(year=d.year+1), 
             lambda d: str(d.year+1)),
        ]
    
    def parse_date(self, date_str: str) -> Optional[datetime]:
        """解析各种日期格式"""
        if not date_str:
            return None
        
        # 首先清理 LoCoMo 格式: "1:56 pm on 8 May, 2023" -> "8 May 2023"
        # 匹配 "time on date" 或 "time at date" 格式
        locomo_match = re.search(r'\d{1,2}:\d{2}\s*(?:am|pm)\s+(?:on|at)\s+([\d]{1,2}\s+[A-Za-z]+,?\s*\d{4})', date_str, re.IGNORECASE)
        if locomo_match:
            date_str = locomo_match.group(1)
        
        # 移除逗号 "8 May, 2023" -> "8 May 2023"
        date_str = date_str.replace(',', '')
        
        formats = [
            '%d %B %Y',      # 8 May 2023
            '%d %b %Y',      # 8 May 2023
            '%Y-%m-%d',      # 2023-05-08
            '%B %d %Y',      # May 8 2023
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_str.strip(), fmt)
            except:
                continue
        
        # 手动解析 "8 May 2023"
        match = re.match(r'(\d{1,2})\s+([A-Za-z]+)\s*(\d{4})', date_str)
        if match:
            day, month_str, year = match.groups()
            month = self.MONTH_MAP.get(month_str.lower())
            if month:
                return datetime(int(year), month, int(day))
        
        return None
    
    def normalize(self, text: str, reference_date: str) -> str:
        """
        标准化文本中的相对时间
        
        Args:
            text: 原始文本
            reference_date: 参考日期（如 "8 May 2023"）
        
        Returns:
            替换后的文本
        """
        ref_date = self.parse_date(reference_date)
        if not ref_date:
            return text
        
        result = text
        
        # yesterday → 昨天日期
        if re.search(r'\byesterday\b', result, re.IGNORECASE):
            yesterday = ref_date - timedelta(days=1)
            result = re.sub(r'\byesterday\b', yesterday.strftime('%d %B %Y'), result, flags=re.IGNORECASE)
        
        # today → 当天日期
        if re.search(r'\btoday\b', result, re.IGNORECASE):
            today_str = ref_date.strftime('%d %B %Y')
            result = re.sub(r'\btoday\b', today_str, result, flags=re.IGNORECASE)
        
        # tomorrow → 明天日期
        if re.search(r'\btomorrow\b', result, re.IGNORECASE):
            tomorrow = ref_date + timedelta(days=1)
            result = re.sub(r'\btomorrow\b', tomorrow.strftime('%d %B %Y'), result, flags=re.IGNORECASE)
        
        # last week → 上周同一天
        if re.search(r'\blast week\b', result, re.IGNORECASE):
            last_week = ref_date - timedelta(days=7)
            result = re.sub(r'\blast week\b', last_week.strftime('%d %B %Y'), result, flags=re.IGNORECASE)
        
        # next week → 下周同一天
        if re.search(r'\bnext week\b', result, re.IGNORECASE):
            next_week = ref_date + timedelta(days=7)
            result = re.sub(r'\bnext week\b', next_week.strftime('%d %B %Y'), result, flags=re.IGNORECASE)
        
        # last year → 去年
        if re.search(r'\blast year\b', result, re.IGNORECASE):
            last_year = str(ref_date.year - 1)
            result = re.sub(r'\blast year\b', last_year, result, flags=re.IGNORECASE)
        
        # next year → 明年
        if re.search(r'\bnext year\b', result, re.IGNORECASE):
            next_year = str(ref_date.year + 1)
            result = re.sub(r'\bnext year\b', next_year, result, flags=re.IGNORECASE)
        
        # 处理星期几
        result = self._replace_weekday(result, ref_date)
        
        return result
    
    def _replace_weekday(self, text: str, ref_date: datetime) -> str:
        """替换 last/next [weekday]"""
        
        # last Monday, next Tuesday, etc.
        pattern = r'\b(last|next)\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b'
        
        def replace(match):
            direction = match.group(1).lower()
            weekday_name = match.group(2).lower()
            target_weekday = self.WEEKDAY_MAP[weekday_name]
            
            if direction == 'last':
                # 找到上周的这一天
                days_diff = (ref_date.weekday() - target_weekday) % 7
                if days_diff == 0:
                    days_diff = 7
                target_date = ref_date - timedelta(days=days_diff)
            else:  # next
                # 找到下周的这一天
                days_diff = (target_weekday - ref_date.weekday()) % 7
                if days_diff == 0:
                    days_diff = 7
                target_date = ref_date + timedelta(days=days_diff)
            
            return target_date.strftime('%d %B %Y')
        
        return re.sub(pattern, replace, text, flags=re.IGNORECASE)


class ContentProcessor:
    """
    内容处理器 - 使用 LLM 智能提取事实
    """
    
    def __init__(self, llm_client=None):
        self.llm = llm_client
        self.temporal_normalizer = TemporalNormalizer()
    
    def process_conversation(
        self, 
        messages: List[Dict], 
        session_date: str,
        source_type: str = 'conversation'
    ) -> List[Dict]:
        """
        处理对话内容 - 时序标准化 + LLM 提取
        """
        processed_memories = []
        
        # 1. 先对每条消息进行时序标准化（关键！在 LLM 之前）
        normalized_messages = []
        for msg in messages:
            normalized_text = self.temporal_normalizer.normalize(
                msg.get('text', ''), 
                session_date
            )
            normalized_messages.append({
                'speaker': msg.get('speaker', 'Unknown'),
                'text': normalized_text
            })
        
        # 2. 合并标准化后的对话文本
        conversation_text = "\n".join([
            f"{msg['speaker']}: {msg['text']}"
            for msg in normalized_messages
        ])
        
        # 3. LLM 智能提取事实（基于已标准化的文本）
        facts = self._llm_extract_facts(conversation_text, session_date)
        
        # 2. 每条事实进行时序标准化并创建记忆
        for fact in facts:
            # 时序标准化
            normalized_fact = self.temporal_normalizer.normalize(fact, session_date)
            
            memory = {
                'content': normalized_fact,
                'source_type': source_type,
                'source_id': '',
                'created_at': session_date,
                'session_date': session_date,
                'speaker': self._extract_speaker(fact),
                'entities': self._extract_entities(normalized_fact),
                'topics': self._classify_topic(normalized_fact),
                'content_type': self._classify_content_type(normalized_fact),
                'temporal_info': self._extract_temporal_info(normalized_fact),
            }
            processed_memories.append(memory)
        
        return processed_memories
    
    def _llm_extract_facts(self, conversation_text: str, session_date: str) -> List[str]:
        """
        使用 LLM 从对话中提取有意义的事实
        """
        if not self.llm:
            # 降级：简单分割
            return [line for line in conversation_text.split('\n') if len(line) > 10]
        
        prompt = f"""从以下对话中提取所有客观事实和重要信息。

对话内容：
{conversation_text}

参考日期：{session_date}

提取规则：
1. 提取人物属性（身份、职业、关系状态、兴趣爱好等）
2. 提取具体事件（做了什么、什么时候、在哪里）
3. 提取计划和意图（将要做什么）
4. **关键：原样保留时间表达式**（如 yesterday, last year, next week）- 我会后续处理转换
5. 每个事实应该自包含，不依赖上下文
6. 用中文或英文输出（保持与原文相同语言）

**重要：不要改写或解释时间，原样保留 "yesterday", "last year" 等表达**

输出格式：JSON 字符串数组，每行一个事实

示例输出：
[
  "Caroline visited the LGBTQ support group yesterday",
  "Caroline is a transgender woman",
  "Melanie painted a sunrise last year",
  "Caroline is planning to adopt a child next week"
]

请提取所有相关事实（保留原始时间表达）："""

        try:
            response = self.llm.invoke_mistral(prompt, max_tokens=1000, temperature=0.0)
            import json
            facts = json.loads(response)
            if isinstance(facts, list):
                return [f for f in facts if isinstance(f, str) and len(f) > 5]
        except Exception as e:
            print(f"LLM 提取失败: {e}")
        
        # 降级处理
        return [line for line in conversation_text.split('\n') if len(line) > 10]
    
    def _extract_speaker(self, fact: str) -> str:
        """从事实中提取说话人"""
        match = re.match(r'^([A-Za-z]+):', fact)
        if match:
            return match.group(1)
        return ''
    
    def _extract_entities(self, text: str) -> List[str]:
        """提取实体（简化版，可以用 NER）"""
        # 大写单词可能是实体
        entities = re.findall(r'\b[A-Z][a-z]+\b', text)
        return list(set(entities))
    
    def _classify_topic(self, text: str) -> List[str]:
        """主题分类"""
        topics = []
        text_lower = text.lower()
        
        topic_keywords = {
            'identity': ['transgender', 'woman', 'man', 'gender', 'lgbtq'],
            'family': ['family', 'parent', 'child', 'adoption', 'single', 'married'],
            'work': ['work', 'job', 'career', 'company', 'project'],
            'health': ['health', 'doctor', 'therapy', 'support group'],
            'education': ['school', 'study', 'degree', 'education', 'learn'],
        }
        
        for topic, keywords in topic_keywords.items():
            if any(kw in text_lower for kw in keywords):
                topics.append(topic)
        
        return topics
    
    def _classify_content_type(self, text: str) -> str:
        """分类内容类型"""
        text_lower = text.lower()
        
        if any(w in text_lower for w in ['is', 'are', 'was', 'were', 'am']):
            if any(w in text_lower for w in ['i ', 'my ', 'me ']):
                return 'personal_info'
        
        if any(w in text_lower for w in ['plan', 'will', 'going to', 'want to']):
            return 'plan'
        
        if any(w in text_lower for w in ['like', 'love', 'hate', 'prefer', 'enjoy']):
            return 'preference'
        
        if any(w in text_lower for w in ['visited', 'went', 'did', 'happened', 'yesterday', 'last']):
            return 'event'
        
        return 'fact'
    
    def _extract_temporal_info(self, text: str) -> Dict:
        """提取时间信息"""
        info = {
            'has_date': False,
            'dates': []
        }
        
        # 查找日期模式
        date_patterns = [
            r'\d{1,2}\s+[A-Za-z]+\s+\d{4}',  # 7 May 2023
            r'\d{4}-\d{2}-\d{2}',  # 2023-05-07
        ]
        
        for pattern in date_patterns:
            matches = re.findall(pattern, text)
            if matches:
                info['has_date'] = True
                info['dates'].extend(matches)
        
        return info


# 使用示例
if __name__ == "__main__":
    processor = ContentProcessor()
    
    # 测试时序标准化
    test_cases = [
        ("I visited the group yesterday.", "8 May 2023"),
        ("Melanie painted a sunrise last year.", "8 May 2023"),
        ("We went camping last Saturday.", "7 May 2023"),
    ]
    
    print("时序标准化测试")
    print("=" * 60)
    
    for text, session_date in test_cases:
        # 调试
        parsed = processor.temporal_normalizer.parse_date(session_date)
        normalized = processor.temporal_normalizer.normalize(text, session_date)
        print(f"\n输入: {text}")
        print(f"参考: {session_date} (parsed: {parsed})")
        print(f"输出: {normalized}")
    
    # 测试对话处理
    print("\n" + "=" * 60)
    print("对话处理测试")
    print("=" * 60)
    
    messages = [
        {'speaker': 'Caroline', 'text': 'I visited the LGBTQ support group yesterday.'},
        {'speaker': 'Friend', 'text': 'How was it?'},
        {'speaker': 'Caroline', 'text': 'It was great. I am a transgender woman.'},
    ]
    
    memories = processor.process_conversation(messages, "8 May 2023")
    
    print(f"\n生成了 {len(memories)} 条记忆:\n")
    
    for i, mem in enumerate(memories, 1):
        print(f"{i}. {mem['content'][:80]}...")
        print(f"   Type: {mem['content_type']}")
        print(f"   Topics: {mem['topics']}")
        print(f"   Entities: {mem['entities']}")
        print(f"   Temporal: {mem['temporal_info']}")
        print()
