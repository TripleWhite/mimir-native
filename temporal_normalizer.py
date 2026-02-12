#!/usr/bin/env python3
"""
TemporalNormalizer - 时序标准化模块

功能：
1. 将相对时间转换为绝对时间
2. 构建时间线
3. 标准化日期格式
4. 处理时序推理
"""

import re
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import calendar


class TemporalNormalizer:
    """时序标准化器"""
    
    # 月份名称映射
    MONTH_MAP = {
        'january': 1, 'february': 2, 'march': 3, 'april': 4,
        'may': 5, 'june': 6, 'july': 7, 'august': 8,
        'september': 9, 'october': 10, 'november': 11, 'december': 12,
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'jun': 6,
        'jul': 7, 'aug': 8, 'sep': 9, 'sept': 9, 'oct': 10, 'nov': 11, 'dec': 12
    }
    
    # 相对时间模式
    RELATIVE_PATTERNS = {
        r'last year': 'last_year',
        r'this year': 'this_year',
        r'next year': 'next_year',
        r'last month': 'last_month',
        r'this month': 'this_month',
        r'next month': 'next_month',
        r'yesterday': 'yesterday',
        r'today': 'today',
        r'tomorrow': 'tomorrow',
        r'the week before (.+)': 'week_before',
        r'the week after (.+)': 'week_after',
        r'the day before (.+)': 'day_before',
        r'the day after (.+)': 'day_after',
        r'(\d+) years? ago': 'years_ago',
        r'(\d+) months? ago': 'months_ago',
        r'(\d+) weeks? ago': 'weeks_ago',
        r'(\d+) days? ago': 'days_ago',
    }
    
    def __init__(self, reference_date: str = None):
        """
        初始化时序标准化器
        
        Args:
            reference_date: 参考日期，用于解析相对时间
        """
        self.reference_date = self._parse_date(reference_date) if reference_date else None
        self.timeline = []
    
    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """解析日期字符串为 datetime 对象"""
        if not date_str:
            return None
        
        date_str = str(date_str).lower().strip()
        
        # 尝试多种格式
        formats = [
            '%d %B %Y',      # 7 May 2023
            '%B %d, %Y',     # May 7, 2023
            '%B %d %Y',      # May 7 2023
            '%Y-%m-%d',      # 2023-05-07
            '%d %b %Y',      # 7 May 2023
            '%b %d, %Y',     # May 7, 2023
            '%Y',            # 2023
            '%B %Y',         # May 2023
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except:
                continue
        
        # 尝试提取数字和月份
        return self._extract_date_components(date_str)
    
    def _extract_date_components(self, date_str: str) -> Optional[datetime]:
        """从文本中提取日期组件"""
        # 提取年份
        year_match = re.search(r'\b(20\d{2})\b', date_str)
        year = int(year_match.group(1)) if year_match else None
        
        # 提取月份
        month = None
        for month_name, month_num in self.MONTH_MAP.items():
            if month_name in date_str:
                month = month_num
                break
        
        # 提取日期
        day_match = re.search(r'\b(\d{1,2})\b', date_str)
        day = int(day_match.group(1)) if day_match else 1
        
        if year and month:
            try:
                return datetime(year, month, day)
            except:
                pass
        
        if year:
            return datetime(year, 1, 1)
        
        return None
    
    def _apply_relative_time(self, relative_expr: str, base_date: datetime) -> Optional[str]:
        """应用相对时间表达式"""
        relative_lower = relative_expr.lower().strip()
        
        # 匹配各种相对时间模式
        for pattern, action in self.RELATIVE_PATTERNS.items():
            match = re.search(pattern, relative_lower, re.IGNORECASE)
            if match:
                result = self._calculate_relative_date(action, match, base_date)
                if result:
                    return result
        
        return None
    
    def _calculate_relative_date(self, action: str, match: re.Match, base_date: datetime) -> Optional[str]:
        """计算相对日期"""
        if action == 'last_year':
            return str(base_date.year - 1)
        
        elif action == 'this_year':
            return str(base_date.year)
        
        elif action == 'next_year':
            return str(base_date.year + 1)
        
        elif action == 'yesterday':
            result = base_date - timedelta(days=1)
            return result.strftime('%d %B %Y')
        
        elif action == 'today':
            return base_date.strftime('%d %B %Y')
        
        elif action == 'tomorrow':
            result = base_date + timedelta(days=1)
            return result.strftime('%d %B %Y')
        
        elif action == 'week_before':
            ref_date_str = match.group(1).strip()
            ref_date = self._parse_date(ref_date_str)
            if ref_date:
                result = ref_date - timedelta(weeks=1)
                return result.strftime('%d %B %Y')
        
        elif action == 'week_after':
            ref_date_str = match.group(1).strip()
            ref_date = self._parse_date(ref_date_str)
            if ref_date:
                result = ref_date + timedelta(weeks=1)
                return result.strftime('%d %B %Y')
        
        elif action == 'day_before':
            ref_date_str = match.group(1).strip()
            ref_date = self._parse_date(ref_date_str)
            if ref_date:
                result = ref_date - timedelta(days=1)
                return result.strftime('%d %B %Y')
        
        elif action == 'day_after':
            ref_date_str = match.group(1).strip()
            ref_date = self._parse_date(ref_date_str)
            if ref_date:
                result = ref_date + timedelta(days=1)
                return result.strftime('%d %B %Y')
        
        elif action == 'years_ago':
            years = int(match.group(1))
            result = base_date.year - years
            return str(result)
        
        elif action == 'months_ago':
            months = int(match.group(1))
            month = base_date.month - months
            year = base_date.year
            while month <= 0:
                month += 12
                year -= 1
            return f"{calendar.month_name[month]} {year}"
        
        elif action == 'weeks_ago':
            weeks = int(match.group(1))
            result = base_date - timedelta(weeks=weeks)
            return result.strftime('%d %B %Y')
        
        elif action == 'days_ago':
            days = int(match.group(1))
            result = base_date - timedelta(days=days)
            return result.strftime('%d %B %Y')
        
        return None
    
    def normalize_fact(self, fact: str, session_date: str = None) -> str:
        """
        标准化单个事实中的时间
        
        Args:
            fact: 事实文本
            session_date: 会话日期，作为参考
        
        Returns:
            标准化后的事实文本
        """
        # 解析会话日期
        base_date = self._parse_date(session_date) if session_date else self.reference_date
        if not base_date:
            return fact
        
        # 查找相对时间表达式
        normalized = fact
        
        for pattern, action in self.RELATIVE_PATTERNS.items():
            matches = re.finditer(pattern, fact, re.IGNORECASE)
            for match in matches:
                relative_expr = match.group(0)
                absolute_date = self._apply_relative_time(relative_expr, base_date)
                
                if absolute_date:
                    normalized = normalized.replace(relative_expr, absolute_date)
        
        return normalized
    
    def build_timeline(self, facts: List[Dict]) -> List[Dict]:
        """
        从事实列表构建时间线
        
        Args:
            facts: 事实列表，每个包含 'fact', 'session', 'date' 等
        
        Returns:
            按时间排序的时间线
        """
        timeline = []
        
        for fact in facts:
            # 提取或推断日期
            date = self._extract_date_from_fact(fact)
            
            if date:
                timeline.append({
                    'date': date,
                    'fact': fact.get('fact', ''),
                    'source': fact.get('source', ''),
                    'session': fact.get('session', '')
                })
        
        # 按日期排序
        timeline.sort(key=lambda x: x['date'] if x['date'] else datetime.min)
        
        self.timeline = timeline
        return timeline
    
    def _extract_date_from_fact(self, fact: Dict) -> Optional[datetime]:
        """从事实中提取日期"""
        # 1. 直接获取 date 字段
        if 'date' in fact and fact['date']:
            return self._parse_date(fact['date'])
        
        # 2. 从 session 字段推断
        if 'session' in fact:
            # 尝试从 session 名解析日期
            session = fact['session']
            if hasattr(self, 'session_dates') and session in self.session_dates:
                return self._parse_date(self.session_dates[session])
        
        # 3. 从 fact 文本中提取日期
        fact_text = fact.get('fact', '')
        return self._parse_date(fact_text)
    
    def set_session_dates(self, session_dates: Dict[str, str]):
        """设置会话日期映射"""
        self.session_dates = session_dates
    
    def answer_when_question(self, question: str, facts: List[Dict]) -> Optional[str]:
        """
        专门回答 When 类型问题
        
        Args:
            question: 问题文本
            facts: 相关事实列表
        
        Returns:
            答案日期或 None
        """
        # 1. 构建时间线（如果还没有）
        if not self.timeline:
            self.build_timeline(facts)
        
        # 2. 在时间线中查找相关事实
        relevant_dates = []
        
        # 提取问题中的关键词
        keywords = self._extract_keywords(question)
        
        for entry in self.timeline:
            fact_text = entry['fact'].lower()
            if any(kw in fact_text for kw in keywords):
                relevant_dates.append(entry)
        
        # 3. 返回最可能的日期
        if relevant_dates:
            # 返回第一个匹配的日期（可以改进为选择最具体的）
            date = relevant_dates[0]['date']
            if date:
                return date.strftime('%d %B %Y')
        
        # 4. 如果没有找到，尝试从事实文本中直接提取
        for fact in facts:
            date_match = re.search(r'\d{1,2}\s+[A-Za-z]+\s+\d{4}', fact.get('fact', ''))
            if date_match:
                return date_match.group(0)
        
        return None
    
    def _extract_keywords(self, text: str) -> List[str]:
        """从文本中提取关键词"""
        # 去除停用词，提取关键名词
        stop_words = {'when', 'did', 'the', 'a', 'an', 'to', 'in', 'on', 'at', 'and', 'or'}
        words = re.findall(r'\b[A-Za-z]+\b', text.lower())
        return [w for w in words if w not in stop_words and len(w) > 2]
    
    def normalize_answer(self, answer: str) -> str:
        """标准化答案格式"""
        if not answer or answer == 'Unknown':
            return answer
        
        # 尝试解析并重新格式化
        date = self._parse_date(answer)
        if date:
            return date.strftime('%d %B %Y')
        
        return answer


# 测试函数
if __name__ == "__main__":
    # 测试时序标准化
    normalizer = TemporalNormalizer("7 May 2023")
    
    test_cases = [
        ("last year", "2022"),
        ("yesterday", "06 May 2023"),
        ("the week before 9 June 2023", "02 June 2023"),
        ("2 years ago", "2021"),
    ]
    
    print("测试 TemporalNormalizer:")
    print("-" * 50)
    
    for expr, expected in test_cases:
        result = normalizer._apply_relative_time(expr, normalizer.reference_date)
        status = "✓" if result == expected else "✗"
        print(f"{status} '{expr}' → '{result}' (期望: '{expected}')")
