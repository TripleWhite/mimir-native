"""
Temporal Resolver - 将相对时间解析为绝对时间

解决 LoCoMo 中的时序问题：
- "yesterday" → "7 May 2023"
- "last year" → "2022"
- "last Saturday" → "21 May 2023"
"""

import re
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class TemporalResolver:
    """
    时间解析器 - 将对话中的相对时间转换为绝对时间
    """
    
    def __init__(self):
        self.months = {
            'january': 1, 'february': 2, 'march': 3, 'april': 4,
            'may': 5, 'june': 6, 'july': 7, 'august': 8,
            'september': 9, 'october': 10, 'november': 11, 'december': 12,
            'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'jun': 6, 
            'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
        }
        
        self.weekdays = {
            'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3,
            'friday': 4, 'saturday': 5, 'sunday': 6
        }
    
    def resolve_relative_time(
        self, 
        relative_expr: str, 
        reference_date: datetime,
        context: Dict[str, Any] = None
    ) -> Optional[str]:
        """
        将相对时间表达式解析为绝对日期
        
        Args:
            relative_expr: 相对时间表达式（如 "yesterday", "last week"）
            reference_date: 参考日期（对话发生的日期）
            context: 额外上下文
            
        Returns:
            绝对日期字符串（如 "7 May 2023"）或 None
        """
        expr = relative_expr.lower().strip()
        
        # 直接匹配绝对日期（已经是绝对日期）
        absolute = self._parse_absolute_date(expr)
        if absolute:
            return absolute
        
        # 相对时间解析
        if expr in ['yesterday', 'the day before']:
            date = reference_date - timedelta(days=1)
            return date.strftime("%-d %B %Y")
        
        if expr in ['today', 'now']:
            return reference_date.strftime("%-d %B %Y")
        
        if expr in ['tomorrow', 'the next day']:
            date = reference_date + timedelta(days=1)
            return date.strftime("%-d %B %Y")
        
        # "last week", "this week", "next week"
        week_match = re.match(r'(last|this|next) week', expr)
        if week_match:
            direction = week_match.group(1)
            if direction == 'last':
                date = reference_date - timedelta(weeks=1)
            elif direction == 'next':
                date = reference_date + timedelta(weeks=1)
            else:
                date = reference_date
            # 返回那周的周一
            monday = date - timedelta(days=date.weekday())
            return f"the week of {monday.strftime('%-d %B %Y')}"
        
        # "last month", "this month", "next month"
        month_match = re.match(r'(last|this|next) month', expr)
        if month_match:
            direction = month_match.group(1)
            if direction == 'last':
                if reference_date.month == 1:
                    date = reference_date.replace(year=reference_date.year-1, month=12)
                else:
                    date = reference_date.replace(month=reference_date.month-1)
            elif direction == 'next':
                if reference_date.month == 12:
                    date = reference_date.replace(year=reference_date.year+1, month=1)
                else:
                    date = reference_date.replace(month=reference_date.month+1)
            else:
                date = reference_date
            return date.strftime("%B %Y")
        
        # "last year", "this year", "next year"
        year_match = re.match(r'(last|this|next) year', expr)
        if year_match:
            direction = year_match.group(1)
            if direction == 'last':
                year = reference_date.year - 1
            elif direction == 'next':
                year = reference_date.year + 1
            else:
                year = reference_date.year
            return str(year)
        
        # "last Saturday", "next Monday", etc.
        weekday_match = re.match(r'(last|this|next) (\w+)', expr)
        if weekday_match:
            direction, day_name = weekday_match.groups()
            day_name = day_name.lower().rstrip('s')  # handle plurals
            
            if day_name in self.weekdays:
                target_weekday = self.weekdays[day_name]
                current_weekday = reference_date.weekday()
                
                if direction == 'last':
                    # 找到上一个 target_weekday
                    days_diff = (current_weekday - target_weekday) % 7
                    if days_diff == 0:
                        days_diff = 7
                    date = reference_date - timedelta(days=days_diff)
                elif direction == 'next':
                    # 找到下一个 target_weekday
                    days_diff = (target_weekday - current_weekday) % 7
                    if days_diff == 0:
                        days_diff = 7
                    date = reference_date + timedelta(days=days_diff)
                else:  # this
                    days_diff = target_weekday - current_weekday
                    date = reference_date + timedelta(days=days_diff)
                
                return date.strftime("%-d %B %Y")
        
        # "a week ago", "two days ago", etc.
        ago_match = re.match(r'(\w+|\d+) (day|week|month|year)s? ago', expr)
        if ago_match:
            num, unit = ago_match.groups()
            try:
                num = int(num) if num.isdigit() else self._word_to_num(num)
                if unit == 'day':
                    date = reference_date - timedelta(days=num)
                elif unit == 'week':
                    date = reference_date - timedelta(weeks=num)
                elif unit == 'month':
                    # 简化处理
                    month = reference_date.month - num
                    year = reference_date.year
                    while month <= 0:
                        month += 12
                        year -= 1
                    date = reference_date.replace(year=year, month=month)
                elif unit == 'year':
                    date = reference_date.replace(year=reference_date.year - num)
                return date.strftime("%-d %B %Y")
            except:
                pass
        
        return None
    
    def _parse_absolute_date(self, text: str) -> Optional[str]:
        """检查是否已经是绝对日期"""
        # 匹配 "7 May 2023" 或 "May 7, 2023" 等格式
        patterns = [
            r'\b(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})\b',
            r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s+(\d{4})\b',
            r'\b(\d{4})[-/](\d{1,2})[-/](\d{1,2})\b',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(0)
        
        return None
    
    def _word_to_num(self, word: str) -> int:
        """将英文数字单词转为数字"""
        numbers = {
            'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
            'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
            'a': 1, 'an': 1
        }
        return numbers.get(word.lower(), 1)
    
    def extract_and_resolve(
        self, 
        text: str, 
        reference_date: datetime
    ) -> Dict[str, str]:
        """
        从文本中提取时间表达式并解析为绝对时间
        
        Args:
            text: 输入文本
            reference_date: 参考日期
            
        Returns:
            {原始表达式: 绝对日期}
        """
        resolved = {}
        
        # 常见相对时间模式
        patterns = [
            r'\b(yesterday|today|tomorrow|the day before|the next day)\b',
            r'\b(last|this|next) week\b',
            r'\b(last|this|next) month\b',
            r'\b(last|this|next) year\b',
            r'\b(last|this|next) (Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)s?\b',
            r'\b(\w+|\d+) (day|week|month|year)s? ago\b',
        ]
        
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                expr = match.group(0)
                absolute = self.resolve_relative_time(expr, reference_date)
                if absolute:
                    resolved[expr] = absolute
        
        return resolved


# 便捷函数
def resolve_temporal_expressions(text: str, reference_date: datetime) -> str:
    """
    将文本中的相对时间替换为绝对时间
    
    Args:
        text: 原始文本
        reference_date: 参考日期
        
    Returns:
        替换后的文本
    """
    resolver = TemporalResolver()
    resolved = resolver.extract_and_resolve(text, reference_date)
    
    result = text
    for expr, absolute in resolved.items():
        result = result.replace(expr, f"{expr} ({absolute})")
    
    return result


if __name__ == "__main__":
    # 测试
    resolver = TemporalResolver()
    
    # LoCoMo 场景：参考日期是 8 May 2023
    ref_date = datetime(2023, 5, 8)
    
    test_cases = [
        "yesterday",
        "last year", 
        "last Saturday",
        "the week before 25 May 2023",
    ]
    
    for tc in test_cases:
        result = resolver.resolve_relative_time(tc, ref_date)
        print(f"'{tc}' -> '{result}'")
