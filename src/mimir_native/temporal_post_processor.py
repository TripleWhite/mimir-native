"""
时序解析后处理 - 强制替换版本

策略：不信任 LLM，用规则强制替换相对时间
"""

import re
from datetime import datetime, timedelta
from typing import Dict, Optional


class TemporalPostProcessor:
    """
    时序后处理器 - 强制替换相对时间为绝对日期
    """
    
    # 星期映射
    WEEKDAY_MAP = {
        'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3,
        'friday': 4, 'saturday': 5, 'sunday': 6
    }
    
    def __init__(self):
        self.month_map = {
            'january': 1, 'february': 2, 'march': 3, 'april': 4,
            'may': 5, 'june': 6, 'july': 7, 'august': 8,
            'september': 9, 'october': 10, 'november': 11, 'december': 12
        }
    
    def parse_session_date(self, date_str: str) -> Optional[datetime]:
        """解析 session 日期"""
        if not date_str:
            return None
        
        formats = [
            '%d %B %Y',      # 8 May 2023
            '%d %b %Y',      # 8 May 2023
            '%Y-%m-%d',      # 2023-05-08
            '%B %d, %Y',     # May 8, 2023
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_str.strip(), fmt)
            except:
                continue
        
        # 尝试手动解析 "8 May 2023"
        match = re.match(r'(\d{1,2})\s+([A-Za-z]+)\s*,?\s*(\d{4})', date_str)
        if match:
            day, month_str, year = match.groups()
            month = self.month_map.get(month_str.lower())
            if month:
                return datetime(int(year), month, int(day))
        
        return None
    
    def process_fact(self, fact: str, session_date: str) -> str:
        """
        处理单个 fact，强制替换相对时间
        
        Args:
            fact: 原始事实文本
            session_date: 参考日期（如 "8 May 2023"）
        
        Returns:
            替换后的文本
        """
        ref_date = self.parse_session_date(session_date)
        if not ref_date:
            return fact
        
        result = fact
        
        # 1. yesterday → 昨天日期
        if 'yesterday' in result.lower():
            yesterday = ref_date - timedelta(days=1)
            yesterday_str = yesterday.strftime('%d %B %Y')
            result = re.sub(r'\byesterday\b', yesterday_str, result, flags=re.IGNORECASE)
        
        # 2. today → 当天日期
        if 'today' in result.lower():
            today_str = ref_date.strftime('%d %B %Y')
            result = re.sub(r'\btoday\b', today_str, result, flags=re.IGNORECASE)
        
        # 3. tomorrow → 明天日期
        if 'tomorrow' in result.lower():
            tomorrow = ref_date + timedelta(days=1)
            tomorrow_str = tomorrow.strftime('%d %B %Y')
            result = re.sub(r'\btomorrow\b', tomorrow_str, result, flags=re.IGNORECASE)
        
        # 4. last week → 上周同一天
        if 'last week' in result.lower():
            last_week = ref_date - timedelta(days=7)
            last_week_str = last_week.strftime('%d %B %Y')
            result = re.sub(r'\blast week\b', last_week_str, result, flags=re.IGNORECASE)
        
        # 5. next week → 下周同一天
        if 'next week' in result.lower():
            next_week = ref_date + timedelta(days=7)
            next_week_str = next_week.strftime('%d %B %Y')
            result = re.sub(r'\bnext week\b', next_week_str, result, flags=re.IGNORECASE)
        
        # 6. last year → 去年
        if 'last year' in result.lower():
            last_year = ref_date.year - 1
            result = re.sub(r'\blast year\b', str(last_year), result, flags=re.IGNORECASE)
        
        # 7. next year → 明年
        if 'next year' in result.lower():
            next_year = ref_date.year + 1
            result = re.sub(r'\bnext year\b', str(next_year), result, flags=re.IGNORECASE)
        
        # 8. last <weekday> → 上周某天
        result = self._replace_last_weekday(result, ref_date)
        
        # 9. next <weekday> → 下周某天
        result = self._replace_next_weekday(result, ref_date)
        
        return result
    
    def _replace_last_weekday(self, text: str, ref_date: datetime) -> str:
        """替换 last Monday/Tuesday 等"""
        pattern = r'\blast\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b'
        
        def replace(match):
            weekday_name = match.group(1).lower()
            target_weekday = self.WEEKDAY_MAP[weekday_name]
            
            # 找到上周的这一天
            days_diff = (ref_date.weekday() - target_weekday) % 7
            if days_diff == 0:
                days_diff = 7  # 如果今天就是这一天，取上周的
            target_date = ref_date - timedelta(days=days_diff)
            
            return target_date.strftime('%d %B %Y')
        
        return re.sub(pattern, replace, text, flags=re.IGNORECASE)
    
    def _replace_next_weekday(self, text: str, ref_date: datetime) -> str:
        """替换 next Monday/Tuesday 等"""
        pattern = r'\bnext\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b'
        
        def replace(match):
            weekday_name = match.group(1).lower()
            target_weekday = self.WEEKDAY_MAP[weekday_name]
            
            # 找到下周的这一天
            days_diff = (target_weekday - ref_date.weekday()) % 7
            if days_diff == 0:
                days_diff = 7
            target_date = ref_date + timedelta(days=days_diff)
            
            return target_date.strftime('%d %B %Y')
        
        return re.sub(pattern, replace, text, flags=re.IGNORECASE)


# 快速测试
if __name__ == "__main__":
    processor = TemporalPostProcessor()
    
    test_cases = [
        ("I visited the group yesterday.", "8 May 2023"),
        ("Melanie painted a sunrise last year.", "8 May 2023"),
        ("We went camping last Saturday.", "7 May 2023"),  # Sunday
        ("Meeting next Monday.", "8 May 2023"),  # Monday
        ("The event is tomorrow.", "8 May 2023"),
        ("I saw him today.", "8 May 2023"),
    ]
    
    print("Temporal Post-Processing Test")
    print("=" * 60)
    
    for fact, session_date in test_cases:
        result = processor.process_fact(fact, session_date)
        print(f"\nInput:  {fact}")
        print(f"Ref:    {session_date}")
        print(f"Output: {result}")
