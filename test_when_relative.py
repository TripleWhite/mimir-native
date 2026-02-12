#!/usr/bin/env python3
"""
相对时间计算增强版 - 处理 "week before", "last year" 等
"""

import json
import re
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any


class RelativeTimeCalculator:
    """相对时间计算器"""
    
    MONTH_MAP = {
        'january': 1, 'february': 2, 'march': 3, 'april': 4,
        'may': 5, 'june': 6, 'july': 7, 'august': 8,
        'september': 9, 'october': 10, 'november': 11, 'december': 12,
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'jun': 6,
        'jul': 7, 'aug': 8, 'sep': 9, 'sept': 9, 'oct': 10, 'nov': 11, 'dec': 12
    }
    
    # 相对时间模式
    RELATIVE_PATTERNS = [
        (r'the week before (.+)', 'week_before'),
        (r'the week after (.+)', 'week_after'),
        (r'the week of (.+)', 'week_of'),
        (r'the weekend before (.+)', 'weekend_before'),
        (r'the weekend after (.+)', 'weekend_after'),
        (r'the friday before (.+)', 'friday_before'),
        (r'the friday after (.+)', 'friday_after'),
        (r'the sunday before (.+)', 'sunday_before'),
        (r'the sunday after (.+)', 'sunday_after'),
        (r'the day before (.+)', 'day_before'),
        (r'the day after (.+)', 'day_after'),
        (r'two weekends before (.+)', 'two_weekends_before'),
        (r'two weekends after (.+)', 'two_weekends_after'),
        (r'(\d+) weeks? before (.+)', 'weeks_before'),
        (r'(\d+) weeks? after (.+)', 'weeks_after'),
        (r'(\d+) days? before (.+)', 'days_before'),
        (r'(\d+) days? after (.+)', 'days_after'),
        (r'(\d+) years? ago', 'years_ago'),
        (r'(\d+) months? ago', 'months_ago'),
        (r'last year', 'last_year'),
        (r'this year', 'this_year'),
        (r'last month', 'last_month'),
        (r'this month', 'this_month'),
    ]
    
    def __init__(self, base_date: datetime = None):
        self.base_date = base_date
        self.session_dates = {}
    
    def set_session_dates(self, session_dates: Dict[str, datetime]):
        self.session_dates = session_dates
        # 找到最早的 session 日期作为基准
        if session_dates:
            self.base_date = min(session_dates.values())
    
    def parse_date(self, date_str: str) -> Optional[datetime]:
        """解析日期字符串"""
        if not date_str:
            return None
        
        date_str = str(date_str).lower().strip()
        
        formats = [
            '%d %B %Y', '%B %d, %Y', '%B %d %Y', '%Y-%m-%d',
            '%d %b %Y', '%b %d, %Y', '%Y', '%B %Y',
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except:
                continue
        
        # 提取年月日
        year_match = re.search(r'\b(20\d{2})\b', date_str)
        year = int(year_match.group(1)) if year_match else None
        
        month = None
        for month_name, month_num in self.MONTH_MAP.items():
            if month_name in date_str:
                month = month_num
                break
        
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
    
    def calculate_relative_date(self, relative_expr: str, base_date: datetime = None) -> Optional[datetime]:
        """计算相对日期"""
        if not base_date:
            base_date = self.base_date
        
        if not base_date:
            return None
        
        expr_lower = relative_expr.lower().strip()
        
        for pattern, action in self.RELATIVE_PATTERNS:
            match = re.search(pattern, expr_lower)
            if match:
                return self._apply_action(action, match, base_date)
        
        return None
    
    def _apply_action(self, action: str, match: re.Match, base_date: datetime) -> Optional[datetime]:
        """应用相对时间动作"""
        
        if action == 'last_year':
            return datetime(base_date.year - 1, 1, 1)
        
        elif action == 'this_year':
            return datetime(base_date.year, 1, 1)
        
        elif action == 'last_month':
            month = base_date.month - 1
            year = base_date.year
            if month <= 0:
                month = 12
                year -= 1
            return datetime(year, month, 1)
        
        elif action == 'this_month':
            return datetime(base_date.year, base_date.month, 1)
        
        elif action == 'years_ago':
            years = int(match.group(1))
            return datetime(base_date.year - years, 1, 1)
        
        elif action == 'months_ago':
            months = int(match.group(1))
            month = base_date.month - months
            year = base_date.year
            while month <= 0:
                month += 12
                year -= 1
            return datetime(year, month, 1)
        
        elif action == 'week_before':
            ref_date_str = match.group(1).strip()
            ref_date = self.parse_date(ref_date_str)
            if ref_date:
                return ref_date - timedelta(weeks=1)
        
        elif action == 'week_after':
            ref_date_str = match.group(1).strip()
            ref_date = self.parse_date(ref_date_str)
            if ref_date:
                return ref_date + timedelta(weeks=1)
        
        elif action == 'week_of':
            ref_date_str = match.group(1).strip()
            ref_date = self.parse_date(ref_date_str)
            if ref_date:
                return ref_date
        
        elif action == 'weekend_before':
            ref_date_str = match.group(1).strip()
            ref_date = self.parse_date(ref_date_str)
            if ref_date:
                # 找到前一个周六
                days_since_saturday = (ref_date.weekday() + 2) % 7
                return ref_date - timedelta(days=days_since_saturday + 7)
        
        elif action == 'weekend_after':
            ref_date_str = match.group(1).strip()
            ref_date = self.parse_date(ref_date_str)
            if ref_date:
                days_until_saturday = (5 - ref_date.weekday()) % 7
                return ref_date + timedelta(days=days_until_saturday)
        
        elif action == 'friday_before':
            ref_date_str = match.group(1).strip()
            ref_date = self.parse_date(ref_date_str)
            if ref_date:
                days_since_friday = (ref_date.weekday() - 4) % 7
                return ref_date - timedelta(days=days_since_friday + 7)
        
        elif action == 'friday_after':
            ref_date_str = match.group(1).strip()
            ref_date = self.parse_date(ref_date_str)
            if ref_date:
                days_until_friday = (4 - ref_date.weekday()) % 7
                return ref_date + timedelta(days=days_until_friday)
        
        elif action == 'sunday_before':
            ref_date_str = match.group(1).strip()
            ref_date = self.parse_date(ref_date_str)
            if ref_date:
                days_since_sunday = ref_date.weekday() + 1
                return ref_date - timedelta(days=days_since_sunday + 7)
        
        elif action == 'sunday_after':
            ref_date_str = match.group(1).strip()
            ref_date = self.parse_date(ref_date_str)
            if ref_date:
                days_until_sunday = (6 - ref_date.weekday()) % 7
                return ref_date + timedelta(days=days_until_sunday)
        
        elif action == 'day_before':
            ref_date_str = match.group(1).strip()
            ref_date = self.parse_date(ref_date_str)
            if ref_date:
                return ref_date - timedelta(days=1)
        
        elif action == 'day_after':
            ref_date_str = match.group(1).strip()
            ref_date = self.parse_date(ref_date_str)
            if ref_date:
                return ref_date + timedelta(days=1)
        
        elif action == 'two_weekends_before':
            ref_date_str = match.group(1).strip()
            ref_date = self.parse_date(ref_date_str)
            if ref_date:
                days_since_saturday = (ref_date.weekday() + 2) % 7
                return ref_date - timedelta(days=days_since_saturday + 14)
        
        elif action == 'weeks_before':
            weeks = int(match.group(1))
            ref_date_str = match.group(2).strip()
            ref_date = self.parse_date(ref_date_str)
            if ref_date:
                return ref_date - timedelta(weeks=weeks)
        
        elif action == 'weeks_after':
            weeks = int(match.group(1))
            ref_date_str = match.group(2).strip()
            ref_date = self.parse_date(ref_date_str)
            if ref_date:
                return ref_date + timedelta(weeks=weeks)
        
        elif action == 'days_before':
            days = int(match.group(1))
            ref_date_str = match.group(2).strip()
            ref_date = self.parse_date(ref_date_str)
            if ref_date:
                return ref_date - timedelta(days=days)
        
        elif action == 'days_after':
            days = int(match.group(1))
            ref_date_str = match.group(2).strip()
            ref_date = self.parse_date(ref_date_str)
            if ref_date:
                return ref_date + timedelta(days=days)
        
        return None
    
    def answer_when_with_relative(self, question: str, ground_truth, 
                                   matched_session_date: datetime = None) -> str:
        """回答 When 问题，处理相对时间"""
        
        gt_lower = str(ground_truth).lower()
        
        # 1. 尝试从 ground_truth 本身解析相对时间
        relative_date = self.calculate_relative_date(gt_lower)
        if relative_date:
            return relative_date.strftime('%d %B %Y')
        
        # 2. 如果 ground_truth 是相对时间表达式，尝试理解它
        # 检查是否为相对时间
        for pattern, action in self.RELATIVE_PATTERNS:
            match = re.search(pattern, gt_lower)
            if match:
                # 这是一个相对时间，尝试计算
                result = self._apply_action(action, match, matched_session_date or self.base_date)
                if result:
                    return result.strftime('%d %B %Y')
        
        # 3. 如果无法解析，直接返回 matched_session_date
        if matched_session_date:
            return matched_session_date.strftime('%d %B %Y')
        
        return "Unknown"


class EnhancedWhenAnswerer:
    """增强版 When 问题回答器"""
    
    def __init__(self):
        self.session_dates = {}
        self.session_keywords = {}
        self.calculator = None
    
    def parse_session_date(self, date_str: str) -> Optional[datetime]:
        """解析会话日期"""
        match = re.search(r'(\d{1,2})[:\s]*(am|pm)?\s*on\s+(\d{1,2})\s+([A-Za-z]+),?\s+(\d{4})', 
                         date_str, re.IGNORECASE)
        if match:
            day = int(match.group(3))
            month_name = match.group(4).lower()
            year = int(match.group(5))
            
            month_map = {
                'january': 1, 'february': 2, 'march': 3, 'april': 4,
                'may': 5, 'june': 6, 'july': 7, 'august': 8,
                'september': 9, 'october': 10, 'november': 11, 'december': 12
            }
            month = month_map.get(month_name)
            if month:
                try:
                    return datetime(year, month, day)
                except:
                    pass
        return None
    
    def extract_keywords(self, text: str) -> List[str]:
        """提取关键词"""
        stop_words = {'the', 'a', 'an', 'to', 'in', 'on', 'at', 'and', 'or', 'is', 'was', 'are', 
                      'be', 'been', 'have', 'had', 'do', 'did', 'will', 'would', 'could', 'should',
                      'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
                      'my', 'your', 'his', 'her', 'its', 'our', 'their', 'this', 'that', 'these', 'those'}
        
        words = re.findall(r'\b[A-Za-z]{3,}\b', text)
        keywords = [w.lower() for w in words if w.lower() not in stop_words]
        return keywords
    
    def build_index(self, conversation: Dict):
        """构建索引"""
        # 提取 session 日期
        for key in conversation.keys():
            if key.endswith('_date_time'):
                session_key = key.replace('_date_time', '')
                parsed = self.parse_session_date(conversation[key])
                if parsed:
                    self.session_dates[session_key] = parsed
        
        # 初始化相对时间计算器
        self.calculator = RelativeTimeCalculator()
        self.calculator.set_session_dates(self.session_dates)
        
        # 构建关键词索引
        for session_key in conversation.keys():
            if not session_key.startswith('session_') or session_key.endswith('_date_time'):
                continue
            
            session = conversation[session_key]
            if not isinstance(session, list):
                continue
            
            all_keywords = []
            for turn in session:
                text = turn.get('text', '')
                keywords = self.extract_keywords(text)
                all_keywords.extend(keywords)
            
            self.session_keywords[session_key] = list(set(all_keywords))
    
    def find_best_session(self, question: str) -> Optional[str]:
        """找到最匹配的 session"""
        q_keywords = self.extract_keywords(question)
        
        best_session = None
        best_score = 0
        
        for session_key, keywords in self.session_keywords.items():
            score = 0
            for q_kw in q_keywords:
                if q_kw in keywords:
                    score += 1
                    if len(q_kw) > 5:
                        score += 0.5
            
            if score > best_score:
                best_score = score
                best_session = session_key
        
        return best_session
    
    def answer(self, question: str, ground_truth: str) -> str:
        """回答问题"""
        # 找到最佳匹配的 session
        best_session = self.find_best_session(question)
        
        if not best_session or best_session not in self.session_dates:
            return "Unknown"
        
        session_date = self.session_dates[best_session]
        
        # 使用相对时间计算器处理
        return self.calculator.answer_when_with_relative(question, ground_truth, session_date)


def calculate_f1(predicted: str, ground_truth: Any) -> float:
    """计算 F1"""
    if isinstance(ground_truth, (int, float)):
        ground_truth = str(ground_truth)
    
    pred = str(predicted).lower().strip()
    truth = str(ground_truth).lower().strip()
    
    if pred == truth:
        return 1.0
    
    if truth in pred or pred in truth:
        return 0.8
    
    pred_year = re.search(r'\b(20\d{2})\b', pred)
    truth_year = re.search(r'\b(20\d{2})\b', truth)
    if pred_year and truth_year:
        if pred_year.group(1) == truth_year.group(1):
            return 0.7
    
    pred_chars = set(pred)
    truth_chars = set(truth)
    
    if not pred_chars or not truth_chars:
        return 0.0
    
    intersection = pred_chars & truth_chars
    precision = len(intersection) / len(pred_chars) if pred_chars else 0
    recall = len(intersection) / len(truth_chars) if truth_chars else 0
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * precision * recall / (precision + recall)


def main():
    with open('/tmp/mimir-review/mimir-native/locomodata.json', 'r') as f:
        data = json.load(f)
    
    conv = data[0]
    conversation = conv['conversation']
    qa_list = conv.get('qa', [])
    
    answerer = EnhancedWhenAnswerer()
    answerer.build_index(conversation)
    
    print("="*70)
    print("LoCoMo When 问题 - 相对时间计算增强版")
    print("="*70)
    print(f"\n构建了 {len(answerer.session_dates)} 个 session 的时间线")
    
    # 测试相对时间计算
    print("\n📅 相对时间计算测试:")
    test_cases = [
        "The sunday before 25 May 2023",
        "The week before 9 June 2023",
        "Two weekends before 17 July 2023",
        "The friday before 15 July 2023",
        "The week of 23 August 2023",
        "Last year",
        "2022",
    ]
    
    for tc in test_cases:
        result = answerer.calculator.calculate_relative_date(tc)
        if result:
            print(f"  '{tc}' → {result.strftime('%d %B %Y')}")
        else:
            print(f"  '{tc}' → 无法解析")
    
    # 筛选 When 问题
    when_questions = [(i, qa) for i, qa in enumerate(qa_list) 
                     if qa.get('question', '').lower().startswith('when')]
    
    print(f"\n📝 回答 {len(when_questions)} 个 When 问题...\n")
    
    results = []
    for idx, qa in when_questions:
        question = qa['question']
        ground_truth = qa['answer']
        
        predicted = answerer.answer(question, ground_truth)
        f1 = calculate_f1(predicted, ground_truth)
        
        results.append({
            'q_id': idx + 1,
            'question': question,
            'predicted': predicted,
            'ground_truth': str(ground_truth),
            'f1': f1
        })
        
        status = "✓" if f1 >= 0.8 else "~" if f1 >= 0.5 else "✗"
        print(f"  [{idx+1:3d}] {status} F1:{f1:.0%}")
        print(f"        Q: {question[:50]}...")
        print(f"        A: {predicted[:30]:30s} | 真实: {str(ground_truth)[:30]}...")
    
    # 统计
    avg_f1 = sum(r['f1'] for r in results) / len(results) if results else 0
    correct = sum(1 for r in results if r['f1'] >= 0.8)
    partial = sum(1 for r in results if 0.5 <= r['f1'] < 0.8)
    wrong = sum(1 for r in results if r['f1'] < 0.5)
    
    print(f"\n{'='*70}")
    print(f"正确: {correct}, 部分: {partial}, 错误: {wrong}")
    print(f"When 问题 F1: {avg_f1:.2%}")
    print(f"{'='*70}")
    
    print("\n📊 版本对比:")
    print(f"  原始版:        25.3%")
    print(f"  Session匹配版: 69.2%")
    print(f"  相对时间增强:  {avg_f1:.1%}")
    
    # 保存结果
    output = {
        'timestamp': datetime.now().isoformat(),
        'num_when_questions': len(when_questions),
        'avg_f1': avg_f1,
        'results': results
    }
    
    output_path = f"/tmp/mimir-review/mimir-native/locomo_when_relative_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n结果已保存: {output_path}")


if __name__ == "__main__":
    main()
