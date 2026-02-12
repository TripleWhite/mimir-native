#!/usr/bin/env python3
"""
LoCoMo When 问题快速测试 - 修复日期格式
"""

import json
import re
from datetime import datetime


def parse_session_date(date_str):
    """解析 LoCoMo 的会话日期格式"""
    # 格式: "1:56 pm on 8 May, 2023"
    match = re.search(r'(\d{1,2})[:\s]*(am|pm)?\s*on\s+(\d{1,2})\s+([A-Za-z]+),?\s+(\d{4})', date_str, re.IGNORECASE)
    if match:
        day = int(match.group(3))
        month_name = match.group(4).lower()
        year = int(match.group(5))
        
        month_map = {
            'january': 1, 'february': 2, 'march': 3, 'april': 4,
            'may': 5, 'june': 6, 'july': 7, 'august': 8,
            'september': 9, 'october': 10, 'november': 11, 'december': 12,
            'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
            'jul': 7, 'aug': 8, 'sep': 9, 'sept': 9, 'oct': 10, 'nov': 11, 'dec': 12
        }
        month = month_map.get(month_name)
        if month:
            return datetime(year, month, day)
    return None


def extract_dates_from_dialogue(text):
    """从对话文本中提取日期"""
    dates = []
    text_lower = text.lower()
    
    # 模式 1: "7 May 2023" 或 "May 7, 2023" 或 "7 May, 2023"
    pattern1 = r'\b(\d{1,2})\s+([a-z]+)[,\s]+(20\d{2})\b'
    for match in re.finditer(pattern1, text_lower):
        day = int(match.group(1))
        month_name = match.group(2)
        year = int(match.group(3))
        
        month_map = {
            'january': 1, 'february': 2, 'march': 3, 'april': 4,
            'may': 5, 'june': 6, 'july': 7, 'august': 8,
            'september': 9, 'october': 10, 'november': 11, 'december': 12
        }
        month = month_map.get(month_name)
        if month:
            try:
                dates.append(datetime(year, month, day))
            except:
                pass
    
    # 模式 2: 只有年份 "2022"
    pattern2 = r'\b(20\d{2})\b'
    for match in re.finditer(pattern2, text_lower):
        year = int(match.group(1))
        dates.append(datetime(year, 1, 1))
    
    # 模式 3: "June 2023"
    pattern3 = r'\b([a-z]+)\s+(20\d{2})\b'
    for match in re.finditer(pattern3, text_lower):
        month_name = match.group(1)
        year = int(match.group(2))
        
        month_map = {
            'january': 1, 'february': 2, 'march': 3, 'april': 4,
            'may': 5, 'june': 6, 'july': 7, 'august': 8,
            'september': 9, 'october': 10, 'november': 11, 'december': 12
        }
        month = month_map.get(month_name)
        if month:
            dates.append(datetime(year, month, 1))
    
    return dates


def main():
    with open('/tmp/mimir-review/mimir-native/locomodata.json', 'r') as f:
        data = json.load(f)
    
    conv = data[0]
    conversation = conv['conversation']
    qa_list = conv.get('qa', [])
    
    # 提取会话日期
    session_dates = {}
    for key in conversation.keys():
        if key.endswith('_date_time'):
            session_key = key.replace('_date_time', '')
            date_str = conversation[key]
            parsed = parse_session_date(date_str)
            if parsed:
                session_dates[session_key] = parsed
    
    print("="*70)
    print("修复版 When 问题测试")
    print("="*70)
    print(f"\n解析到 {len(session_dates)} 个会话日期")
    
    # 显示前5个
    for k, v in list(session_dates.items())[:5]:
        print(f"  {k}: {v.strftime('%Y-%m-%d')}")
    
    # 从对话中提取日期
    print("\n从对话中提取日期...")
    all_dates = []
    
    for session_key in sorted(conversation.keys()):
        if not session_key.startswith('session_') or session_key.endswith('_date_time'):
            continue
        
        session = conversation[session_key]
        if not isinstance(session, list):
            continue
        
        for turn in session:
            text = turn.get('text', '')
            dates = extract_dates_from_dialogue(text)
            for d in dates:
                all_dates.append({
                    'date': d,
                    'text': text[:80],
                    'session': session_key
                })
    
    print(f"  从对话中提取到 {len(all_dates)} 个日期事件")
    
    # 显示前10个
    for d in all_dates[:10]:
        print(f"  [{d['date'].strftime('%Y-%m-%d')}] {d['text'][:50]}...")
    
    # 筛选 When 问题并回答
    when_questions = [(i, qa) for i, qa in enumerate(qa_list) 
                     if qa.get('question', '').lower().startswith('when')]
    
    print(f"\nWhen 问题数: {len(when_questions)}")
    
    results = []
    for idx, qa in when_questions:
        question = qa['question']
        ground_truth = qa['answer']
        
        # 简单回答策略：从 ground_truth 提取日期模式来验证
        gt_lower = str(ground_truth).lower()
        
        # 计算 F1
        pred = "Unknown"  # 暂时使用 Unknown
        
        pred_lower = pred.lower()
        truth_lower = gt_lower
        
        if pred_lower == truth_lower:
            f1 = 1.0
        elif truth_lower in pred_lower or pred_lower in truth_lower:
            f1 = 0.8
        else:
            # 提取年份匹配
            pred_year = re.search(r'\b(20\d{2})\b', pred_lower)
            truth_year = re.search(r'\b(20\d{2})\b', truth_lower)
            if pred_year and truth_year and pred_year.group(1) == truth_year.group(1):
                f1 = 0.7
            else:
                # 字符 F1
                pred_chars = set(pred_lower)
                truth_chars = set(truth_lower)
                if pred_chars and truth_chars:
                    intersection = pred_chars & truth_chars
                    precision = len(intersection) / len(pred_chars)
                    recall = len(intersection) / len(truth_chars)
                    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
                else:
                    f1 = 0.0
        
        results.append({
            'q_id': idx + 1,
            'question': question,
            'ground_truth': str(ground_truth),
            'f1': f1
        })
    
    # 统计
    avg_f1 = sum(r['f1'] for r in results) / len(results) if results else 0
    
    print(f"\n当前 When 问题 F1 (基线): {avg_f1:.2%}")
    print(f"问题数: {len(results)}")
    
    # 分析 ground_truth 中的日期格式
    print("\nGround Truth 日期格式分析:")
    for r in results[:10]:
        gt = r['ground_truth']
        print(f"  [{r['q_id']:3d}] {gt}")


if __name__ == "__main__":
    main()
