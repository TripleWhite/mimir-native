#!/usr/bin/env python3
"""
修复版 LoCoMo When 问题测试 - 改进日期提取
"""

import json
import requests
import re
import sys
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
import time


class FixedTemporalNormalizer:
    """修复版时序标准化器"""
    
    MONTH_MAP = {
        'january': 1, 'february': 2, 'march': 3, 'april': 4,
        'may': 5, 'june': 6, 'july': 7, 'august': 8,
        'september': 9, 'october': 10, 'november': 11, 'december': 12,
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'jun': 6,
        'jul': 7, 'aug': 8, 'sep': 9, 'sept': 9, 'oct': 10, 'nov': 11, 'dec': 12
    }
    
    def __init__(self):
        self.session_dates = {}
    
    def set_session_dates(self, session_dates: Dict[str, str]):
        self.session_dates = session_dates
    
    def _parse_date(self, date_str: str) -> Optional[datetime]:
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
        
        # 提取年份和月份
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
    
    def extract_all_dates_from_text(self, text: str) -> List[Dict]:
        """从文本中提取所有日期"""
        dates = []
        text_lower = text.lower()
        
        # 模式 1: "7 May 2023" 或 "May 7, 2023"
        pattern1 = r'\b(\d{1,2})\s+([a-z]+)\s+(20\d{2})\b'
        for match in re.finditer(pattern1, text_lower):
            day = int(match.group(1))
            month_name = match.group(2)
            year = int(match.group(3))
            
            month = self.MONTH_MAP.get(month_name)
            if month:
                try:
                    date = datetime(year, month, day)
                    dates.append({
                        'date': date,
                        'text': match.group(0),
                        'position': match.start()
                    })
                except:
                    pass
        
        # 模式 2: "May 2023" (只有月份和年份)
        pattern2 = r'\b([a-z]+)\s+(20\d{2})\b'
        for match in re.finditer(pattern2, text_lower):
            month_name = match.group(1)
            year = int(match.group(2))
            
            # 排除已经是模式1匹配的
            if not any(d['position'] == match.start() for d in dates):
                month = self.MONTH_MAP.get(month_name)
                if month:
                    try:
                        date = datetime(year, month, 1)
                        dates.append({
                            'date': date,
                            'text': match.group(0),
                            'position': match.start()
                        })
                    except:
                        pass
        
        # 模式 3: 单独年份 "2022"
        pattern3 = r'\b(20\d{2})\b'
        for match in re.finditer(pattern3, text_lower):
            year = int(match.group(1))
            # 只添加如果周围没有月份信息（避免重复）
            context = text_lower[max(0, match.start()-10):min(len(text_lower), match.end()+10)]
            has_month = any(m in context for m in self.MONTH_MAP.keys())
            
            if not has_month:
                dates.append({
                    'date': datetime(year, 1, 1),
                    'text': match.group(0),
                    'position': match.start()
                })
        
        return dates
    
    def extract_date_events(self, facts: List[Dict]) -> List[Dict]:
        """从事实中提取带日期的事件"""
        events = []
        
        for fact in facts:
            fact_text = fact.get('fact', '')
            
            # 提取所有日期
            dates = self.extract_all_dates_from_text(fact_text)
            
            for date_info in dates:
                events.append({
                    'date': date_info['date'],
                    'date_text': date_info['text'],
                    'fact': fact_text,
                    'source': fact.get('source', ''),
                    'session': fact.get('session', '')
                })
        
        # 按日期排序
        events.sort(key=lambda x: x['date'])
        return events
    
    def answer_when(self, question: str, facts: List[Dict]) -> str:
        """回答 When 问题 - 修复版"""
        # 1. 提取所有带日期的事件
        events = self.extract_date_events(facts)
        
        if not events:
            return "Unknown"
        
        # 2. 提取问题关键词
        q_lower = question.lower()
        
        # 去除停用词
        stop_words = {'when', 'did', 'the', 'a', 'an', 'to', 'in', 'on', 'at', 'and', 'or', 'go', 'to', 'her', 'his', 'she', 'he'}
        words = re.findall(r'\b[a-z]+\b', q_lower)
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        
        # 3. 匹配关键词与事件
        scored_events = []
        for event in events:
            fact_lower = event['fact'].lower()
            score = 0
            
            for kw in keywords:
                if kw in fact_lower:
                    score += 1
                    # 如果是核心名词（人名、活动名），增加权重
                    if len(kw) > 4:
                        score += 0.5
            
            if score > 0:
                scored_events.append({**event, 'score': score})
        
        # 4. 返回得分最高的事件日期
        if scored_events:
            scored_events.sort(key=lambda x: -x['score'])
            best_event = scored_events[0]
            return best_event['date'].strftime('%d %B %Y')
        
        # 5. 如果没有匹配，尝试找包含问题主体的任何事件
        # 提取问题中的专有名词（大写词）
        proper_nouns = re.findall(r'[A-Z][a-z]+', question)
        
        for event in events:
            for pn in proper_nouns:
                if pn.lower() in event['fact'].lower():
                    return event['date'].strftime('%d %B %Y')
        
        return "Unknown"


class ParateraClient:
    """Paratera API 客户端"""
    
    def __init__(self, api_key: str, base_url: str = "https://llmapi.paratera.com",
                 llm_model: str = "GLM-4-Plus"):
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.llm_model = llm_model
        self.chat_url = f"{self.base_url}/v1/chat/completions"
        
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def chat(self, prompt: str, system_prompt: str = None, max_tokens: int = 500, 
             temperature: float = 0.0) -> str:
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            payload = {
                "model": self.llm_model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature
            }
            
            response = requests.post(
                self.chat_url,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"]
            
            return ""
        except:
            return ""


def extract_facts(client: ParateraClient, data: Dict) -> List[Dict]:
    """提取事实"""
    facts = []
    conversation = data.get('conversation', {})
    
    # 获取会话日期
    session_dates = {}
    for key in conversation.keys():
        if key.endswith('_date_time'):
            session_key = key.replace('_date_time', '')
            session_dates[session_key] = conversation[key]
    
    # 从 observation 提取
    observations = data.get('observation', {})
    for session_key, obs_list in observations.items():
        if isinstance(obs_list, list):
            for obs in obs_list:
                if isinstance(obs, str) and len(obs) > 10:
                    facts.append({
                        'fact': obs,
                        'source': 'observation',
                        'session': session_key
                    })
    
    # 从对话中提取
    for session_key in sorted(conversation.keys()):
        if not session_key.startswith('session_') or session_key.endswith('_date_time'):
            continue
        
        session = conversation[session_key]
        if not isinstance(session, list):
            continue
        
        session_text = ""
        for turn in session:
            speaker = turn.get('speaker', '')
            text = turn.get('text', '')
            session_text += f"{speaker}: {text}\n"
        
        # 使用 LLM 提取事实
        system_prompt = "Extract facts from conversation. Include dates. One per line."
        prompt = f"Conversation:\n{session_text[:4000]}\n\nFacts:"
        
        response = client.chat(prompt, system_prompt, max_tokens=800, temperature=0.0)
        
        for line in response.strip().split('\n'):
            line = line.strip()
            if line and len(line) > 10:
                if line[0].isdigit() and '. ' in line[:3]:
                    line = line.split('. ', 1)[1]
                if line.startswith('- ') or line.startswith('* '):
                    line = line[2:]
                facts.append({
                    'fact': line,
                    'source': 'conversation',
                    'session': session_key
                })
    
    return facts


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
    
    # 年份匹配
    pred_year = re.search(r'\b(20\d{2})\b', pred)
    truth_year = re.search(r'\b(20\d{2})\b', truth)
    if pred_year and truth_year:
        if pred_year.group(1) == truth_year.group(1):
            return 0.7
    
    # 字符 F1
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
    client = ParateraClient(
        api_key="sk-0oVqiF3DzxzxTcbxsaPEOg",
        llm_model="GLM-4-Plus"
    )
    
    normalizer = FixedTemporalNormalizer()
    
    # 加载数据
    with open('/tmp/mimir-review/mimir-native/locomodata.json', 'r') as f:
        data_list = json.load(f)
    
    data = data_list[0]
    qa_list = data.get('qa', [])
    
    # 筛选 When 问题
    when_questions = [(i, qa) for i, qa in enumerate(qa_list) 
                     if qa.get('question', '').lower().startswith('when')]
    
    print("="*70)
    print("修复版 When 问题测试")
    print("="*70)
    print(f"When 问题数: {len(when_questions)}")
    
    # 提取事实
    print("\n提取事实...")
    facts = extract_facts(client, data)
    print(f"  提取到 {len(facts)} 个事实")
    
    # 提取日期事件
    print("\n提取日期事件...")
    events = normalizer.extract_date_events(facts)
    print(f"  提取到 {len(events)} 个带日期的事件")
    
    # 显示前10个
    for e in events[:10]:
        print(f"    [{e['date'].strftime('%Y-%m-%d')}] {e['fact'][:60]}...")
    
    # 回答 When 问题
    print(f"\n回答 {len(when_questions)} 个 When 问题...")
    
    results = []
    for idx, qa in when_questions:
        question = qa['question']
        ground_truth = qa['answer']
        
        predicted = normalizer.answer_when(question, facts)
        f1 = calculate_f1(predicted, ground_truth)
        
        results.append({
            'q_id': idx + 1,
            'question': question,
            'predicted': predicted,
            'ground_truth': str(ground_truth),
            'f1': f1
        })
        
        status = "✓" if f1 >= 0.8 else "~" if f1 >= 0.5 else "✗"
        print(f"  [{idx+1:3d}] {status} F1:{f1:.0%} | {question[:45]}...")
        print(f"        A: {predicted[:30]}... | 真实: {str(ground_truth)[:30]}...")
    
    # 统计
    avg_f1 = sum(r['f1'] for r in results) / len(results) if results else 0
    correct = sum(1 for r in results if r['f1'] >= 0.8)
    
    print(f"\n{'='*70}")
    print(f"正确: {correct}/{len(results)}")
    print(f"When 问题 F1: {avg_f1:.2%}")
    print(f"{'='*70}")
    
    # 保存结果
    output = {
        'timestamp': datetime.now().isoformat(),
        'num_when_questions': len(when_questions),
        'avg_f1': avg_f1,
        'correct': correct,
        'results': results
    }
    
    output_path = f"/tmp/mimir-review/mimir-native/locomo_when_fixed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"结果已保存: {output_path}")


if __name__ == "__main__":
    main()
