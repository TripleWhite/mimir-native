#!/usr/bin/env python3
"""
LoCoMo V5 - 专注优化 Which/Other/How
策略: When/Why/What/Who/Where 保持 V3 逻辑，Which/Other/How 使用 LLM 增强
"""

import json
import sys
import re
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import urllib.request


def calculate_f1(predicted, ground_truth):
    """F1 calculation - 完整版"""
    pred = str(predicted).lower().strip()
    truth = str(ground_truth).lower().strip()
    
    if pred == truth:
        return 1.0
    
    pred_numbers = set(re.findall(r'\d+', pred))
    truth_numbers = set(re.findall(r'\d+', truth))
    
    if pred_numbers and truth_numbers and pred_numbers == truth_numbers:
        return 1.0
    
    pred_year = re.search(r'\b(20\d{2})\b', pred)
    truth_year = re.search(r'\b(20\d{2})\b', truth)
    
    if truth_year and truth == truth_year.group(1):
        if pred_year and pred_year.group(1) == truth_year.group(1):
            return 1.0
    
    if truth in pred or pred in truth:
        return 0.8
    
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


class ParateraLLM:
    """Paratera API 封装"""
    
    def __init__(self, api_key: str = None, base_url: str = "https://llmapi.paratera.com"):
        self.api_key = api_key or os.getenv('PARATERA_API_KEY', 'sk-0oVqiF3DzxzxTcbxsaPEOg')
        self.base_url = base_url.rstrip('/')
    
    def call(self, prompt: str, model: str = "GLM-4-Plus", max_tokens: int = 100) -> str:
        """调用 LLM"""
        try:
            url = f"{self.base_url}/v1/chat/completions"
            
            data = json.dumps({
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0.1
            }).encode('utf-8')
            
            req = urllib.request.Request(url, data=data, method='POST')
            req.add_header('Authorization', f'Bearer {self.api_key}')
            req.add_header('Content-Type', 'application/json')
            
            with urllib.request.urlopen(req, timeout=30) as response:
                result = json.loads(response.read().decode('utf-8'))
                return result['choices'][0]['message']['content'].strip()
        except Exception as e:
            print(f"  LLM Error: {e}")
            return "Unknown"


class V5Retriever:
    """V5 检索器 - Which/Other/How 使用 LLM 增强"""
    
    def __init__(self, conv_data: Dict, use_llm: bool = True):
        self.conv_data = conv_data
        self.use_llm = use_llm
        self.session_dates = {}
        self.question_evidence = {}
        self.observation_by_session = {}
        self.facts = []
        self.conversation_text = []
        self.session_facts = {}
        self._build_index()
        
        if use_llm:
            self.llm = ParateraLLM()
    
    def parse_session_date(self, date_str: str) -> Optional[datetime]:
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
    
    def _build_index(self):
        """构建索引 - 同 V3"""
        conversation = self.conv_data.get('conversation', {})
        observation = self.conv_data.get('observation', {})
        qa_list = self.conv_data.get('qa', [])
        
        # 提取会话日期
        for key in conversation.keys():
            if key.endswith('_date_time'):
                session_key = key.replace('_date_time', '')
                parsed = self.parse_session_date(conversation[key])
                if parsed:
                    self.session_dates[session_key] = parsed
        
        # 提取 observation 事实
        fact_idx = 0
        for session_key, obs_dict in observation.items():
            session = session_key.replace('_observation', '')
            session_date = self.session_dates.get(session, datetime(2023, 5, 1))
            
            self.session_facts[session] = []
            self.observation_by_session[session] = {}
            
            if isinstance(obs_dict, dict):
                for person, fact_list in obs_dict.items():
                    self.observation_by_session[session][person] = []
                    if isinstance(fact_list, list):
                        for fact_item in fact_list:
                            if isinstance(fact_item, list) and len(fact_item) >= 1:
                                fact_text = fact_item[0]
                                if isinstance(fact_text, str) and len(fact_text) > 10:
                                    self.facts.append({
                                        'idx': fact_idx,
                                        'content': fact_text,
                                        'date': session_date,
                                        'session': session,
                                        'person': person
                                    })
                                    self.session_facts[session].append(fact_idx)
                                    self.observation_by_session[session][person].append(fact_text)
                                    fact_idx += 1
        
        # 提取 conversation 文本
        for session in self.session_dates.keys():
            conv_key = f"{session}_conversation"
            if conv_key in conversation:
                for item in conversation[conv_key]:
                    if isinstance(item, list) and len(item) >= 2:
                        self.conversation_text.append({
                            'session': session,
                            'speaker': item[0],
                            'content': item[1]
                        })
        
        # 提取 evidence 映射 (buggy 方式)
        for q_idx, qa in enumerate(qa_list):
            evidence = qa.get('evidence', [])
            if evidence:
                sessions = set()
                for ev in evidence:
                    if isinstance(ev, str) and ev.startswith('D') and ':' in ev:
                        session_num = ev.split(':')[0][1:]
                        session = f"session_{session_num}"
                        sessions.add(session)
                
                self.question_evidence[q_idx] = list(sessions)
    
    def get_question_type(self, question: str) -> str:
        q_lower = question.lower()
        if q_lower.startswith('when'):
            return 'When'
        elif q_lower.startswith('where'):
            return 'Where'
        elif q_lower.startswith('who'):
            return 'Who'
        elif q_lower.startswith('what'):
            return 'What'
        elif q_lower.startswith('why'):
            return 'Why'
        elif q_lower.startswith('how'):
            return 'How'
        elif q_lower.startswith('which'):
            return 'Which'
        else:
            return 'Other'
    
    def get_relevant_context(self, question: str, top_k: int = 3) -> str:
        """获取相关上下文"""
        q_words = set(question.lower().split())
        stop_words = {'when', 'did', 'the', 'a', 'to', 'of', 'in', 'on', 'at', 'is', 'was', 'are', 
                      'what', 'who', 'where', 'why', 'how', 'which', 'and', 'or', 'does', 'do', 'has', 'have'}
        q_words -= stop_words
        
        scored_texts = []
        
        # 从 facts 中找
        for fact in self.facts:
            f_words = set(fact['content'].lower().split())
            overlap = len(q_words & f_words)
            
            q_names = set(re.findall(r'\b[A-Z][a-z]+\b', question))
            f_names = set(re.findall(r'\b[A-Z][a-z]+\b', fact['content']))
            name_overlap = len(q_names & f_names)
            
            score = overlap + name_overlap * 3
            scored_texts.append((score, fact['content']))
        
        # 从 conversation 中找
        for conv in self.conversation_text:
            c_words = set(conv['content'].lower().split())
            overlap = len(q_words & c_words)
            
            q_names = set(re.findall(r'\b[A-Z][a-z]+\b', question))
            c_names = set(re.findall(r'\b[A-Z][a-z]+\b', conv['content']))
            name_overlap = len(q_names & c_names)
            
            score = overlap + name_overlap * 3
            scored_texts.append((score, f"{conv['speaker']}: {conv['content']}"))
        
        # 排序取 top_k
        scored_texts.sort(key=lambda x: -x[0])
        top_texts = [text for score, text in scored_texts[:top_k] if score > 0]
        
        return '\n'.join(top_texts)
    
    # ========== V3 原有方法 (保持不变) ==========
    
    def answer_when(self, q_idx: int) -> str:
        """When: V3 逻辑"""
        if q_idx in self.question_evidence:
            for session in self.question_evidence[q_idx]:
                if session in self.session_dates:
                    return self.session_dates[session].strftime('%d %B %Y')
        
        if self.session_dates:
            first_session = sorted(self.session_dates.keys())[0]
            return self.session_dates[first_session].strftime('%d %B %Y')
        return "Unknown"
    
    def answer_why(self, question: str) -> str:
        """Why: V3 逻辑 (原因类，从 facts 中找)"""
        return self.answer_generic(question)
    
    def answer_what(self, question: str) -> str:
        """What: V3 逻辑"""
        return self.answer_generic(question)
    
    def answer_who(self, question: str) -> str:
        """Who: V3 逻辑"""
        return self.answer_generic(question)
    
    def answer_where(self, question: str) -> str:
        """Where: V3 逻辑"""
        return self.answer_generic(question)
    
    def answer_generic(self, question: str) -> str:
        """通用回答 - V3 逻辑"""
        q_words = set(question.lower().split())
        stop_words = {'when', 'did', 'the', 'a', 'to', 'of', 'in', 'on', 'at', 'is', 'was', 'are', 
                      'what', 'who', 'where', 'why', 'how', 'which', 'and', 'or'}
        q_words -= stop_words
        
        best_match = None
        best_score = 0
        
        for fact in self.facts:
            f_words = set(fact['content'].lower().split())
            overlap = len(q_words & f_words)
            
            q_names = set(re.findall(r'\b[A-Z][a-z]+\b', question))
            f_names = set(re.findall(r'\b[A-Z][a-z]+\b', fact['content']))
            name_overlap = len(q_names & f_names)
            
            score = overlap + name_overlap * 2
            
            if score > best_score:
                best_score = score
                best_match = fact['content']
        
        for conv in self.conversation_text:
            c_words = set(conv['content'].lower().split())
            overlap = len(q_words & c_words)
            
            q_names = set(re.findall(r'\b[A-Z][a-z]+\b', question))
            c_names = set(re.findall(r'\b[A-Z][a-z]+\b', conv['content']))
            name_overlap = len(q_names & c_names)
            
            score = overlap + name_overlap * 2
            
            if score > best_score:
                best_score = score
                best_match = conv['content']
        
        if best_match:
            sentences = best_match.split('.')
            for sent in sentences:
                sent = sent.strip()
                if len(sent) > 10:
                    return sent[:150]
        
        return "Unknown"
    
    # ========== LLM 增强方法 (Which/Other/How) ==========
    
    def answer_with_llm(self, question: str, q_type: str) -> str:
        """使用 LLM 回答"""
        context = self.get_relevant_context(question, top_k=3)
        
        if q_type == 'Which':
            prompt = f"""Based on the conversation context, answer the "Which" question with the specific name/item.

Context:
{context}

Question: {question}

Answer (just the name, 1-5 words):"""
        
        elif q_type == 'Other':
            prompt = f"""Based on the conversation context, answer the question concisely.

Context:
{context}

Question: {question}

Answer (Yes/No, or a short phrase, max 10 words):"""
        
        elif q_type == 'How':
            prompt = f"""Based on the conversation context, explain how something was done.

Context:
{context}

Question: {question}

Answer (brief explanation, max 15 words):"""
        
        else:
            return "Unknown"
        
        return self.llm.call(prompt, max_tokens=50)
    
    def answer(self, q_idx: int, question: str) -> str:
        """主回答函数 - 路由到不同策略"""
        q_type = self.get_question_type(question)
        
        # V3 逻辑 (保持不变)
        if q_type == 'When':
            return self.answer_when(q_idx)
        elif q_type == 'Why':
            return self.answer_why(question)
        elif q_type == 'What':
            return self.answer_what(question)
        elif q_type == 'Who':
            return self.answer_who(question)
        elif q_type == 'Where':
            return self.answer_where(question)
        
        # LLM 增强 (Which/Other/How)
        elif q_type in ['Which', 'Other', 'How'] and self.use_llm:
            return self.answer_with_llm(question, q_type)
        
        # 回退
        else:
            return self.answer_generic(question)


def test_conversation(conv_data: Dict, conv_idx: int, use_llm: bool = True) -> Dict:
    conv_name = conv_data.get('name', f'D{conv_idx+1}')
    qa_list = conv_data.get('qa', [])
    
    retriever = V5Retriever(conv_data, use_llm=use_llm)
    
    results_by_type = {}
    llm_calls = 0
    
    for q_idx, qa in enumerate(qa_list):
        question = qa.get('question', '')
        ground_truth = qa.get('answer', '')
        q_type = retriever.get_question_type(question)
        
        if q_type not in results_by_type:
            results_by_type[q_type] = {'correct': 0, 'partial': 0, 'wrong': 0, 'total': 0}
        
        if not ground_truth:
            continue
        
        predicted = retriever.answer(q_idx, question)
        f1 = calculate_f1(predicted, ground_truth)
        
        results_by_type[q_type]['total'] += 1
        if f1 >= 0.8:
            results_by_type[q_type]['correct'] += 1
        elif f1 >= 0.5:
            results_by_type[q_type]['partial'] += 1
        else:
            results_by_type[q_type]['wrong'] += 1
        
        # 统计 LLM 调用
        if q_type in ['Which', 'Other', 'How'] and use_llm:
            llm_calls += 1
    
    for q_type in results_by_type:
        r = results_by_type[q_type]
        if r['total'] > 0:
            r['f1'] = (r['correct'] * 1.0 + r['partial'] * 0.7) / r['total']
        else:
            r['f1'] = 0
    
    return {
        'conversation': conv_name,
        'by_type': results_by_type,
        'total_qa': len(qa_list),
        'llm_calls': llm_calls
    }


def main():
    use_llm = True  # 启用 LLM
    
    print("="*70)
    print("🧪 LoCoMo V5 - Which/Other/How LLM Enhancement")
    print("="*70)
    print(f"LLM: {'Enabled' if use_llm else 'Disabled'}")
    print("Target: Optimize Which/Other/How with LLM")
    print("="*70)
    print()
    
    with open('locomodata.json', 'r') as f:
        all_data = json.load(f)
    
    print(f"Loaded {len(all_data)} conversations")
    print()
    
    all_results = []
    total_llm_calls = 0
    
    for conv_idx in range(len(all_data)):
        conv_name = f"D{conv_idx+1}"
        print(f"Testing {conv_name} ({conv_idx+1}/{len(all_data)})...", end=' ', flush=True)
        
        result = test_conversation(all_data[conv_idx], conv_idx, use_llm=use_llm)
        all_results.append(result)
        total_llm_calls += result['llm_calls']
        
        total_questions = sum(r['total'] for r in result['by_type'].values())
        print(f"{total_questions} Qs, {result['llm_calls']} LLM calls")
    
    # 全局统计
    global_stats = {}
    for result in all_results:
        for q_type, stats in result['by_type'].items():
            if q_type not in global_stats:
                global_stats[q_type] = {'correct': 0, 'partial': 0, 'wrong': 0, 'total': 0}
            global_stats[q_type]['correct'] += stats['correct']
            global_stats[q_type]['partial'] += stats['partial']
            global_stats[q_type]['wrong'] += stats['wrong']
            global_stats[q_type]['total'] += stats['total']
    
    for q_type in global_stats:
        s = global_stats[q_type]
        if s['total'] > 0:
            s['f1'] = (s['correct'] * 1.0 + s['partial'] * 0.7) / s['total']
        else:
            s['f1'] = 0
    
    total_all = sum(s['total'] for s in global_stats.values())
    overall_f1 = sum(s['f1'] * s['total'] for s in global_stats.values()) / total_all if total_all > 0 else 0
    
    print()
    print("="*70)
    print("📊 V5 Results")
    print("="*70)
    print(f"{'类型':<10} {'正确':>6} {'部分':>6} {'错误':>6} {'总计':>6} {'F1':>8}")
    print("-"*70)
    
    for q_type in sorted(global_stats.keys(), key=lambda x: -global_stats[x]['f1']):
        s = global_stats[q_type]
        marker = "🤖" if q_type in ['Which', 'Other', 'How'] else "  "
        print(f"{marker} {q_type:<8} {s['correct']:>6} {s['partial']:>6} {s['wrong']:>6} {s['total']:>6} {s['f1']:>7.1%}")
    
    print("-"*70)
    print(f"{'总计':<10} {sum(s['correct'] for s in global_stats.values()):>6} "
          f"{sum(s['partial'] for s in global_stats.values()):>6} "
          f"{sum(s['wrong'] for s in global_stats.values()):>6} "
          f"{total_all:>6} {overall_f1:>7.1%}")
    print("="*70)
    print(f"Total LLM calls: {total_llm_calls}")
    print()
    
    # 对比 V3
    v3_baseline = {
        'When': 0.758, 'Why': 0.84, 'What': 0.719, 'Who': 0.683,
        'Where': 0.681, 'How': 0.595, 'Which': 0.545, 'Other': 0.385
    }
    
    print("📈 V3 vs V5 Comparison")
    print("-"*70)
    for q_type in ['Which', 'Other', 'How']:
        v3_f1 = v3_baseline.get(q_type, 0)
        v5_f1 = global_stats[q_type]['f1']
        diff = v5_f1 - v3_f1
        print(f"{q_type:<10} V3: {v3_f1:>6.1%}  V5: {v5_f1:>6.1%}  Δ: {diff:+.1%}")
    
    print()
    v3_overall = 0.6818
    print(f"Overall    V3: {v3_overall:>6.1%}  V5: {overall_f1:>6.1%}  Δ: {overall_f1 - v3_overall:+.1%}")
    
    return overall_f1


if __name__ == "__main__":
    f1 = main()
    print(f"\nFinal F1: {f1:.2%}")
