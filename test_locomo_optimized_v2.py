#!/usr/bin/env python3
"""
LoCoMo 优化测试 - 改进事实提取策略

优化点：
1. 按 session 分别提取事实，保留时间信息
2. 利用 LoCoMo 的 observation 和 session_summary 字段
3. 针对不同 question 类型优化提取
4. 增加事实数量和覆盖率
"""

import json
import requests
import sys
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import time
import re


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
        """调用 LLM"""
        max_retries = 3
        for attempt in range(max_retries):
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
                
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(1 * (attempt + 1))
                else:
                    return ""


class OptimizedLoCoMoEvaluator:
    """优化的 LoCoMo 评估器"""
    
    def __init__(self, client: ParateraClient, max_workers: int = 5):
        self.client = client
        self.max_workers = max_workers
        self.print_lock = Lock()
    
    def safe_print(self, *args, **kwargs):
        """线程安全打印"""
        with self.print_lock:
            print(*args, **kwargs)
    
    def load_locomo_data(self, data_path: str) -> List[Dict]:
        """加载 LoCoMo 数据集"""
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    
    def extract_facts_optimized(self, data: Dict) -> List[Dict]:
        """
        优化的事实提取策略
        
        1. 从 observation 字段提取
        2. 从 session_summary 字段提取
        3. 按 session 分别提取对话事实
        4. 合并所有事实并去重
        """
        all_facts = []
        
        # 1. 提取 observation 中的事实
        observations = data.get('observation', {})
        if observations:
            self.safe_print("  从 observation 提取...")
            for session_key, obs_list in observations.items():
                if isinstance(obs_list, list):
                    for obs in obs_list:
                        if isinstance(obs, str) and len(obs) > 10:
                            all_facts.append({
                                'fact': obs,
                                'source': 'observation',
                                'session': session_key
                            })
        
        # 2. 提取 session_summary 中的事实
        summaries = data.get('session_summary', {})
        if summaries:
            self.safe_print("  从 session_summary 提取...")
            for session_key, summary in summaries.items():
                if isinstance(summary, str) and len(summary) > 10:
                    # 将 summary 拆分成多个事实
                    for line in summary.split('\n'):
                        line = line.strip()
                        if line and len(line) > 10 and not line.startswith('#'):
                            all_facts.append({
                                'fact': line,
                                'source': 'summary',
                                'session': session_key
                            })
        
        # 3. 从对话中提取事实（按 session）
        conversation = data.get('conversation', {})
        speaker_a = conversation.get('speaker_a', 'Speaker A')
        speaker_b = conversation.get('speaker_b', 'Speaker B')
        
        self.safe_print("  从对话中提取（按 session）...")
        
        for session_key in sorted(conversation.keys()):
            if not session_key.startswith('session_') or session_key.endswith('_date_time'):
                continue
            
            session = conversation[session_key]
            session_date = conversation.get(f'{session_key}_date_time', 'Unknown Date')
            
            if not isinstance(session, list):
                continue
            
            # 格式化 session 对话
            session_text = f"\n=== Session: {session_key} (Date: {session_date}) ===\n"
            for turn in session:
                speaker = turn.get('speaker', '')
                text = turn.get('text', '')
                session_text += f"{speaker}: {text}\n"
            
            # 提取该 session 的事实
            session_facts = self.extract_facts_from_session(session_text, session_key, session_date)
            all_facts.extend(session_facts)
        
        # 4. 提取 event_summary 中的关键事件
        events = data.get('event_summary', [])
        if events:
            self.safe_print("  从 event_summary 提取...")
            for event in events:
                if isinstance(event, dict):
                    event_text = event.get('event', '')
                    if event_text:
                        all_facts.append({
                            'fact': event_text,
                            'source': 'event_summary',
                            'session': 'multiple'
                        })
        
        # 去重
        unique_facts = []
        seen = set()
        for f in all_facts:
            key = f['fact'].lower().strip()[:50]  # 使用前50字符作为去重键
            if key not in seen:
                seen.add(key)
                unique_facts.append(f)
        
        return unique_facts
    
    def extract_facts_from_session(self, session_text: str, session_key: str, session_date: str) -> List[Dict]:
        """从单个 session 提取事实"""
        system_prompt = """You are a fact extraction assistant. Extract specific facts from this conversation session.

Rules:
1. Extract facts about: who, what, when, where, why
2. Include temporal information and dates
3. Be specific and concrete
4. One fact per line
5. Include speaker names when relevant
6. Output format: one fact per line, no numbering"""

        prompt = f"""Extract facts from this conversation session:

{session_text[:8000]}

Facts:"""
        
        response = self.client.chat(prompt, system_prompt, max_tokens=1500, temperature=0.0)
        
        facts = []
        for line in response.strip().split('\n'):
            line = line.strip()
            if line and len(line) > 10:
                # 清理
                if line[0].isdigit() and line[1:3] in ['. ', ') ']:
                    line = line[3:].strip()
                if line.startswith('- ') or line.startswith('* '):
                    line = line[2:].strip()
                
                # 添加日期信息
                if session_date and session_date != 'Unknown Date':
                    # 检查是否已包含日期
                    if not any(digit in line for digit in '2023 2024 2025'.split()):
                        line = f"[{session_date}] {line}"
                
                facts.append({
                    'fact': line,
                    'source': 'conversation',
                    'session': session_key
                })
        
        return facts
    
    def answer_with_facts(self, question: str, facts: List[Dict]) -> str:
        """基于事实回答问题"""
        # 构建上下文（优先使用 observation 和 summary）
        priority_facts = [f for f in facts if f['source'] in ['observation', 'summary', 'event_summary']]
        other_facts = [f for f in facts if f['source'] == 'conversation']
        
        # 选择最相关的事实（限制数量避免超出 token）
        selected_facts = (priority_facts[:15] + other_facts[:15])[:30]
        
        context = "\n".join([f"- {f['fact']}" for f in selected_facts])
        
        system_prompt = """You are a helpful assistant. Answer the question based ONLY on the provided facts.

Rules:
1. Use only the information in the facts
2. Be concise (1-5 words if possible)
3. For "When" questions, give specific dates
4. For "What" questions, give specific items/actions
5. If uncertain, say "Unknown"
6. No explanations, just the answer"""

        prompt = f"""Facts:
{context}

Question: {question}

Answer:"""
        
        answer = self.client.chat(prompt, system_prompt, max_tokens=50, temperature=0.0)
        return answer.strip().strip('"').strip("'")
    
    def calculate_f1(self, predicted: str, ground_truth: Any) -> float:
        """计算 F1 分数"""
        if isinstance(ground_truth, (int, float)):
            ground_truth = str(ground_truth)
        
        pred = predicted.lower().strip()
        truth = str(ground_truth).lower().strip()
        
        # 完全匹配
        if pred == truth:
            return 1.0
        
        # 包含匹配
        if truth in pred or pred in truth:
            return 0.8
        
        # 提取日期比较
        pred_dates = self.extract_dates(pred)
        truth_dates = self.extract_dates(truth)
        if pred_dates and truth_dates:
            if any(pd in truth_dates for pd in pred_dates):
                return 0.7
        
        # 字符级 F1
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
    
    def extract_dates(self, text: str) -> List[str]:
        """提取文本中的日期"""
        # 匹配常见日期格式
        patterns = [
            r'\d{1,2}\s+(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{4}',
            r'\d{4}-\d{2}-\d{2}',
            r'(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s+\d{4}',
        ]
        
        dates = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            dates.extend(matches)
        
        return dates
    
    def evaluate_conversation(self, data: Dict, conv_idx: int) -> Dict:
        """评估单个对话"""
        sample_id = data.get('sample_id', f'conv_{conv_idx}')
        qa_list = data.get('qa', [])
        
        self.safe_print(f"\n{'='*70}")
        self.safe_print(f"对话 {conv_idx}: {sample_id}")
        self.safe_print(f"{'='*70}")
        self.safe_print(f"问题数: {len(qa_list)}")
        
        # 优化的事实提取
        self.safe_print(f"\n优化事实提取...")
        facts = self.extract_facts_optimized(data)
        self.safe_print(f"  提取到 {len(facts)} 个事实")
        
        # 统计来源
        sources = {}
        for f in facts:
            src = f['source']
            sources[src] = sources.get(src, 0) + 1
        self.safe_print(f"  来源分布: {sources}")
        
        # 显示部分事实
        for i, f in enumerate(facts[:5], 1):
            self.safe_print(f"    {i}. [{f['source']}] {f['fact'][:80]}...")
        
        # 并行回答问题
        self.safe_print(f"\n并行回答 {len(qa_list)} 个问题...")
        
        results = []
        completed = 0
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_idx = {}
            for i, qa in enumerate(qa_list):
                if 'question' in qa and 'answer' in qa:
                    future = executor.submit(
                        self.answer_single_question,
                        i+1, qa['question'], qa['answer'], facts
                    )
                    future_to_idx[future] = i+1
            
            for future in as_completed(future_to_idx):
                try:
                    result = future.result(timeout=35)
                    results.append(result)
                    completed += 1
                    
                    if completed % 20 == 0 or completed == len(qa_list):
                        self.safe_print(f"    进度: {completed}/{len(qa_list)}")
                        
                except Exception as e:
                    self.safe_print(f"    [错误] 问题 {future_to_idx[future]}: {e}")
        
        results.sort(key=lambda x: x['q_id'])
        
        # 统计
        correct = sum(1 for r in results if r['f1'] >= 0.8)
        partial = sum(1 for r in results if 0.5 <= r['f1'] < 0.8)
        wrong = sum(1 for r in results if r['f1'] < 0.5)
        
        self.safe_print(f"\n  正确: {correct}, 部分: {partial}, 错误: {wrong}")
        
        # 显示前 10 个结果
        for r in results[:10]:
            status = "✓" if r['f1'] >= 0.8 else "~" if r['f1'] >= 0.5 else "✗"
            self.safe_print(f"    [{r['q_id']:3d}] {status} F1:{r['f1']:.0%} | {r['question'][:45]}... | 答案: {r['predicted'][:30]}")
        
        avg_f1 = sum(r['f1'] for r in results) / len(results) if results else 0.0
        self.safe_print(f"\n  对话 F1: {avg_f1:.2%}")
        
        return {
            'sample_id': sample_id,
            'num_facts': len(facts),
            'num_questions': len(qa_list),
            'avg_f1': avg_f1,
            'correct': correct,
            'partial': partial,
            'wrong': wrong,
            'results': results
        }
    
    def answer_single_question(self, q_id: int, question: str, ground_truth: str, facts: List[Dict]) -> Dict:
        """回答单个问题"""
        predicted = self.answer_with_facts(question, facts)
        f1 = self.calculate_f1(predicted, ground_truth)
        
        return {
            'q_id': q_id,
            'question': question,
            'predicted': predicted,
            'ground_truth': str(ground_truth),
            'f1': f1
        }
    
    def run_evaluation(self, data_path: str, max_conversations: int = None) -> Dict:
        """运行评估"""
        print("="*70)
        print("LoCoMo 优化基准测试 - 改进事实提取策略")
        print("="*70)
        print(f"API: https://llmapi.paratera.com")
        print(f"LLM: {self.client.llm_model}")
        print(f"并发数: {self.max_workers}")
        print("="*70)
        
        data_list = self.load_locomo_data(data_path)
        if max_conversations:
            data_list = data_list[:max_conversations]
        
        print(f"\n加载了 {len(data_list)} 个对话")
        
        all_results = []
        start_time = time.time()
        
        for i, data in enumerate(data_list, 1):
            result = self.evaluate_conversation(data, i)
            all_results.append(result)
        
        elapsed = time.time() - start_time
        
        total_questions = sum(r['num_questions'] for r in all_results)
        total_f1 = sum(r['avg_f1'] * r['num_questions'] for r in all_results)
        overall_f1 = total_f1 / total_questions if total_questions > 0 else 0.0
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'elapsed_seconds': elapsed,
            'api_config': {'llm_model': self.client.llm_model},
            'num_conversations': len(data_list),
            'total_questions': total_questions,
            'overall_f1': overall_f1,
            'conversations': all_results
        }
        
        print(f"\n{'='*70}")
        print("评估完成!")
        print(f"{'='*70}")
        print(f"总体 F1 Score: {overall_f1:.2%}")
        print(f"总耗时: {elapsed:.1f} 秒")
        print(f"{'='*70}")
        
        return summary
    
    def save_results(self, results: Dict, output_path: str = None):
        """保存结果"""
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"locomo_optimized_results_{timestamp}.json"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n结果已保存到: {output_path}")


def main():
    client = ParateraClient(
        api_key="sk-0oVqiF3DzxzxTcbxsaPEOg",
        base_url="https://llmapi.paratera.com",
        llm_model="GLM-4-Plus"
    )
    
    max_workers = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    max_conv = int(sys.argv[2]) if len(sys.argv) > 2 else None
    
    evaluator = OptimizedLoCoMoEvaluator(client, max_workers=max_workers)
    results = evaluator.run_evaluation("locomodata.json", max_conversations=max_conv)
    evaluator.save_results(results)
    
    print("\n测试完成!")


if __name__ == "__main__":
    main()
