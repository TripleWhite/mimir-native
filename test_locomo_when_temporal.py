#!/usr/bin/env python3
"""
LoCoMo When 类型问题专项测试 - 带时序标准化
"""

import json
import requests
import re
import sys
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import time

# 导入时序标准化器
from temporal_normalizer import TemporalNormalizer


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
            return ""


class TemporalLoCoMoEvaluator:
    """带时序标准化的 LoCoMo 评估器"""
    
    def __init__(self, client: ParateraClient, max_workers: int = 5):
        self.client = client
        self.max_workers = max_workers
        self.print_lock = Lock()
        self.normalizer = TemporalNormalizer()
    
    def safe_print(self, *args, **kwargs):
        """线程安全打印"""
        with self.print_lock:
            print(*args, **kwargs)
    
    def load_locomo_data(self, data_path: str) -> List[Dict]:
        """加载 LoCoMo 数据集"""
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    
    def extract_facts_with_temporal(self, data: Dict) -> List[Dict]:
        """提取带时序信息的事实"""
        all_facts = []
        
        # 获取会话日期映射
        conversation = data.get('conversation', {})
        session_dates = {}
        
        for key in conversation.keys():
            if key.endswith('_date_time'):
                session_key = key.replace('_date_time', '')
                session_dates[session_key] = conversation[key]
        
        # 设置会话日期
        self.normalizer.set_session_dates(session_dates)
        
        # 从 observation 提取
        observations = data.get('observation', {})
        for session_key, obs_list in observations.items():
            if isinstance(obs_list, list):
                session_date = session_dates.get(session_key, '')
                for obs in obs_list:
                    if isinstance(obs, str) and len(obs) > 10:
                        # 标准化时间
                        normalized = self.normalizer.normalize_fact(obs, session_date)
                        all_facts.append({
                            'fact': normalized,
                            'source': 'observation',
                            'session': session_key,
                            'date': session_date
                        })
        
        # 从对话中提取
        for session_key in sorted(conversation.keys()):
            if not session_key.startswith('session_') or session_key.endswith('_date_time'):
                continue
            
            session = conversation[session_key]
            session_date = session_dates.get(session_key, '')
            
            if not isinstance(session, list):
                continue
            
            # 合并会话文本
            session_text = ""
            for turn in session:
                speaker = turn.get('speaker', '')
                text = turn.get('text', '')
                session_text += f"{speaker}: {text}\n"
            
            # 提取该会话的事实（使用LLM）
            session_facts = self._extract_facts_from_text(session_text, session_key, session_date)
            
            # 标准化时间
            for fact in session_facts:
                fact['fact'] = self.normalizer.normalize_fact(fact['fact'], session_date)
                fact['date'] = session_date
            
            all_facts.extend(session_facts)
        
        return all_facts
    
    def _extract_facts_from_text(self, text: str, session_key: str, session_date: str) -> List[Dict]:
        """从文本提取事实"""
        system_prompt = """Extract facts from this conversation. Include dates when mentioned.
Rules:
1. One fact per line
2. Include temporal information (when, dates)
3. Be specific
4. No numbering"""

        prompt = f"Extract facts:\n\n{text[:5000]}\n\nFacts:"
        
        response = self.client.chat(prompt, system_prompt, max_tokens=1500, temperature=0.0)
        
        facts = []
        for line in response.strip().split('\n'):
            line = line.strip()
            if line and len(line) > 10:
                if line[0].isdigit() and line[1:3] in ['. ', ') ']:
                    line = line[3:].strip()
                if line.startswith('- ') or line.startswith('* '):
                    line = line[2:].strip()
                facts.append({
                    'fact': line,
                    'source': 'conversation',
                    'session': session_key
                })
        
        return facts
    
    def answer_when_with_temporal(self, question: str, facts: List[Dict]) -> str:
        """使用时序标准化回答 When 问题"""
        # 1. 构建时间线
        timeline = self.normalizer.build_timeline(facts)
        
        # 2. 直接使用时序标准化器回答
        answer = self.normalizer.answer_when_question(question, facts)
        
        if answer and answer != 'Unknown':
            return answer
        
        # 3. 如果时序标准化器无法回答，使用 LLM 辅助
        # 从时间线构建提示
        timeline_text = ""
        for entry in timeline[:15]:
            if entry['date']:
                date_str = entry['date'].strftime('%d %B %Y') if isinstance(entry['date'], datetime) else str(entry['date'])
                timeline_text += f"{date_str}: {entry['fact']}\n"
        
        system_prompt = """You are a temporal reasoning assistant.
Given a timeline of events, answer WHEN questions.
Rules:
1. Use the timeline to find the date
2. Convert relative times to absolute dates
3. Answer in format: "DD Month YYYY" or "Month YYYY"
4. If not found, say "Unknown"""

        prompt = f"""Timeline:
{timeline_text}

Question: {question}

Answer (date only):"""
        
        llm_answer = self.client.chat(prompt, system_prompt, max_tokens=50, temperature=0.0)
        
        # 标准化答案
        normalized = self.normalizer.normalize_answer(llm_answer.strip())
        
        return normalized if normalized else "Unknown"
    
    def calculate_f1(self, predicted: str, ground_truth: Any) -> float:
        """计算 F1 分数"""
        if isinstance(ground_truth, (int, float)):
            ground_truth = str(ground_truth)
        
        pred = str(predicted).lower().strip()
        truth = str(ground_truth).lower().strip()
        
        # 完全匹配
        if pred == truth:
            return 1.0
        
        # 包含匹配
        if truth in pred or pred in truth:
            return 0.8
        
        # 提取年份匹配
        pred_year = re.search(r'\b(20\d{2})\b', pred)
        truth_year = re.search(r'\b(20\d{2})\b', truth)
        
        if pred_year and truth_year:
            if pred_year.group(1) == truth_year.group(1):
                return 0.7  # 年份正确
        
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
    
    def evaluate_when_questions(self, data: Dict, conv_idx: int) -> Dict:
        """专门评估 When 类型问题"""
        sample_id = data.get('sample_id', f'conv_{conv_idx}')
        qa_list = data.get('qa', [])
        
        # 筛选 When 类型问题
        when_questions = [(i, qa) for i, qa in enumerate(qa_list) 
                         if qa.get('question', '').lower().startswith('when')]
        
        if not when_questions:
            return None
        
        self.safe_print(f"\n{'='*70}")
        self.safe_print(f"对话 {conv_idx}: {sample_id} - When 问题专项测试")
        self.safe_print(f"{'='*70}")
        self.safe_print(f"When 问题数: {len(when_questions)}/{len(qa_list)}")
        
        # 提取带时序的事实
        self.safe_print(f"\n提取带时序的事实...")
        facts = self.extract_facts_with_temporal(data)
        self.safe_print(f"  提取到 {len(facts)} 个事实")
        
        # 构建时间线
        timeline = self.normalizer.build_timeline(facts)
        self.safe_print(f"  构建时间线: {len(timeline)} 个事件")
        
        # 显示时间线前5个
        for entry in timeline[:5]:
            date_str = entry['date'].strftime('%Y-%m-%d') if isinstance(entry['date'], datetime) else str(entry['date'])
            self.safe_print(f"    [{date_str}] {entry['fact'][:60]}...")
        
        # 回答 When 问题
        self.safe_print(f"\n回答 {len(when_questions)} 个 When 问题...")
        
        results = []
        for idx, qa in when_questions:
            question = qa['question']
            ground_truth = qa['answer']
            
            # 使用时序标准化回答
            predicted = self.answer_when_with_temporal(question, facts)
            f1 = self.calculate_f1(predicted, ground_truth)
            
            results.append({
                'q_id': idx + 1,
                'question': question,
                'predicted': predicted,
                'ground_truth': str(ground_truth),
                'f1': f1
            })
            
            status = "✓" if f1 >= 0.8 else "~" if f1 >= 0.5 else "✗"
            self.safe_print(f"  [{idx+1:3d}] {status} F1:{f1:.0%}")
            self.safe_print(f"        Q: {question[:50]}...")
            self.safe_print(f"        A: {predicted[:40]}... (真实: {str(ground_truth)[:40]}...)")
        
        # 统计
        avg_f1 = sum(r['f1'] for r in results) / len(results) if results else 0.0
        correct = sum(1 for r in results if r['f1'] >= 0.8)
        
        self.safe_print(f"\n  正确: {correct}/{len(results)}")
        self.safe_print(f"  When 问题 F1: {avg_f1:.2%}")
        
        return {
            'sample_id': sample_id,
            'num_when_questions': len(when_questions),
            'avg_f1': avg_f1,
            'correct': correct,
            'results': results
        }
    
    def run_evaluation(self, data_path: str, max_conversations: int = None) -> Dict:
        """运行 When 问题专项评估"""
        print("="*70)
        print("LoCoMo When 类型问题专项测试 - 带时序标准化")
        print("="*70)
        print(f"API: https://llmapi.paratera.com")
        print(f"LLM: {self.client.llm_model}")
        print("="*70)
        
        data_list = self.load_locomo_data(data_path)
        if max_conversations:
            data_list = data_list[:max_conversations]
        
        print(f"\n加载了 {len(data_list)} 个对话")
        
        all_results = []
        start_time = time.time()
        
        for i, data in enumerate(data_list, 1):
            result = self.evaluate_when_questions(data, i)
            if result:
                all_results.append(result)
        
        elapsed = time.time() - start_time
        
        # 汇总
        total_when = sum(r['num_when_questions'] for r in all_results)
        total_f1 = sum(r['avg_f1'] * r['num_when_questions'] for r in all_results)
        overall_f1 = total_f1 / total_when if total_when > 0 else 0.0
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'elapsed_seconds': elapsed,
            'num_conversations': len(all_results),
            'total_when_questions': total_when,
            'overall_f1': overall_f1,
            'conversations': all_results
        }
        
        print(f"\n{'='*70}")
        print("When 问题专项测试完成!")
        print(f"{'='*70}")
        print(f"测试对话数: {len(all_results)}")
        print(f"When 问题总数: {total_when}")
        print(f"When 问题 F1 Score: {overall_f1:.2%}")
        print(f"总耗时: {elapsed:.1f} 秒")
        print(f"{'='*70}")
        
        return summary
    
    def save_results(self, results: Dict, output_path: str = None):
        """保存结果"""
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"locomo_when_temporal_results_{timestamp}.json"
        
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
    
    evaluator = TemporalLoCoMoEvaluator(client, max_workers=max_workers)
    results = evaluator.run_evaluation("locomodata.json", max_conversations=max_conv)
    evaluator.save_results(results)
    
    print("\n测试完成!")


if __name__ == "__main__":
    main()
