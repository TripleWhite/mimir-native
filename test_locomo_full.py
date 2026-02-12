#!/usr/bin/env python3
"""
LoCoMo 完整基准测试 - 使用 Paratera API 和真实数据集

数据集: locomodata.json (10 个长对话，每个对话包含多个问题和答案)
"""

import json
import requests
import sys
from typing import List, Dict, Any, Optional
from datetime import datetime


class ParateraClient:
    """Paratera API 客户端"""
    
    def __init__(self, api_key: str, base_url: str = "https://llmapi.paratera.com",
                 embedding_model: str = "GLM-Embedding-3",
                 llm_model: str = "GLM-4-Plus"):
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        
        self.embed_url = f"{self.base_url}/v1/embeddings"
        self.chat_url = f"{self.base_url}/v1/chat/completions"
        
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def embed(self, text: str) -> Optional[List[float]]:
        """获取文本 embedding"""
        if not text or not text.strip():
            return None
        
        try:
            payload = {
                "model": self.embedding_model,
                "input": text,
                "encoding_format": "float"
            }
            
            response = requests.post(
                self.embed_url,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            if "data" in result and len(result["data"]) > 0:
                embedding = result["data"][0].get("embedding")
                if isinstance(embedding, list):
                    return embedding
            
            return None
            
        except Exception as e:
            print(f"[ERROR] Embed failed: {e}")
            return None
    
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
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"]
            
            return ""
            
        except Exception as e:
            print(f"[ERROR] Chat failed: {e}")
            return ""


class LoCoMoEvaluator:
    """LoCoMo 评估器"""
    
    def __init__(self, client: ParateraClient):
        self.client = client
        self.total_questions = 0
        self.total_f1 = 0.0
    
    def load_locomo_data(self, data_path: str) -> List[Dict]:
        """加载 LoCoMo 数据集"""
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    
    def format_conversation(self, data: Dict) -> str:
        """格式化对话为文本"""
        lines = []
        
        # 获取 conversation 对象
        conversation = data.get('conversation', {})
        
        # 获取说话者名称
        speaker_a = conversation.get('speaker_a', 'Speaker A')
        speaker_b = conversation.get('speaker_b', 'Speaker B')
        
        # 遍历所有会话
        for session_key in sorted(conversation.keys()):
            if not session_key.startswith('session_') or session_key.endswith('_date_time'):
                continue
            
            session = conversation[session_key]
            session_date = conversation.get(f'{session_key}_date_time', 'Unknown Date')
            
            lines.append(f"\n=== {session_key} ({session_date}) ===")
            
            # 遍历对话轮次
            for turn in session:
                speaker = turn.get('speaker', '')
                text = turn.get('text', '')
                lines.append(f"{speaker}: {text}")
        
        return "\n".join(lines)
    
    def extract_facts(self, conversation_text: str) -> List[str]:
        """从对话中提取事实"""
        system_prompt = """You are a fact extraction assistant. Extract specific facts from the conversation.

Rules:
1. Extract one fact per line
2. Include temporal information (dates, times) when available
3. Be specific and concrete
4. Use exact wording from conversation when possible
5. Output format: one fact per line, no numbering, no bullets"""

        prompt = f"""Extract all facts from this conversation:

{conversation_text[:8000]}  # 限制长度避免超出 token 限制

Facts:"""
        
        print("    提取事实...")
        response = self.client.chat(prompt, system_prompt, max_tokens=2000, temperature=0.0)
        
        # 解析事实
        facts = []
        for line in response.strip().split('\n'):
            line = line.strip()
            if line and not line.startswith('#') and len(line) > 10:
                # 去除编号和列表符号
                if line[0].isdigit() and line[1:3] in ['. ', ') ']:
                    line = line[3:].strip()
                if line.startswith('- ') or line.startswith('* '):
                    line = line[2:].strip()
                facts.append(line)
        
        return facts
    
    def answer_question(self, question: str, facts: List[str]) -> str:
        """基于事实回答问题"""
        context = "\n".join([f"- {fact}" for fact in facts[:30]])  # 限制事实数量
        
        system_prompt = """You are a helpful assistant. Answer the question based ONLY on the provided facts.

Rules:
1. Use only the information in the facts
2. Be concise (1-5 words if possible)
3. If the answer is not in the facts, say "Unknown"
4. No explanations, just the answer"""

        prompt = f"""Facts:
{context}

Question: {question}

Answer:"""
        
        answer = self.client.chat(prompt, system_prompt, max_tokens=100, temperature=0.0)
        return answer.strip().strip('"').strip("'")
    
    def normalize_answer(self, answer: str) -> str:
        """标准化答案用于比较"""
        # 转换为小写
        answer = answer.lower().strip()
        
        # 去除常见前缀
        prefixes = ['the ', 'a ', 'an ', 'in ', 'on ', 'at ']
        for prefix in prefixes:
            if answer.startswith(prefix):
                answer = answer[len(prefix):]
        
        # 去除标点
        answer = answer.strip('.,!?;:')
        
        return answer
    
    def calculate_f1(self, predicted: str, ground_truth: Any) -> float:
        """计算 F1 分数"""
        # 处理数字类型
        if isinstance(ground_truth, (int, float)):
            ground_truth = str(ground_truth)
        
        pred = self.normalize_answer(predicted)
        truth = self.normalize_answer(str(ground_truth))
        
        # 完全匹配
        if pred == truth:
            return 1.0
        
        # 包含匹配
        if truth in pred or pred in truth:
            return 0.8
        
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
    
    def evaluate_conversation(self, data: Dict, conv_idx: int) -> Dict:
        """评估单个对话"""
        sample_id = data.get('sample_id', f'conv_{conv_idx}')
        qa_list = data.get('qa', [])
        
        print(f"\n{'='*70}")
        print(f"对话 {conv_idx}: {sample_id}")
        print(f"{'='*70}")
        
        # 格式化对话
        conversation_text = self.format_conversation(data)
        
        # 统计信息
        num_sessions = sum(1 for k in data.keys() if k.startswith('session_') and not k.endswith('_date_time'))
        print(f"会话数: {num_sessions}")
        print(f"问题数: {len(qa_list)}")
        
        # 提取事实
        print(f"\n提取事实...")
        facts = self.extract_facts(conversation_text)
        print(f"提取到 {len(facts)} 个事实")
        
        # 显示前 5 个事实
        for i, fact in enumerate(facts[:5], 1):
            print(f"  {i}. {fact[:100]}...")
        
        # 评估每个问题
        print(f"\n回答问题...")
        results = []
        conv_f1 = 0.0
        
        for i, qa in enumerate(qa_list, 1):
            question = qa.get('question', '')
            ground_truth = qa.get('answer', '')
            category = qa.get('category', 0)
            
            predicted = self.answer_question(question, facts)
            f1 = self.calculate_f1(predicted, ground_truth)
            conv_f1 += f1
            
            status = "✓" if f1 >= 0.8 else "~" if f1 >= 0.5 else "✗"
            print(f"  [{i:2d}] {status} F1:{f1:.0%} | Q: {question[:50]}... | A: {predicted[:30]}...")
            
            results.append({
                'question': question,
                'ground_truth': str(ground_truth),
                'predicted': predicted,
                'f1': f1,
                'category': category
            })
        
        # 计算平均 F1
        avg_f1 = conv_f1 / len(qa_list) if qa_list else 0.0
        
        print(f"\n对话 F1: {avg_f1:.2%}")
        
        return {
            'sample_id': sample_id,
            'num_facts': len(facts),
            'num_questions': len(qa_list),
            'avg_f1': avg_f1,
            'results': results
        }
    
    def run_evaluation(self, data_path: str, max_conversations: int = None) -> Dict:
        """运行完整评估"""
        print("="*70)
        print("LoCoMo 完整基准测试 - 使用 Paratera API")
        print("="*70)
        print(f"API: https://llmapi.paratera.com")
        print(f"Embedding: {self.client.embedding_model}")
        print(f"LLM: {self.client.llm_model}")
        print(f"数据集: {data_path}")
        print("="*70)
        
        # 加载数据
        print("\n加载 LoCoMo 数据集...")
        data_list = self.load_locomo_data(data_path)
        
        if max_conversations:
            data_list = data_list[:max_conversations]
        
        print(f"加载了 {len(data_list)} 个对话")
        
        # 评估每个对话
        all_results = []
        total_f1 = 0.0
        total_questions = 0
        
        for i, data in enumerate(data_list, 1):
            result = self.evaluate_conversation(data, i)
            all_results.append(result)
            total_f1 += result['avg_f1'] * result['num_questions']
            total_questions += result['num_questions']
            
            # 累计统计
            self.total_questions = total_questions
            self.total_f1 = total_f1
        
        # 计算总体 F1
        overall_f1 = total_f1 / total_questions if total_questions > 0 else 0.0
        
        # 汇总
        summary = {
            'timestamp': datetime.now().isoformat(),
            'api_config': {
                'base_url': self.client.base_url,
                'embedding_model': self.client.embedding_model,
                'llm_model': self.client.llm_model
            },
            'num_conversations': len(data_list),
            'total_questions': total_questions,
            'overall_f1': overall_f1,
            'conversations': all_results
        }
        
        # 打印汇总
        print(f"\n{'='*70}")
        print("评估完成!")
        print(f"{'='*70}")
        print(f"总对话数: {len(data_list)}")
        print(f"总问题数: {total_questions}")
        print(f"总体 F1 Score: {overall_f1:.2%}")
        print(f"{'='*70}")
        
        return summary
    
    def save_results(self, results: Dict, output_path: str = None):
        """保存结果"""
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"locomo_full_results_{timestamp}.json"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n结果已保存到: {output_path}")


def main():
    """主函数"""
    # 初始化客户端
    client = ParateraClient(
        api_key="sk-0oVqiF3DzxzxTcbxsaPEOg",
        base_url="https://llmapi.paratera.com",
        embedding_model="GLM-Embedding-3",
        llm_model="GLM-4-Plus"
    )
    
    # 运行评估
    evaluator = LoCoMoEvaluator(client)
    
    # 可以限制测试的对话数量（用于快速测试）
    # 设置 max_conversations=2 测试前 2 个对话
    # 设置为 None 测试所有 10 个对话
    max_conv = 2 if len(sys.argv) > 1 and sys.argv[1] == '--quick' else None
    
    if max_conv:
        print(f"\n[快速模式] 只测试前 {max_conv} 个对话")
    
    results = evaluator.run_evaluation(
        data_path="locomodata.json",
        max_conversations=max_conv
    )
    
    # 保存结果
    evaluator.save_results(results)
    
    print("\n测试完成!")


if __name__ == "__main__":
    main()
