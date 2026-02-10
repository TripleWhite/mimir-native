"""
Mimir 实用评估体系 - 关注真实用户体验，而非 benchmark 分数
"""

import json
import time
from typing import List, Dict, Any
from dataclasses import dataclass
from enum import Enum


class EvalMetric(Enum):
    """评估指标"""
    INGESTION_RATE = "ingestion_rate"      # 记忆写入成功率
    RETRIEVAL_PRECISION = "retrieval_precision"  # 检索精准度
    CONTEXT_RELEVANCE = "context_relevance"  # 上下文相关性
    USER_SATISFACTION = "user_satisfaction"  # 用户满意度
    END_TO_END_LATENCY = "end_to_end_latency"  # 端到端延迟


@dataclass
class EvalResult:
    """评估结果"""
    metric: str
    score: float  # 0-1
    details: Dict[str, Any]
    

class MimirEvaluator:
    """
    Mimir 实用评估器
    
    评估维度：
    1. 记忆提取 - 能否从对话中提取有用信息
    2. 检索质量 - 能否在需要时找到相关信息
    3. 上下文增强 - 提供的上下文是否有帮助
    4. 用户体验 - 整体使用感受
    """
    
    def __init__(self, mimir_memory, llm_client):
        self.memory = mimir_memory
        self.llm = llm_client
    
    # ========== 1. 记忆提取评估 ==========
    
    def evaluate_ingestion(
        self,
        test_conversations: List[Dict],
        expected_facts: List[str]
    ) -> EvalResult:
        """
        评估记忆写入能力
        
        Args:
            test_conversations: 测试对话列表
            expected_facts: 期望提取的关键事实
            
        Returns:
            {
                'metric': 'ingestion_rate',
                'score': 0.85,
                'details': {
                    'total_messages': 100,
                    'memories_created': 45,
                    'expected_facts_found': 17/20
                }
            }
        """
        total_messages = 0
        memories_created = 0
        facts_found = 0
        
        for conv in test_conversations:
            messages = conv.get('messages', [])
            total_messages += len(messages)
            
            # 摄入对话
            for msg in messages:
                result = self.memory.add_content(
                    msg['text'],
                    content_type='text',
                    user_id='eval_test'
                )
                memories_created += len(result) if isinstance(result, list) else 0
        
        # 检查期望事实是否被提取
        for fact in expected_facts:
            # 检索这个 fact
            results = self.memory.query(fact, user_id='eval_test', top_k=3)
            if results:
                facts_found += 1
        
        ingestion_rate = memories_created / max(total_messages, 1)
        fact_coverage = facts_found / max(len(expected_facts), 1)
        
        return EvalResult(
            metric='ingestion_rate',
            score=(ingestion_rate + fact_coverage) / 2,
            details={
                'total_messages': total_messages,
                'memories_created': memories_created,
                'expected_facts': len(expected_facts),
                'facts_found': facts_found,
                'ingestion_rate': ingestion_rate,
                'fact_coverage': fact_coverage
            }
        )
    
    # ========== 2. 检索质量评估 ==========
    
    def evaluate_retrieval(
        self,
        test_queries: List[Dict]
    ) -> EvalResult:
        """
        评估检索质量
        
        Args:
            test_queries: [
                {
                    'query': 'Caroline 的身份',
                    'relevant_keywords': ['transgender', 'woman'],
                    'expected_memory_contains': 'transgender woman'
                }
            ]
            
        返回:
            precision@3: 前3条结果中相关的比例
        """
        total_queries = len(test_queries)
        relevant_count = 0
        details = []
        
        for test in test_queries:
            query = test['query']
            keywords = test.get('relevant_keywords', [])
            expected_contains = test.get('expected_memory_contains', '')
            
            # 执行检索
            results = self.memory.query(query, user_id='eval_test', top_k=3)
            
            # 检查相关性
            query_relevant = 0
            for r in results:
                content = str(r.memory.content if hasattr(r, 'memory') else r).lower()
                # 检查是否包含关键词
                if any(kw.lower() in content for kw in keywords):
                    if not expected_contains or expected_contains.lower() in content:
                        query_relevant += 1
            
            precision = query_relevant / max(len(results), 1)
            relevant_count += precision
            
            details.append({
                'query': query,
                'precision@3': precision,
                'results_count': len(results)
            })
        
        avg_precision = relevant_count / max(total_queries, 1)
        
        return EvalResult(
            metric='retrieval_precision',
            score=avg_precision,
            details={
                'total_queries': total_queries,
                'avg_precision@3': avg_precision,
                'query_details': details
            }
        )
    
    # ========== 3. 上下文增强评估 ==========
    
    def evaluate_context_enhancement(
        self,
        test_scenarios: List[Dict]
    ) -> EvalResult:
        """
        评估上下文增强是否有用
        
        核心问题：加了 Mimir 上下文后，AI 回答是否更好？
        
        test_scenarios: [
            {
                'user_input': '继续那个项目',
                'context_snippets': ['用户系统重构项目，基于微服务架构'],
                'platform': 'claude',
                'expected_improvement': '能具体提到用户系统和微服务'
            }
        ]
        """
        total = len(test_scenarios)
        improved = 0
        details = []
        
        for scenario in test_scenarios:
            user_input = scenario['user_input']
            context = scenario.get('context_snippets', [])
            expected = scenario.get('expected_improvement', '')
            
            # 构造有无上下文的两个 prompt
            prompt_without = user_input
            prompt_with = f"上下文：{context}\n\n用户：{user_input}\n请基于以上上下文回答。"
            
            # 让 LLM 评估哪个回答更好
            eval_prompt = f"""比较以下两个 AI 回答，判断哪个更有帮助：

场景：{user_input}

回答A（无上下文）：基于 "{prompt_without}"
回答B（有上下文）：基于 "{prompt_with}"

期望改进：{expected}

哪个回答更好？输出 JSON：
{{
  "better": "A|B",
  "reason": "...",
  "improvement_score": 0.8  // 0-1，B 比 A 好多少
}}"""
            
            try:
                response = self.llm.invoke_mistral(eval_prompt, max_tokens=300)
                result = json.loads(response)
                score = result.get('improvement_score', 0)
                
                if result.get('better') == 'B' and score > 0.5:
                    improved += 1
                
                details.append({
                    'scenario': user_input,
                    'better': result.get('better'),
                    'score': score
                })
            except:
                details.append({
                    'scenario': user_input,
                    'error': 'eval failed'
                })
        
        improvement_rate = improved / max(total, 1)
        
        return EvalResult(
            metric='context_relevance',
            score=improvement_rate,
            details={
                'total_scenarios': total,
                'improved_count': improved,
                'improvement_rate': improvement_rate,
                'details': details
            }
        )
    
    # ========== 4. 端到端场景测试 ==========
    
    def run_end_to_end_test(self) -> List[EvalResult]:
        """
        端到端场景测试
        
        模拟真实使用场景，测试完整流程
        """
        results = []
        
        # 场景1：编码助手
        print("\n🧪 测试场景1：Claude 编码助手")
        coding_test = [
            {
                'query': '用户登录功能怎么实现？',
                'setup_memories': [
                    '项目使用微服务架构',
                    '用户服务基于 JWT 认证',
                    '数据库使用 PostgreSQL'
                ],
                'expected_keywords': ['JWT', '微服务', 'PostgreSQL']
            }
        ]
        # 先写入记忆
        for mem in coding_test[0]['setup_memories']:
            self.memory.add_content(mem, content_type='text', user_id='e2e_test')
        
        result1 = self.evaluate_retrieval([
            {
                'query': coding_test[0]['query'],
                'relevant_keywords': coding_test[0]['expected_keywords']
            }
        ])
        results.append(result1)
        print(f"  检索精准度: {result1.score:.2%}")
        
        # 场景2：设计助手
        print("\n🧪 测试场景2：Midjourney 风格记忆")
        design_test = {
            'setup_memories': [
                '用户喜欢赛博朋克风格，蓝紫色调',
                '偏好高对比度，霓虹灯效果',
                '不喜欢过于复杂的背景'
            ],
            'query': '生成一张未来城市图片',
            'expected_keywords': ['赛博朋克', '蓝紫', '霓虹']
        }
        
        for mem in design_test['setup_memories']:
            self.memory.add_content(mem, content_type='text', user_id='e2e_test')
        
        result2 = self.evaluate_retrieval([
            {
                'query': design_test['query'],
                'relevant_keywords': design_test['expected_keywords']
            }
        ])
        results.append(result2)
        print(f"  检索精准度: {result2.score:.2%}")
        
        # 场景3：邮件助手
        print("\n🧪 测试场景3：邮件上下文")
        email_test = {
            'setup_memories': [
                '上周与投资人会议，讨论了估值问题',
                '投资人希望看到更多用户增长数据',
                '需要在本周五前发送更新邮件'
            ],
            'query': '给投资人写邮件',
            'expected_keywords': ['估值', '增长数据', '周五']
        }
        
        for mem in email_test['setup_memories']:
            self.memory.add_content(mem, content_type='text', user_id='e2e_test')
        
        result3 = self.evaluate_retrieval([
            {
                'query': email_test['query'],
                'relevant_keywords': email_test['expected_keywords']
            }
        ])
        results.append(result3)
        print(f"  检索精准度: {result3.score:.2%}")
        
        return results
    
    def generate_report(self, results: List[EvalResult]) -> str:
        """生成评估报告"""
        report = []
        report.append("=" * 60)
        report.append("Mimir 评估报告")
        report.append("=" * 60)
        
        for result in results:
            report.append(f"\n📊 {result.metric}: {result.score:.2%}")
            for key, value in result.details.items():
                if key != 'details':
                    report.append(f"   {key}: {value}")
        
        # 综合评价
        avg_score = sum(r.score for r in results) / len(results)
        report.append(f"\n{'=' * 60}")
        report.append(f"综合得分: {avg_score:.2%}")
        
        if avg_score >= 0.8:
            report.append("评级: 🟢 优秀")
        elif avg_score >= 0.6:
            report.append("评级: 🟡 良好")
        else:
            report.append("评级: 🔴 需改进")
        
        report.append("=" * 60)
        
        return "\n".join(report)


# 使用示例
if __name__ == "__main__":
    from mimir_native import MimirMemory
    from mimir_native.llm_client import BedrockClient
    
    # 初始化
    mimir = MimirMemory(db_path=':memory:')
    llm = BedrockClient()
    evaluator = MimirEvaluator(mimir, llm)
    
    print("🚀 开始 Mimir 评估\n")
    
    # 运行端到端测试
    results = evaluator.run_end_to_end_test()
    
    # 生成报告
    report = evaluator.generate_report(results)
    print("\n" + report)
