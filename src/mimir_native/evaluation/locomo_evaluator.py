"""
Mimir Memory V2 - LoCoMo Benchmark Evaluator

LoCoMo Benchmark 评估器
测试 Mimir 在 LoCoMo 数据集上的表现
"""

import json
import logging
from typing import List, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class LoCoMoEvaluator:
    """
    LoCoMo Benchmark 评估器
    
    测试 Mimir 在 LoCoMo 数据集上的表现
    """
    
    def __init__(self, mimir_memory, llm_client):
        """
        初始化 LoCoMo 评估器
        
        Args:
            mimir_memory: MimirMemory 实例
            llm_client: LLM 客户端实例（用于答案生成）
        """
        self.mimir = mimir_memory
        self.llm = llm_client
        self.results = []
        self.max_questions = None  # 可设置限制评估问题数量
    
    def load_locomo_data(self, data_path: str) -> List[Dict]:
        """
        加载 LoCoMo 数据集
        
        关键：正确处理 session_X_date_time 字段
        
        Args:
            data_path: LoCoMo JSON 文件路径
            
        Returns:
            List[Dict]: LoCoMo 数据列表
        """
        logger.info(f"Loading LoCoMo data from {data_path}")
        with open(data_path, 'r') as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} conversations")
        return data
    
    def ingest_conversation(self, conversation: Dict, user_id: str = 'locomo_test'):
        """
        将 LoCoMo 对话摄入 Mimir

        优化：批量处理整个 session 的消息，减少 API 调用次数

        关键步骤：
        1. 收集所有消息
        2. 批量预处理
        3. 批量存储原始内容
        4. 批量提取事实和 embedding
        5. 批量存储记忆
        6. 构建时序图谱

        Args:
            conversation: LoCoMo 对话字典
            user_id: 用户 ID（默认为 'locomo_test'）
        """
        # 第一阶段：收集所有需要处理的消息
        all_messages = []
        message_metadata = []

        for session_key, messages in conversation.items():
            # 只处理 session_X 键，跳过 _date_time 和其他键
            if not (session_key.startswith('session_') and '_date_time' not in session_key):
                continue
            
            # 跳过非列表类型的值（如日期字符串）
            if not isinstance(messages, list):
                continue

            # 获取 session 日期（关键！）
            date_key = f"{session_key}_date_time"
            session_date = conversation.get(date_key)

            if not session_date:
                logger.warning(f"No date found for {session_key}")
                continue

            for msg in messages:
                all_messages.append({
                    'messages': [msg],
                    'session_date': session_date
                })
                message_metadata.append({
                    'source': 'locomo',
                    'session': session_key
                })

        if not all_messages:
            return

        logger.debug(f"批量处理 {len(all_messages)} 条消息")

        # 第二阶段：批量预处理
        preproc_contents = []
        for msg_data, metadata in zip(all_messages, message_metadata):
            preproc_content = self.mimir.preprocessor.process(
                content=msg_data,
                content_type='conversation',
                metadata=metadata
            )
            preproc_contents.append(preproc_content)

        # 第三阶段：批量存储原始内容并收集 raw_content 对象
        raw_contents = []
        from ..database import RawContentCreate

        for preproc_content, metadata in zip(preproc_contents, message_metadata):
            raw_content_create = RawContentCreate(
                user_id=user_id,
                content_type='conversation',
                raw_text=preproc_content.text,
                metadata=json.dumps(preproc_content.metadata) if preproc_content.metadata else None,
                occurred_at=preproc_content.occurred_at
            )

            raw_content_id = self.mimir.db.create_raw_content(raw_content_create)
            raw_content = self.mimir.db.get_raw_content(raw_content_id)

            if raw_content:
                raw_contents.append(raw_content)
            else:
                logger.warning(f"Failed to retrieve raw_content for id: {raw_content_id}")

        # 第四阶段：批量处理 raw_contents 提取事实
        all_memories = []
        for raw_content in raw_contents:
            try:
                memories = self.mimir.memory_agent.process_raw_content(raw_content)
                all_memories.extend(memories)
            except Exception as e:
                logger.error(f"处理 raw_content {raw_content.id} 失败: {e}")

        logger.debug(f"从 {len(raw_contents)} 条消息中提取了 {len(all_memories)} 条记忆")

        # 第五阶段：批量更新图谱
        for memory in all_memories:
            try:
                entities, relations = self.mimir.kg.extract_entities_and_relations(
                    memory.content, self.llm
                )
                self.mimir.kg.add_fact(memory, entities, relations)
            except Exception as e:
                logger.warning(f"更新知识图谱失败 for memory {memory.id}: {e}")
    
    def evaluate_question(self, qa: Dict, user_id: str = 'locomo_test') -> Dict:
        """
        评估单个问题
        
        Args:
            qa: {'question': '...', 'answer': '...', 'category': 1/2/3}
            user_id: 用户 ID
        
        Returns:
            {
                'question': '...',
                'ground_truth': '...',
                'predicted': '...',
                'f1': 0.85,
                'em': True/False,
                'category': 1/2/3
            }
        """
        question = qa.get('question', '')
        # 优先使用 answer，否则使用 adversarial_answer
        ground_truth = str(qa.get('answer') or qa.get('adversarial_answer', ''))
        category = qa.get('category', 1)
        
        # 使用 HybridRetriever 检索相关记忆
        try:
            results = self.mimir.retriever.retrieve(
                query=question,
                user_id=user_id,
                top_k=10
            )
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            results = []
        
        # 构建上下文
        context = self._build_context(results)
        
        # 使用 LLM 生成答案
        predicted = self._generate_answer(question, context)
        
        # 计算 F1 和 EM
        f1 = self._calculate_f1(predicted, ground_truth)
        em = self._calculate_em(predicted, ground_truth)
        
        return {
            'question': question,
            'ground_truth': ground_truth,
            'predicted': predicted,
            'f1': f1,
            'em': em,
            'category': category,
            'retrieved_memories': len(results)
        }
    
    def _build_context(self, results: List[Any]) -> str:
        """从检索结果构建上下文"""
        contexts = []
        for r in results:
            memory = r.memory if hasattr(r, 'memory') else r
            content = memory.content if hasattr(memory, 'content') else str(memory)
            contexts.append(f"- {content}")
        return "\n".join(contexts)
    
    def _generate_answer(self, question: str, context: str) -> str:
        """使用 LLM 生成答案"""
        prompt = f"""基于以下记忆，回答问题。只回答事实，不要解释。

记忆:
{context}

问题: {question}

答案（简短）:
"""
        
        try:
            response = self.llm.invoke_mistral(prompt, max_tokens=100, temperature=0.0)
            return response.strip()
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            return ""
    
    def _calculate_f1(self, prediction: str, ground_truth: str) -> float:
        """
        计算 F1 分数
        
        LoCoMo 标准 F1 计算：
        - token-based F1
        - 转换为小写后分词
        """
        pred_tokens = set(prediction.lower().split())
        truth_tokens = set(ground_truth.lower().split())
        
        if not pred_tokens or not truth_tokens:
            return 0.0
        
        common = pred_tokens & truth_tokens
        if not common:
            return 0.0
        
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(truth_tokens)
        
        if precision + recall == 0:
            return 0.0
            
        return 2 * (precision * recall) / (precision + recall)
    
    def _calculate_em(self, prediction: str, ground_truth: str) -> bool:
        """
        计算 Exact Match
        
        LoCoMo 标准 EM：
        - 不区分大小写
        - 去除首尾空格后比较
        """
        return prediction.lower().strip() == ground_truth.lower().strip()
    
    def run_full_evaluation(self, data_path: str, max_questions: int = None) -> Dict:
        """
        运行完整的 LoCoMo 评估
        
        Args:
            data_path: LoCoMo 数据文件路径
            max_questions: 可选，限制评估的问题数量
        
        Returns:
            {
                'overall': {'f1': 0.45, 'em': 0.12, 'total': 199},
                'by_category': {
                    'Cat-1': {'f1': 0.52, 'em': 0.15, 'count': 87},
                    'Cat-2': {'f1': 0.38, 'em': 0.08, 'count': 72},
                    'Cat-3': {'f1': 0.41, 'em': 0.11, 'count': 40}
                },
                'detailed_results': [...]
            }
        """
        self.max_questions = max_questions
        
        # 加载数据
        data = self.load_locomo_data(data_path)
        logger.info(f"Starting evaluation on {len(data)} conversations")
        
        # 摄入对话
        logger.info("Ingesting conversations into Mimir...")
        for i, conv in enumerate(data):
            if 'conversation' in conv:
                self.ingest_conversation(conv['conversation'])
            else:
                # 有些格式直接是 conversation 字典
                self.ingest_conversation(conv)
            
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{len(data)} conversations")
        
        # 评估每个问题
        logger.info("Evaluating questions...")
        all_results = []
        question_count = 0
        
        for conv in data:
            qa_list = conv.get('qa', [])
            for qa in qa_list:
                if max_questions and question_count >= max_questions:
                    logger.info(f"Reached max_questions limit: {max_questions}")
                    break
                
                result = self.evaluate_question(qa)
                all_results.append(result)
                question_count += 1
                
                if question_count % 50 == 0:
                    logger.info(f"Evaluated {question_count} questions...")
            
            if max_questions and question_count >= max_questions:
                break
        
        logger.info(f"Evaluation complete: {len(all_results)} questions")
        
        # 计算总体指标
        total = len(all_results)
        if total == 0:
            return {
                'overall': {'f1': 0.0, 'em': 0.0, 'total': 0},
                'by_category': {},
                'detailed_results': []
            }
        
        overall_f1 = sum(r['f1'] for r in all_results) / total
        overall_em = sum(1 for r in all_results if r['em']) / total
        
        # 按类别统计
        by_category = {}
        for r in all_results:
            cat = f"Cat-{r['category']}"
            if cat not in by_category:
                by_category[cat] = {'f1': 0, 'em': 0, 'count': 0}
            by_category[cat]['f1'] += r['f1']
            by_category[cat]['em'] += 1 if r['em'] else 0
            by_category[cat]['count'] += 1
        
        for cat in by_category:
            by_category[cat]['f1'] /= by_category[cat]['count']
            by_category[cat]['em'] /= by_category[cat]['count']
        
        return {
            'overall': {
                'f1': overall_f1,
                'em': overall_em,
                'total': total
            },
            'by_category': by_category,
            'detailed_results': all_results
        }
    
    def run_incremental_evaluation(self, data_path: str, 
                                   start_idx: int = 0, 
                                   end_idx: int = None) -> Dict:
        """
        增量评估 - 评估指定范围的对话
        
        Args:
            data_path: LoCoMo 数据文件路径
            start_idx: 起始对话索引
            end_idx: 结束对话索引（None 表示到结尾）
        
        Returns:
            评估结果字典
        """
        # 加载数据
        data = self.load_locomo_data(data_path)
        end_idx = end_idx or len(data)
        
        subset = data[start_idx:end_idx]
        logger.info(f"Running incremental evaluation on conversations {start_idx}-{end_idx}")
        
        # 摄入对话
        for conv in subset:
            if 'conversation' in conv:
                self.ingest_conversation(conv['conversation'])
            else:
                self.ingest_conversation(conv)
        
        # 评估问题
        all_results = []
        for conv in subset:
            for qa in conv.get('qa', []):
                result = self.evaluate_question(qa)
                all_results.append(result)
        
        # 计算指标
        total = len(all_results)
        if total == 0:
            return {'overall': {'f1': 0, 'em': 0, 'total': 0}, 'by_category': {}}
        
        overall_f1 = sum(r['f1'] for r in all_results) / total
        overall_em = sum(1 for r in all_results if r['em']) / total
        
        # 按类别统计
        by_category = {}
        for r in all_results:
            cat = f"Cat-{r['category']}"
            if cat not in by_category:
                by_category[cat] = {'f1': 0, 'em': 0, 'count': 0}
            by_category[cat]['f1'] += r['f1']
            by_category[cat]['em'] += 1 if r['em'] else 0
            by_category[cat]['count'] += 1
        
        for cat in by_category:
            by_category[cat]['f1'] /= by_category[cat]['count']
            by_category[cat]['em'] /= by_category[cat]['count']
        
        return {
            'overall': {
                'f1': overall_f1,
                'em': overall_em,
                'total': total
            },
            'by_category': by_category,
            'detailed_results': all_results
        }
