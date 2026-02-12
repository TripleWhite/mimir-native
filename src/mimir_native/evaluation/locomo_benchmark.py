"""LoCoMo Benchmark - Full evaluation on LoCoMo dataset"""

import json
import re
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
from pathlib import Path
import numpy as np


class LoCoMoBenchmark:
    """
    Full LoCoMo benchmark evaluation.
    
    Evaluates memory system on all 10 conversations and all question types.
    """
    
    def __init__(self, data_path: str):
        """
        Initialize benchmark.
        
        Args:
            data_path: Path to locomodata.json
        """
        self.data_path = Path(data_path)
        self.data = self._load_data()
        self.results = []
        
    def _load_data(self) -> List[Dict]:
        """Load LoCoMo data"""
        with open(self.data_path, 'r') as f:
            return json.load(f)
    
    def calculate_f1(self, predicted: str, ground_truth: Any) -> float:
        """
        Calculate F1 score for answer matching.
        
        Handles multiple formats and temporal expressions.
        """
        pred_str = str(predicted).lower().strip()
        gt_str = str(ground_truth).lower().strip()
        
        # Exact match
        if pred_str == gt_str:
            return 1.0
        
        # Extract dates and compare
        pred_dates = self._extract_dates(pred_str)
        gt_dates = self._extract_dates(gt_str)
        
        if pred_dates and gt_dates:
            # Check date overlap
            common = len(set(pred_dates) & set(gt_dates))
            if common > 0:
                return 0.8 + (0.2 * common / max(len(pred_dates), len(gt_dates)))
        
        # Token overlap
        pred_tokens = set(pred_str.split())
        gt_tokens = set(gt_str.split())
        
        if not pred_tokens or not gt_tokens:
            return 0.0
        
        common = len(pred_tokens & gt_tokens)
        precision = common / len(pred_tokens) if pred_tokens else 0
        recall = common / len(gt_tokens) if gt_tokens else 0
        
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * precision * recall / (precision + recall)
        return f1
    
    def _extract_dates(self, text: str) -> set:
        """Extract dates from text"""
        dates = set()
        
        # Patterns
        patterns = [
            r'(\d{1,2})\s+(january|february|march|april|may|june|july|august|september|october|november|december)\s+(20\d{2})',
            r'(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{1,2})[,
\s]+(20\d{2})',
            r'(20\d{2})'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    dates.append(' '.join(match))
                else:
                    dates.append(match)
        
        return set(dates)
    
    def evaluate_conversation(
        self,
        conversation_idx: int,
        answer_fn: callable
    ) -> Dict[str, Any]:
        """
        Evaluate on a single conversation.
        
        Args:
            conversation_idx: Index of conversation (0-9)
            answer_fn: Function that takes (question, context) -> answer
            
        Returns:
            Evaluation results
        """
        conv = self.data[conversation_idx]
        qa_list = conv.get('qa', [])
        
        results = {
            'conversation_idx': conversation_idx,
            'name': conv.get('name', f'D{conversation_idx+1}'),
            'total_qa': len(qa_list),
            'results': [],
            'by_type': {}
        }
        
        for i, qa in enumerate(qa_list):
            question = qa.get('question', '')
            ground_truth = qa.get('answer', '')
            
            # Get answer from system
            predicted = answer_fn(question, conv)
            
            # Calculate F1
            f1 = self.calculate_f1(predicted, ground_truth)
            
            # Detect question type
            q_type = self._detect_question_type(question)
            
            result = {
                'idx': i,
                'question': question,
                'predicted': predicted,
                'ground_truth': ground_truth,
                'f1': f1,
                'type': q_type
            }
            
            results['results'].append(result)
            
            # Group by type
            if q_type not in results['by_type']:
                results['by_type'][q_type] = {'count': 0, 'total_f1': 0}
            results['by_type'][q_type]['count'] += 1
            results['by_type'][q_type]['total_f1'] += f1
        
        # Calculate averages
        if results['results']:
            results['avg_f1'] = sum(r['f1'] for r in results['results']) / len(results['results'])
        
        for q_type, data in results['by_type'].items():
            data['avg_f1'] = data['total_f1'] / data['count'] if data['count'] > 0 else 0
        
        return results
    
    def _detect_question_type(self, question: str) -> str:
        """Detect question type"""
        q_lower = question.lower().strip()
        
        if q_lower.startswith('when'):
            return 'when'
        elif q_lower.startswith('what'):
            return 'what'
        elif q_lower.startswith('who'):
            return 'who'
        elif q_lower.startswith('how'):
            return 'how'
        elif q_lower.startswith('would'):
            return 'would'
        elif q_lower.startswith('where'):
            return 'where'
        else:
            return 'other'
    
    def run_full_evaluation(
        self,
        answer_fn: callable,
        conversations: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Run full evaluation on all or specified conversations.
        
        Args:
            answer_fn: Function that answers questions
            conversations: Optional list of conversation indices
            
        Returns:
            Full evaluation report
        """
        if conversations is None:
            conversations = range(len(self.data))
        
        all_results = []
        
        for idx in conversations:
            print(f"Evaluating conversation {idx+1}/{len(self.data)}...")
            conv_result = self.evaluate_conversation(idx, answer_fn)
            all_results.append(conv_result)
        
        # Aggregate results
        report = {
            'timestamp': datetime.now().isoformat(),
            'conversations_evaluated': len(all_results),
            'per_conversation': all_results,
            'summary': self._summarize_results(all_results)
        }
        
        return report
    
    def _summarize_results(self, results: List[Dict]) -> Dict[str, Any]:
        """Summarize all results"""
        all_f1s = []
        by_type = {}
        
        for conv in results:
            for r in conv['results']:
                all_f1s.append(r['f1'])
                
                q_type = r['type']
                if q_type not in by_type:
                    by_type[q_type] = []
                by_type[q_type].append(r['f1'])
        
        summary = {
            'total_questions': len(all_f1s),
            'overall_f1': np.mean(all_f1s) if all_f1s else 0,
            'by_type': {}
        }
        
        for q_type, f1s in by_type.items():
            summary['by_type'][q_type] = {
                'count': len(f1s),
                'avg_f1': np.mean(f1s),
                'median_f1': np.median(f1s)
            }
        
        return summary
    
    def save_report(self, report: Dict[str, Any], output_path: str):
        """Save evaluation report"""
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Report saved to {output_path}")
