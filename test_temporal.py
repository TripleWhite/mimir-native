#!/usr/bin/env python3
"""
测试时间解析器
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from datetime import datetime
from mimir_native.temporal_resolver import TemporalResolver

def test_resolver():
    print("🧪 Testing Temporal Resolver")
    
    resolver = TemporalResolver()
    
    # LoCoMo 场景：对话发生在 8 May 2023
    ref_date = datetime(2023, 5, 8)
    
    test_cases = [
        ("yesterday", "7 May 2023"),
        ("last year", "2022"),
        ("last Saturday", "6 May 2023"),
        ("Caroline visited the group yesterday", "7 May 2023"),
        ("Melanie painted a sunrise last year", "2022"),
    ]
    
    print(f"\nReference date: {ref_date.strftime('%Y-%m-%d (%A)')}")
    print("-" * 60)
    
    for expr, expected in test_cases:
        result = resolver.resolve_relative_time(expr, ref_date)
        # 也测试 extract_and_resolve
        extracted = resolver.extract_and_resolve(expr, ref_date)
        status = "✅" if result and expected and expected in result else "❌"
        ext_status = "✅" if extracted and any(expected in v for v in extracted.values()) else "❌"
        print(f"{status} resolve: '{expr}' -> '{result}'")
        print(f"{ext_status} extract: '{expr}' -> {extracted}")
        print()

if __name__ == "__main__":
    test_resolver()
