#!/usr/bin/env python3
"""
测试 LoCoMo 日期解析
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from mimir_native.preprocessors.base import parse_date

def test_parse_date():
    print("🧪 Testing parse_date for LoCoMo")
    
    test_cases = [
        "7 May 2023",
        "8 May, 2023",
        "1:56 pm on 8 May, 2023",
        "25 May 2023",
        "9 June 2023",
        "June 2023",
        "2022",
    ]
    
    for tc in test_cases:
        result = parse_date(tc)
        print(f"  '{tc}' -> {result}")

if __name__ == "__main__":
    test_parse_date()
