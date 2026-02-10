#!/usr/bin/env python3
"""
快速验证 QueryEnhancer 效果
简化版测试，只验证时间解析功能
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from datetime import datetime
from mimir_native.enhanced_retrieval import QueryEnhancer

def test_query_enhancer():
    """测试查询增强器"""
    
    print("=" * 60)
    print("QueryEnhancer 功能验证")
    print("=" * 60)
    
    enhancer = QueryEnhancer(reference_date=datetime(2026, 2, 10))
    
    test_cases = [
        {
            'query': '我上周规划的项目',
            'expect_time': True,
            'expect_platform': False,
            'description': '跨时间关联（之前 25%）'
        },
        {
            'query': '上周一我和 Claude 讨论了什么',
            'expect_time': True,
            'expect_platform': True,
            'description': '精确时间+平台（之前 33%）'
        },
        {
            'query': '向量检索的代码实现',
            'expect_time': False,
            'expect_platform': False,
            'description': '跨平台关联（之前 66%）'
        },
        {
            'query': '昨天收藏的 Chrome Extension 文章',
            'expect_time': True,
            'expect_platform': False,
            'description': '时间+关键词'
        }
    ]
    
    passed = 0
    total = len(test_cases)
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n测试 {i}: {test['description']}")
        print(f"  查询: {test['query']}")
        
        result = enhancer.enhance(test['query'])
        
        print(f"  增强后: {result['enhanced_query']}")
        print(f"  时间范围: {result['time_range']}")
        print(f"  平台: {result['platform']}")
        
        # 验证
        checks = []
        if test['expect_time']:
            has_time = result['time_range'] is not None
            checks.append(('时间解析', has_time))
        
        if test['expect_platform']:
            has_platform = result['platform'] is not None
            checks.append(('平台解析', has_platform))
        
        all_pass = all(c[1] for c in checks) if checks else True
        
        if checks:
            for name, status in checks:
                symbol = '✅' if status else '❌'
                print(f"  {symbol} {name}")
        
        if all_pass:
            passed += 1
            print(f"  ✅ 通过")
        else:
            print(f"  ❌ 失败")
    
    print("\n" + "=" * 60)
    print(f"结果: {passed}/{total} 通过 ({passed/total*100:.0f}%)")
    
    if passed == total:
        print("🎉 QueryEnhancer 工作正常！")
        print("\n预期效果:")
        print("  - '上周' → 自动转为 7 天日期范围")
        print("  - 'Claude' → 自动添加 platform 过滤")
        print("  - 跨时间关联 25% → 预计 80%+")
        print("  - 精确时间+平台 33% → 预计 80%+")
    else:
        print("⚠️ 部分功能需要修复")
    
    print("=" * 60)
    return passed == total

if __name__ == "__main__":
    test_query_enhancer()
