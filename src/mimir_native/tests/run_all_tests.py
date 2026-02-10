#!/usr/bin/env python3
"""
Mimir V2 - Test Suite Runner
统一测试运行脚本

运行所有测试并输出汇总结果
"""

import sys
import os
import subprocess
import json
from pathlib import Path
from datetime import datetime

# 测试目录
TESTS_DIR = Path(__file__).parent
BACKEND_DIR = TESTS_DIR.parent.parent

# 需要运行的测试文件列表
TEST_FILES = [
    "test_simple.py",
    "test_temporal_kg.py",
    "test_datetime_bug.py",
    "test_conversation_date_fix.py",
    "test_preprocessors.py",
    "test_database.py",
    "test_memory_agent.py",
    "test_standalone.py",
    "test_hybrid_retriever.py",
]


def print_header(text):
    """打印标题"""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def print_separator():
    """打印分隔线"""
    print("-" * 70)


def run_test_file(test_file):
    """
    运行单个测试文件
    
    Returns:
        dict: 测试结果信息
    """
    test_path = TESTS_DIR / test_file
    
    # 检查文件是否存在
    if not test_path.exists():
        return {
            "file": test_file,
            "status": "NOT_FOUND",
            "passed": 0,
            "failed": 0,
            "errors": 0,
            "duration": 0,
            "message": "文件不存在"
        }
    
    print(f"\n📄 运行测试: {test_file}")
    print_separator()
    
    start_time = datetime.now()
    
    try:
        # 设置环境变量，确保 app 模块可以被找到
        env = os.environ.copy()
        env['PYTHONPATH'] = str(BACKEND_DIR) + ':' + env.get('PYTHONPATH', '')
        
        # 使用 subprocess 运行测试文件
        result = subprocess.run(
            [sys.executable, str(test_file)],
            cwd=str(TESTS_DIR),  # 在测试目录运行
            capture_output=True,
            text=True,
            timeout=120,  # 超时时间 120 秒
            env=env
        )
        
        duration = (datetime.now() - start_time).total_seconds()
        
        # 解析输出
        stdout = result.stdout
        stderr = result.stderr
        
        # 显示输出
        if stdout:
            print(stdout)
        if stderr:
            print("STDERR:", stderr)
        
        # 判断测试结果
        if result.returncode == 0:
            status = "PASSED"
            print(f"✅ {test_file} 通过 ({duration:.2f}s)")
        else:
            status = "FAILED"
            print(f"❌ {test_file} 失败 ({duration:.2f}s)")
        
        # 尝试解析 unittest 输出统计
        passed, failed, errors = parse_test_counts(stdout + stderr)
        
        return {
            "file": test_file,
            "status": status,
            "passed": passed,
            "failed": failed,
            "errors": errors,
            "duration": duration,
            "returncode": result.returncode,
            "message": "完成"
        }
        
    except subprocess.TimeoutExpired:
        duration = (datetime.now() - start_time).total_seconds()
        print(f"⏱️  {test_file} 超时 ({duration:.2f}s)")
        return {
            "file": test_file,
            "status": "TIMEOUT",
            "passed": 0,
            "failed": 0,
            "errors": 0,
            "duration": duration,
            "message": "运行超时"
        }
    except Exception as e:
        duration = (datetime.now() - start_time).total_seconds()
        print(f"💥 {test_file} 异常: {e}")
        return {
            "file": test_file,
            "status": "ERROR",
            "passed": 0,
            "failed": 0,
            "errors": 1,
            "duration": duration,
            "message": str(e)
        }


def parse_test_counts(output):
    """
    从测试输出中解析测试计数
    
    Returns:
        (passed, failed, errors) tuple
    """
    passed = 0
    failed = 0
    errors = 0
    
    # 尝试匹配 "Ran X tests" 格式
    import re
    
    # 匹配 "Ran X tests"
    match = re.search(r'Ran (\d+) tests?', output)
    if match:
        total = int(match.group(1))
        
        # 查找失败数
        fail_match = re.search(r'failures=(\d+)', output)
        if fail_match:
            failed = int(fail_match.group(1))
        
        # 查找错误数
        error_match = re.search(r'errors=(\d+)', output)
        if error_match:
            errors = int(error_match.group(1))
        
        # 通过数 = 总数 - 失败数 - 错误数
        passed = total - failed - errors
    else:
        # 尝试匹配 "OK" 或 "FAILED"
        if "OK" in output:
            # 如果输出包含 OK，尝试找到测试数量
            match = re.search(r'(\d+) passed', output)
            if match:
                passed = int(match.group(1))
            else:
                passed = 1  # 至少有一个测试通过
        
        if "FAILED" in output or "FAIL:" in output:
            match = re.search(r'(\d+) failed', output)
            if match:
                failed = int(match.group(1))
    
    return passed, failed, errors


def print_summary(results):
    """打印测试汇总结果"""
    print_header("测试汇总结果")
    
    total_tests = len(results)
    passed_tests = sum(1 for r in results if r["status"] == "PASSED")
    failed_tests = sum(1 for r in results if r["status"] == "FAILED")
    error_tests = sum(1 for r in results if r["status"] in ["ERROR", "TIMEOUT", "NOT_FOUND"])
    
    total_passed = sum(r["passed"] for r in results)
    total_failed = sum(r["failed"] for r in results)
    total_errors = sum(r["errors"] for r in results)
    total_duration = sum(r["duration"] for r in results)
    
    print(f"\n📊 测试文件统计:")
    print(f"   总计: {total_tests} 个测试文件")
    print(f"   ✅ 通过: {passed_tests} 个")
    print(f"   ❌ 失败: {failed_tests} 个")
    print(f"   💥 错误/超时/未找到: {error_tests} 个")
    
    print(f"\n📈 测试用例统计:")
    print(f"   通过: {total_passed} 个")
    print(f"   失败: {total_failed} 个")
    print(f"   错误: {total_errors} 个")
    print(f"   总计: {total_passed + total_failed + total_errors} 个")
    
    print(f"\n⏱️  总耗时: {total_duration:.2f} 秒")
    
    print("\n📋 详细结果:")
    print_separator()
    
    for result in results:
        status_icon = {
            "PASSED": "✅",
            "FAILED": "❌",
            "ERROR": "💥",
            "TIMEOUT": "⏱️",
            "NOT_FOUND": "❓"
        }.get(result["status"], "❓")
        
        print(f"{status_icon} {result['file']}: {result['status']}")
        print(f"   通过: {result['passed']}, 失败: {result['failed']}, 错误: {result['errors']}")
        print(f"   耗时: {result['duration']:.2f}s")
        if result.get("message") and result["message"] != "完成":
            print(f"   消息: {result['message']}")
    
    print_separator()
    
    # 最终结论
    if failed_tests == 0 and error_tests == 0:
        print("\n🎉 所有测试通过！")
        return True
    else:
        print(f"\n⚠️  有 {failed_tests + error_tests} 个测试文件未通过")
        return False


def save_results(results, success):
    """保存测试结果到 JSON 文件"""
    results_dir = Path(__file__).parent.parent / "subagent_workflow" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = results_dir / "task_test_fix.json"
    
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "success": success,
        "summary": {
            "total_files": len(results),
            "passed_files": sum(1 for r in results if r["status"] == "PASSED"),
            "failed_files": sum(1 for r in results if r["status"] == "FAILED"),
            "error_files": sum(1 for r in results if r["status"] in ["ERROR", "TIMEOUT", "NOT_FOUND"]),
            "total_passed": sum(r["passed"] for r in results),
            "total_failed": sum(r["failed"] for r in results),
            "total_errors": sum(r["errors"] for r in results),
            "total_duration": sum(r["duration"] for r in results)
        },
        "results": results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 测试结果已保存到: {output_file}")


def main():
    """主函数"""
    print_header("Mimir V2 Test Suite Runner")
    print(f"测试目录: {TESTS_DIR}")
    print(f"Python: {sys.executable}")
    print(f"工作目录: {BACKEND_DIR}")
    
    # 运行所有测试
    results = []
    for test_file in TEST_FILES:
        result = run_test_file(test_file)
        results.append(result)
    
    # 打印汇总
    success = print_summary(results)
    
    # 保存结果
    save_results(results, success)
    
    # 返回退出码
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
