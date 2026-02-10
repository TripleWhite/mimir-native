# LoCoMo F1 优化分析 (13.33% → 20%)

## 当前状态
- **F1**: 13.33%
- **EM**: 0%
- **目标 F1**: 20%

## 失败案例分析

### Q1: LGBTQ support group 时间
- **预测**: "07 May 2023"
- **标准**: "7 May 2023"
- **问题**: 前导零导致 F1=0.14
- **根因**: 日期格式未标准化

### Q2: Melanie paint sunrise 时间
- **预测**: "2022" (字符串)
- **标准**: 2022 (数字/字符串均可)
- **问题**: "painted sunrise" vs "paint a sunrise", F1=0
- **根因**: 语义匹配不足

### Q3: Caroline 教育方向
- **预测**: "Counseling or mental health education"
- **标准**: "Psychology, counseling certification"
- **问题**: F1=0.25
- **根因**: 语义相似但词汇不匹配

### Q4: Caroline researched
- **预测**: "Caroline researched adoption agencies"
- **标准**: "Adoption agencies"
- **问题**: F1=0.33 (额外文字)
- **根因**: 答案生成包含多余信息

### Q5: Caroline's identity
- **预测**: "LGBTQ person"
- **标准**: "Transgender woman"
- **问题**: F1=0
- **根因**: 事实提取不够具体！原文有 "transgender woman" 和 "trans stories"

### Q6: Charity race 时间
- **预测**: "20 May 2023"
- **标准**: "The sunday before 25 May 2023"
- **问题**: F1=0.13 (日期差一天，且格式不同)
- **根因**: "last Saturday" 解析错误

### Q7: Camping 时间
- **预测**: "June 20, 2023"
- **标准**: "June 2023"
- **问题**: F1=0.22 (过于具体)
- **根因**: 提取过于具体

### Q8: Relationship status
- **预测**: "LGBTQ person. (Relationship status is not provided)"
- **标准**: "Single"
- **问题**: F1=0
- **根因**: 未提取到 "single" 信息

### Q9/Q10: Speech/meetup 时间
- **预测**: "02 June 2023"
- **标准**: "The week before 9 June 2023"
- **问题**: F1=0.13
- **根因**: "last week" 解析正确但格式不匹配，且需要处理复杂时间表达式

## 根因总结

1. **日期格式不一致**: "07 May" vs "7 May"
2. **事实提取不够具体**: "LGBTQ person" vs "Transgender woman"
3. **复杂时间表达式**: "the week before 9 June 2023"
4. **关键信息遗漏**: "single" 状态未提取
5. **答案格式问题**: 额外文字降低 F1

## 修复策略

### 阶段 1: 快速修复 (目标: 16-18%)
1. **日期格式标准化**: 去除前导零，统一格式
2. **Prompt 优化**: 提取更具体的事实，保留原文表述
3. **答案清理**: 去除多余文字，只保留核心答案

### 阶段 2: 深度优化 (目标: 20%+)
1. **复杂时间解析器**: 处理 "the week before X" 等表达式
2. **事实完整性检查**: 确保关键信息不遗漏
3. **语义匹配优化**: 提升相似语义的 F1 分数
