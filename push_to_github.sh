#!/bin/bash
# LoCoMo 优化代码推送脚本
# 运行: bash push_to_github.sh

echo "=================================="
echo "LoCoMo 优化代码推送"
echo "=================================="
echo ""

# 检查 git
cd /tmp/mimir-review/mimir-native

echo "当前状态:"
git log --oneline -3
echo ""
echo "待推送文件:"
git status --short
echo ""

# 方案 1: 使用 GitHub Personal Access Token
echo "方案 1: 使用 GitHub Token 推送"
echo "--------------------------------"
echo "1. 访问 https://github.com/settings/tokens"
echo "2. 生成 Fine-grained personal access token"
echo "3. 运行: git remote set-url origin https://TOKEN@github.com/TripleWhite/mimir-native.git"
echo "4. 运行: git push origin main"
echo ""

# 方案 2: 使用 SSH
echo "方案 2: 使用 SSH 推送"
echo "----------------------"
echo "1. 确保已配置 SSH key: cat ~/.ssh/id_rsa.pub"
echo "2. 添加到 GitHub: https://github.com/settings/keys"
echo "3. 运行: git remote set-url origin git@github.com:TripleWhite/mimir-native.git"
echo "4. 运行: git push origin main"
echo ""

# 方案 3: 直接下载并手动上传
echo "方案 3: 手动上传"
echo "----------------"
echo "关键文件位置:"
echo "  - /tmp/mimir-review/mimir-native/test_evidence_retriever.py"
echo "  - /tmp/mimir-review/mimir-native/LOCOMO_OPTIMIZATION_REPORT.md"
echo "  - /tmp/mimir-review/mimir-native/temporal_normalizer.py"
echo ""
echo "通过 GitHub Web 界面上传这些文件"
echo ""

echo "=================================="
echo "提交信息:"
echo "  feat: LoCoMo When question optimization - F1 25.3% → 86.1%"
echo ""
echo "包含能力:"
echo "  - Evidence-Based Retriever V2"
echo "  - Relative Time Calculator"
echo "  - Historical Event Handler"
echo "  - Multi-Evidence Fusion (RRF)"
echo "=================================="
