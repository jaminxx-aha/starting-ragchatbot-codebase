#!/usr/bin/env bash
# 快速检查脚本（仅格式检查和 linting，不运行测试）

set -e

echo "=== Running Black (format check) ==="
uv run black --check backend/ main.py

echo ""
echo "=== Running Ruff (linting) ==="
uv run ruff check backend/ main.py

echo ""
echo "=== Quick checks passed! ==="