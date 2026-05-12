#!/usr/bin/env bash
# 自动格式化代码

set -e

echo "=== Running Black (formatting) ==="
uv run black backend/ main.py

echo ""
echo "=== Running Ruff (auto-fix) ==="
uv run ruff check --fix backend/ main.py

echo ""
echo "=== Formatting complete! ==="