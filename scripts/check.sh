#!/usr/bin/env bash
# 代码质量检查脚本

set -e

echo "=== Running Black (format check) ==="
uv run black --check backend/ main.py

echo ""
echo "=== Running Ruff (linting) ==="
uv run ruff check backend/ main.py

echo ""
echo "=== Running MyPy (type checking) ==="
uv run mypy backend/ main.py

echo ""
echo "=== Running Tests ==="
uv run pytest backend/tests/ -v

echo ""
echo "=== All quality checks passed! ==="