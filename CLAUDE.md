# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 用户偏好

- **所有回复使用中文**
- **永远使用 uv 运行服务器和管理依赖，不直接使用 pip**

## 运行命令

启动应用：
```bash
./run.sh
# 或手动启动
cd backend && uv run uvicorn app:app --reload --port 8000
```

安装依赖：
```bash
uv sync
```

需要 `.env` 文件配置 `ANTHROPIC_API_KEY`。

## 架构概述

这是一个 RAG（检索增强生成）系统，用于查询课程材料。

**核心流程**：用户查询 → Claude API → 工具调用决策 → 向量搜索 → 基于搜索结果生成响应

**关键组件**（backend/）：
- `app.py` — FastAPI 入口，`/api/query` 和 `/api/courses` 端点
- `rag_system.py` — 主协调器，串联所有组件
- `ai_generator.py` — Claude API 交互，支持工具调用
- `vector_store.py` — ChromaDB 向量存储，两个集合：
  - `course_catalog`：课程元数据（标题、讲师、链接）
  - `course_content`：课程内容分块（用于语义搜索）
- `search_tools.py` — 定义 `search_course_content` 工具供 Claude 调用
- `document_processor.py` — 解析课程文档（PDF/DOCX/TXT），分块处理
- `session_manager.py` — 会话历史管理

**数据模型**（models.py）：
- `Course` — 课程对象，title 作为唯一标识
- `Lesson` — 课时对象，含编号、标题、链接
- `CourseChunk` — 内容分块，含课程标题、课时编号、分块索引

## 课程文档格式

docs/ 目录下的课程文档格式：
```
Course Title: [标题]
Course Link: [URL]
Course Instructor: [讲师名]

Lesson 0: [课时标题]
Lesson Link: [URL]
[课时内容...]

Lesson 1: [课时标题]
...
```

应用启动时自动加载 docs/ 目录内容到向量数据库。

## 配置参数

config.py 中的关键参数：
- `CHUNK_SIZE`: 800 字符
- `CHUNK_OVERLAP`: 100 字符
- `MAX_RESULTS`: 5（搜索结果数）
- `MAX_HISTORY`: 2（会话历史条数）
- `EMBEDDING_MODEL`: all-MiniLM-L6-v2
- `ANTHROPIC_MODEL`: claude-sonnet-4-20250514

## 工具调用机制

Claude 可调用 `search_course_content` 工具：
- 参数：`query`（必需），`course_name`（可选，支持部分匹配），`lesson_number`（可选）
- `course_name` 通过向量搜索在 `course_catalog` 中匹配课程标题
- 搜索结果格式化为 `[课程名 - Lesson N]` 前缀 + 内容

## 前端

静态文件在 frontend/：index.html、script.js、style.css
- 使用 marked.js 解析 Markdown
- 会话 ID 由后端生成，前端持久化存储