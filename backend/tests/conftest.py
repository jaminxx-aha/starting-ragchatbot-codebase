import os
import sys
from unittest.mock import Mock, MagicMock, patch, AsyncMock

import pytest

# Add backend directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from config import config
from document_processor import DocumentProcessor
from models import Course, CourseChunk, Lesson
from rag_system import RAGSystem
from search_tools import CourseOutlineTool, CourseSearchTool, ToolManager
from vector_store import VectorStore
from models import SourceInfo


# ============================================================================
# Test App Configuration (without static file mounting)
# ============================================================================

def create_test_app(test_rag_system=None, raise_errors=False):
    """Create a FastAPI app for testing without static file mounting

    Args:
        test_rag_system: The RAG system instance to use
        raise_errors: If True, exceptions will be caught and converted to HTTPException(500)
    """
    app = FastAPI(title="Test App")

    # Pydantic models for request/response
    class QueryRequest(BaseModel):
        query: str
        session_id: Optional[str] = None
        language: Optional[str] = "zh"

    class QueryResponse(BaseModel):
        answer: str
        sources: List[SourceInfo]
        session_id: str

    class CourseStats(BaseModel):
        total_courses: int
        course_titles: List[str]

    rag = test_rag_system

    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        try:
            session_id = request.session_id
            if not session_id:
                session_id = rag.session_manager.create_session()
            answer, sources = rag.query(request.query, session_id, request.language)
            return QueryResponse(answer=answer, sources=sources, session_id=session_id)
        except Exception as e:
            if raise_errors:
                raise HTTPException(status_code=500, detail=str(e))
            raise

    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        try:
            analytics = rag.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"]
            )
        except Exception as e:
            if raise_errors:
                raise HTTPException(status_code=500, detail=str(e))
            raise

    @app.delete("/api/session/{session_id}")
    async def clear_session(session_id: str):
        try:
            rag.session_manager.clear_session(session_id)
            return {"status": "success", "message": "Session cleared"}
        except Exception as e:
            if raise_errors:
                raise HTTPException(status_code=500, detail=str(e))
            raise

    @app.get("/")
    async def root():
        return {"message": "Test API Root"}

    return app


# ============================================================================
# Core Fixtures
# ============================================================================

@pytest.fixture
def temp_chroma_path(tmp_path):
    """Create a temporary ChromaDB path for tests"""
    return str(tmp_path / "test_chroma_db")


@pytest.fixture
def vector_store(temp_chroma_path):
    """Create a VectorStore instance for testing"""
    store = VectorStore(
        chroma_path=temp_chroma_path, embedding_model="all-MiniLM-L6-v2", max_results=5
    )
    return store


@pytest.fixture
def populated_vector_store(vector_store):
    """Create a VectorStore with sample course data"""
    course = Course(
        title="Introduction to Claude",
        course_link="https://example.com/course",
        instructor="Anthropic",
        lessons=[
            Lesson(
                lesson_number=1,
                title="Getting Started",
                lesson_link="https://example.com/lesson1",
            ),
            Lesson(
                lesson_number=2,
                title="Advanced Usage",
                lesson_link="https://example.com/lesson2",
            ),
        ],
    )
    vector_store.add_course_metadata(course)

    chunks = [
        CourseChunk(
            content="Claude is a powerful AI assistant that can help with various tasks.",
            course_title="Introduction to Claude",
            lesson_number=1,
            chunk_index=0,
        ),
        CourseChunk(
            content="To use Claude effectively, you should provide clear and specific prompts.",
            course_title="Introduction to Claude",
            lesson_number=1,
            chunk_index=1,
        ),
        CourseChunk(
            content="Advanced features include tool use and extended thinking capabilities.",
            course_title="Introduction to Claude",
            lesson_number=2,
            chunk_index=2,
        ),
    ]
    vector_store.add_course_content(chunks)

    return vector_store


@pytest.fixture
def search_tool(populated_vector_store):
    """Create a CourseSearchTool with populated data"""
    return CourseSearchTool(populated_vector_store)


@pytest.fixture
def outline_tool(populated_vector_store):
    """Create a CourseOutlineTool with populated data"""
    return CourseOutlineTool(populated_vector_store)


@pytest.fixture
def tool_manager(search_tool, outline_tool):
    """Create a ToolManager with both tools"""
    manager = ToolManager()
    manager.register_tool(search_tool)
    manager.register_tool(outline_tool)
    return manager


@pytest.fixture
def document_processor():
    """Create a DocumentProcessor instance"""
    return DocumentProcessor(chunk_size=800, chunk_overlap=100)


@pytest.fixture
def sample_course_file(tmp_path):
    """Create a sample course document file"""
    content = """Course Title: Test Course for RAG
Course Link: https://test.example.com/course
Course Instructor: Test Instructor

Lesson 0: Introduction
Lesson Link: https://test.example.com/lesson0
This is the introduction lesson content. It explains the basics of the course.

Lesson 1: Main Concepts
Lesson Link: https://test.example.com/lesson1
This lesson covers the main concepts that students need to understand.
Key topics include retrieval, generation, and augmentation.
"""
    file_path = tmp_path / "test_course.txt"
    file_path.write_text(content)
    return str(file_path)


@pytest.fixture
def rag_system(temp_chroma_path, sample_course_file):
    """Create a full RAG system for integration testing"""
    test_config = config.__class__.__new__(config.__class__)
    test_config.CHROMA_PATH = temp_chroma_path
    test_config.CHUNK_SIZE = 800
    test_config.CHUNK_OVERLAP = 100
    test_config.MAX_RESULTS = 5
    test_config.MAX_HISTORY = 2
    test_config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    test_config.ANTHROPIC_API_KEY = config.ANTHROPIC_API_KEY
    test_config.ANTHROPIC_MODEL = config.ANTHROPIC_MODEL
    test_config.ANTHROPIC_BASE_URL = config.ANTHROPIC_BASE_URL

    system = RAGSystem(test_config)
    system.add_course_document(sample_course_file)

    return system


# ============================================================================
# API Testing Fixtures
# ============================================================================

@pytest.fixture
def test_app(rag_system):
    """Create a test FastAPI app without static file mounting"""
    return create_test_app(test_rag_system=rag_system)


@pytest.fixture
def client(test_app):
    """Create a TestClient for API testing"""
    return TestClient(test_app)


@pytest.fixture
def mock_rag_system():
    """Create a mocked RAG system for API tests that don't need real backend"""
    mock_system = MagicMock()

    # Mock session manager
    mock_session_manager = MagicMock()
    mock_session_manager.create_session.return_value = "test-session-123"
    mock_session_manager.clear_session.return_value = None
    mock_system.session_manager = mock_session_manager

    # Mock query method
    mock_system.query.return_value = (
        "这是测试回复",
        [
            SourceInfo(display_text="Test Course - Lesson 1", link="https://test.example.com/lesson1"),
            SourceInfo(display_text="Test Course - Lesson 2", link="https://test.example.com/lesson2"),
        ]
    )

    # Mock analytics
    mock_system.get_course_analytics.return_value = {
        "total_courses": 3,
        "course_titles": ["Course A", "Course B", "Course C"]
    }

    return mock_system


@pytest.fixture
def mock_app(mock_rag_system):
    """Create a test app with mocked RAG system"""
    return create_test_app(test_rag_system=mock_rag_system)


@pytest.fixture
def mock_client(mock_app):
    """Create a TestClient with mocked RAG system"""
    return TestClient(mock_app)


@pytest.fixture
def sample_query_request():
    """Sample query request data"""
    return {
        "query": "What is Claude?",
        "session_id": None,
        "language": "en"
    }


@pytest.fixture
def sample_query_request_chinese():
    """Sample query request in Chinese"""
    return {
        "query": "Claude 是什么？",
        "session_id": None,
        "language": "zh"
    }


@pytest.fixture
def api_error_rag_system():
    """Mock RAG system that raises errors"""
    mock_system = MagicMock()
    mock_session_manager = MagicMock()
    mock_session_manager.create_session.return_value = "error-session"
    mock_system.session_manager = mock_session_manager

    # Mock query to raise exception
    mock_system.query.side_effect = Exception("Simulated API error")

    return mock_system


@pytest.fixture
def error_app(api_error_rag_system):
    """Create a test app that simulates errors with proper HTTPException handling"""
    return create_test_app(test_rag_system=api_error_rag_system, raise_errors=True)


@pytest.fixture
def error_client(error_app):
    """Create a TestClient with error-raising RAG system"""
    return TestClient(error_app)
