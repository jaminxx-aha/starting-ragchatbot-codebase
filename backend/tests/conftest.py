import pytest
import os
import sys

# Add backend directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config
from vector_store import VectorStore
from document_processor import DocumentProcessor
from search_tools import CourseSearchTool, CourseOutlineTool, ToolManager
from ai_generator import AIGenerator
from rag_system import RAGSystem
from models import Course, Lesson, CourseChunk


@pytest.fixture
def temp_chroma_path(tmp_path):
    """Create a temporary ChromaDB path for tests"""
    return str(tmp_path / "test_chroma_db")


@pytest.fixture
def vector_store(temp_chroma_path):
    """Create a VectorStore instance for testing"""
    store = VectorStore(
        chroma_path=temp_chroma_path,
        embedding_model="all-MiniLM-L6-v2",
        max_results=5
    )
    return store


@pytest.fixture
def populated_vector_store(vector_store):
    """Create a VectorStore with sample course data"""
    # Add a sample course to the catalog
    course = Course(
        title="Introduction to Claude",
        course_link="https://example.com/course",
        instructor="Anthropic",
        lessons=[
            Lesson(lesson_number=1, title="Getting Started", lesson_link="https://example.com/lesson1"),
            Lesson(lesson_number=2, title="Advanced Usage", lesson_link="https://example.com/lesson2"),
        ]
    )
    vector_store.add_course_metadata(course)

    # Add sample content chunks
    chunks = [
        CourseChunk(
            content="Claude is a powerful AI assistant that can help with various tasks.",
            course_title="Introduction to Claude",
            lesson_number=1,
            chunk_index=0
        ),
        CourseChunk(
            content="To use Claude effectively, you should provide clear and specific prompts.",
            course_title="Introduction to Claude",
            lesson_number=1,
            chunk_index=1
        ),
        CourseChunk(
            content="Advanced features include tool use and extended thinking capabilities.",
            course_title="Introduction to Claude",
            lesson_number=2,
            chunk_index=2
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
    # Modify config to use temp path
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

    # Add sample course
    system.add_course_document(sample_course_file)

    return system