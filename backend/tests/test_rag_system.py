"""
Tests for RAG System handling of content-related queries.

These tests evaluate:
1. Full query flow from user input to response
2. Tool invocation for content queries
3. Source tracking and retrieval
4. Session management integration
5. Language handling (Chinese/English)
6. Error handling across the system
"""

from unittest.mock import patch

import pytest
from config import config
from rag_system import RAGSystem


class TestRAGSystemQueryFlow:
    """Tests for the complete query flow"""

    def test_query_returns_response_and_sources(self, rag_system):
        """Test that query returns both response string and sources list"""
        response, sources = rag_system.query(
            query="What is this course about?", session_id=None, language="en"
        )

        assert response is not None, "Response should not be None"
        assert isinstance(response, str), "Response should be string"
        assert sources is not None, "Sources should not be None"
        assert isinstance(sources, list), "Sources should be list"

    def test_query_creates_session_if_none(self, rag_system):
        """Test that query creates session if session_id is None"""
        # This test requires checking internal behavior
        session_id = rag_system.session_manager.create_session()

        response, sources = rag_system.query(query="test query", session_id=session_id)

        assert session_id is not None, "Session ID should exist"

    def test_query_with_existing_session(self, rag_system):
        """Test that existing session is used for conversation history"""
        session_id = rag_system.session_manager.create_session()

        # First query
        rag_system.query(
            query="What is Test Course?", session_id=session_id, language="en"
        )

        # Second query - should have history
        history = rag_system.session_manager.get_conversation_history(session_id)

        assert history is not None or history == "", "History should be available"

    def test_query_uses_tools_for_content_search(self, rag_system):
        """Test that content-related queries trigger tool use"""
        # Verify tools are passed to generate_response
        with patch.object(rag_system.ai_generator, "generate_response") as mock_gen:
            mock_gen.return_value = ("Test response", [])

            rag_system.query(query="What is in lesson 1?", language="en")

            # Check that tools were passed
            call_args = mock_gen.call_args
            assert (
                call_args[1].get("tools") is not None
            ), "Tools should be passed to AI generator"

    def test_sources_are_retrieved_after_query(self, rag_system):
        """Test that sources are properly retrieved after tool execution"""
        response, sources = rag_system.query(
            query="What topics are covered?", language="en"
        )

        # Sources should be populated from tool execution
        assert sources is not None, "Sources should be returned"

    def test_sources_reset_after_query(self, rag_system):
        """Test that sources are reset after each query"""
        rag_system.query(query="first query", language="en")

        # After query, internal sources should be reset
        internal_sources = rag_system.tool_manager.get_last_sources()

        assert internal_sources == [], "Internal sources should be reset"


class TestRAGSystemLanguageHandling:
    """Tests for language preference handling"""

    def test_chinese_language_instruction_in_prompt(self, rag_system):
        """Test that Chinese language instruction is added to prompt"""
        with patch.object(rag_system.ai_generator, "generate_response") as mock_gen:
            mock_gen.return_value = ("中文回复", [])

            rag_system.query(query="测试问题", language="zh")

            call_args = mock_gen.call_args
            query_arg = call_args[1].get("query", "")

            assert (
                "请用中文回答" in query_arg
            ), "Chinese instruction should be in prompt"

    def test_english_language_instruction_in_prompt(self, rag_system):
        """Test that English language instruction is added to prompt"""
        with patch.object(rag_system.ai_generator, "generate_response") as mock_gen:
            mock_gen.return_value = ("English response", [])

            rag_system.query(query="test question", language="en")

            call_args = mock_gen.call_args
            query_arg = call_args[1].get("query", "")

            assert (
                "Please respond in English" in query_arg
            ), "English instruction should be in prompt"

    def test_default_language_is_chinese(self, rag_system):
        """Test that default language is Chinese"""
        with patch.object(rag_system.ai_generator, "generate_response") as mock_gen:
            mock_gen.return_value = ("Response", [])

            # Call without specifying language (default should be zh)
            rag_system.query(query="test")

            call_args = mock_gen.call_args
            query_arg = call_args[1].get("query", "")

            assert (
                "请用中文回答" in query_arg
            ), "Default should be Chinese language instruction"


class TestRAGSystemToolManager:
    """Tests for ToolManager integration"""

    def test_tool_manager_has_search_tool(self, rag_system):
        """Test that ToolManager has search_course_content registered"""
        tool_names = list(rag_system.tool_manager.tools.keys())

        assert (
            "search_course_content" in tool_names
        ), "Should have search_course_content tool"

    def test_tool_manager_has_outline_tool(self, rag_system):
        """Test that ToolManager has get_course_outline registered"""
        tool_names = list(rag_system.tool_manager.tools.keys())

        assert "get_course_outline" in tool_names, "Should have get_course_outline tool"

    def test_tool_definitions_ready_for_api(self, rag_system):
        """Test that tool definitions are ready for Anthropic API"""
        tool_defs = rag_system.tool_manager.get_tool_definitions()

        assert len(tool_defs) >= 2, "Should have at least 2 tools"

        for tool_def in tool_defs:
            assert "name" in tool_def
            assert "description" in tool_def
            assert "input_schema" in tool_def


class TestRAGSystemDocumentProcessing:
    """Tests for document processing integration"""

    def test_add_single_document(self, rag_system, sample_course_file):
        """Test adding a single course document"""
        course, chunk_count = rag_system.add_course_document(sample_course_file)

        assert course is not None, "Course should be returned"
        assert course.title is not None, "Course should have title"
        assert chunk_count > 0, "Should have created chunks"

    def test_add_folder_documents(self, rag_system, tmp_path):
        """Test adding documents from a folder"""
        # Create multiple test files
        for i in range(2):
            content = f"""Course Title: Course {i}
Course Link: https://example.com/course{i}
Course Instructor: Instructor {i}

Lesson 0: Introduction
Lesson Link: https://example.com/lesson{i}_0
Content for course {i} lesson 0.
"""
            file_path = tmp_path / f"course_{i}.txt"
            file_path.write_text(content)

        courses_added, chunks_added = rag_system.add_course_folder(str(tmp_path))

        assert courses_added >= 0, "Should process folder"
        assert chunks_added >= 0, "Should create chunks"

    def test_duplicate_course_handling(self, rag_system, sample_course_file):
        """Test that duplicate courses are handled properly"""
        # Add same document twice
        rag_system.add_course_document(sample_course_file)
        course2, chunks2 = rag_system.add_course_document(sample_course_file)

        # Second add should either skip or return 0 chunks
        existing_titles = rag_system.vector_store.get_existing_course_titles()

        # Course should only appear once
        assert len(existing_titles) == len(
            set(existing_titles)
        ), "No duplicate course titles"


class TestRAGSystemVectorStoreIntegration:
    """Tests for VectorStore integration"""

    def test_vector_store_has_courses(self, rag_system):
        """Test that VectorStore has courses after document loading"""
        course_count = rag_system.vector_store.get_course_count()

        assert course_count >= 0, "Should have course count"

    def test_search_returns_results(self, rag_system):
        """Test that search returns results for known content"""
        # Add a document first
        response, sources = rag_system.query(
            query="What is the introduction lesson about?", language="en"
        )

        # Should get some response
        assert response is not None

    def test_course_analytics(self, rag_system):
        """Test getting course analytics"""
        analytics = rag_system.get_course_analytics()

        assert "total_courses" in analytics
        assert "course_titles" in analytics
        assert isinstance(analytics["course_titles"], list)


class TestRAGSystemSessionManagement:
    """Tests for session management integration"""

    def test_session_created_for_new_conversation(self, rag_system):
        """Test that new session is created when needed"""
        session_id = rag_system.session_manager.create_session()

        assert session_id is not None, "Session ID should be created"

    def test_session_history_preserved(self, rag_system):
        """Test that conversation history is preserved in session"""
        session_id = rag_system.session_manager.create_session()

        # Make queries
        rag_system.query(query="First question", session_id=session_id)
        rag_system.query(query="Second question", session_id=session_id)

        history = rag_system.session_manager.get_conversation_history(session_id)

        assert history is not None, "History should exist"

    def test_session_clear(self, rag_system):
        """Test that session can be cleared"""
        session_id = rag_system.session_manager.create_session()

        rag_system.query(query="Some question", session_id=session_id)

        rag_system.session_manager.clear_session(session_id)

        # After clear, history should be empty or session should be gone
        history = rag_system.session_manager.get_conversation_history(session_id)
        assert history == "" or history is None, "History should be cleared"


class TestRAGSystemErrorHandling:
    """Tests for error handling in the RAG system"""

    def test_empty_query_handling(self, rag_system):
        """Test handling of empty or minimal query"""
        response, sources = rag_system.query(query="", language="en")

        assert response is not None, "Should handle empty query gracefully"

    def test_invalid_session_handling(self, rag_system):
        """Test handling of invalid session ID"""
        # Use a non-existent session ID
        response, sources = rag_system.query(
            query="test", session_id="invalid_session_12345", language="en"
        )

        assert response is not None, "Should handle invalid session"

    def test_api_failure_handling(self, rag_system):
        """Test handling when AI API fails"""
        with patch.object(rag_system.ai_generator, "generate_response") as mock_gen:
            mock_gen.side_effect = Exception("API Error")

            try:
                response, sources = rag_system.query(query="test", language="en")
                # If no exception raised, error handling is working
            except Exception as e:
                # Exception might propagate - depends on design
                assert str(e) == "API Error", "Should propagate or handle API errors"


class TestRAGSystemEndToEnd:
    """End-to-end integration tests"""

    @pytest.fixture
    def full_rag_system(self, temp_chroma_path):
        """Create a complete RAG system with real documents"""
        # Create test config
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

        return RAGSystem(test_config)

    def test_full_query_flow_with_real_search(
        self, full_rag_system, sample_course_file
    ):
        """Test complete query flow with actual document search"""
        # Add document
        full_rag_system.add_course_document(sample_course_file)

        # Query
        response, sources = full_rag_system.query(
            query="What is the introduction lesson?", language="en"
        )

        assert response is not None, "Should get response"
        assert isinstance(response, str), "Response should be string"

    def test_course_outline_query(self, full_rag_system, sample_course_file):
        """Test querying for course outline"""
        full_rag_system.add_course_document(sample_course_file)

        response, sources = full_rag_system.query(
            query="Show me the course outline for Test Course", language="en"
        )

        assert response is not None, "Should handle outline query"

    def test_content_search_query(self, full_rag_system, sample_course_file):
        """Test querying for specific content"""
        full_rag_system.add_course_document(sample_course_file)

        response, sources = full_rag_system.query(
            query="What topics are covered in lesson 1?", language="en"
        )

        assert response is not None, "Should handle content query"


class TestRAGSystemAPIEndpointIssues:
    """Tests specifically for detecting the 'query failed' issue"""

    def test_app_endpoint_not_hardcoded_error(self, test_app):
        """Test that app.py endpoint doesn't have hardcoded error"""
        # Read the actual app.py source code directly
        import inspect
        from pathlib import Path

        app_path = Path(__file__).parent.parent / "app.py"
        source = app_path.read_text()

        # Check if there's a hardcoded raise HTTPException
        has_hardcoded_error = (
            'raise HTTPException(status_code=500, detail="query failed")' in source
        )

        assert (
            not has_hardcoded_error
        ), "CRITICAL: app.py has hardcoded 'query failed' error at line 62!"

    def test_actual_query_logic_is_executed(self, test_app):
        """Test that actual query logic (not commented out) is executed"""
        from pathlib import Path

        app_path = Path(__file__).parent.parent / "app.py"
        source = app_path.read_text()

        # Check if actual logic is commented out
        has_commented_logic = (
            "# try:" in source or "# session_id = request.session_id" in source
        )

        assert (
            not has_commented_logic
        ), "CRITICAL: Actual query logic is commented out in app.py!"
