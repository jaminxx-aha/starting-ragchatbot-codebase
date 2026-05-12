"""
Tests for CourseSearchTool.execute method.

These tests evaluate:
1. Basic search functionality (query only)
2. Search with course name filtering
3. Search with lesson number filtering
4. Combined filters (course name + lesson number)
5. Error handling for invalid inputs
6. Edge cases (empty results, partial matches)
"""

import pytest
from search_tools import CourseSearchTool
from models import Course, Lesson, CourseChunk


class TestCourseSearchToolExecute:
    """Tests for the execute method of CourseSearchTool"""

    def test_execute_basic_query_returns_results(self, populated_vector_store):
        """Test that a basic query returns relevant results"""
        tool = CourseSearchTool(populated_vector_store)

        result = tool.execute(query="Claude AI assistant")

        assert result is not None, "Result should not be None"
        assert "Claude" in result or "AI" in result, "Result should contain relevant content"
        assert tool.last_sources is not None, "Sources should be tracked"

    def test_execute_query_with_exact_course_name(self, populated_vector_store):
        """Test search with exact course name filter"""
        tool = CourseSearchTool(populated_vector_store)

        result = tool.execute(
            query="prompts",
            course_name="Introduction to Claude"
        )

        assert result is not None, "Result should not be None"
        # Should find content from the specified course
        assert "Introduction to Claude" in result, "Result should be from the specified course"

    def test_execute_query_with_partial_course_name(self, populated_vector_store):
        """Test that partial course name matches work"""
        tool = CourseSearchTool(populated_vector_store)

        # Use partial course name "Claude" instead of full title
        result = tool.execute(
            query="features",
            course_name="Claude"  # Partial match for "Introduction to Claude"
        )

        assert result is not None, "Result should not be None"
        # Should resolve to full course title
        assert "Introduction to Claude" in result or "features" in result.lower()

    def test_execute_query_with_lesson_number(self, populated_vector_store):
        """Test search filtered by lesson number"""
        tool = CourseSearchTool(populated_vector_store)

        result = tool.execute(
            query="content",
            lesson_number=1
        )

        assert result is not None, "Result should not be None"
        assert "Lesson 1" in result, "Result should be from lesson 1"

    def test_execute_query_with_combined_filters(self, populated_vector_store):
        """Test search with both course name and lesson number"""
        tool = CourseSearchTool(populated_vector_store)

        result = tool.execute(
            query="AI",
            course_name="Introduction to Claude",
            lesson_number=1
        )

        assert result is not None, "Result should not be None"
        assert "Introduction to Claude" in result, "Should be from correct course"
        assert "Lesson 1" in result, "Should be from correct lesson"

    def test_execute_returns_message_for_empty_catalog(self, vector_store):
        """Test that searching in empty catalog returns appropriate message"""
        # Use an empty vector store (no courses)
        tool = CourseSearchTool(vector_store)

        # Search without any course content loaded
        result = tool.execute(query="any query")

        # Should return no results message
        assert "No relevant content found" in result, \
            "Should return 'no content found' message for empty catalog"

    def test_execute_returns_message_for_no_results(self, vector_store):
        """Test behavior when search finds no results"""
        tool = CourseSearchTool(vector_store)

        result = tool.execute(query="random unrelated query xyz123")

        assert "No relevant content found" in result, \
            "Should return 'no content found' message for empty results"

    def test_execute_sources_are_tracked(self, populated_vector_store):
        """Test that sources are properly tracked after search"""
        tool = CourseSearchTool(populated_vector_store)

        tool.execute(query="Claude")

        sources = tool.last_sources
        assert sources is not None, "Sources should be tracked"
        assert isinstance(sources, list), "Sources should be a list"

        if sources:  # If results found
            assert "display_text" in sources[0], "Source should have display_text"
            assert "link" in sources[0], "Source should have link field"

    def test_execute_sources_contain_correct_links(self, populated_vector_store):
        """Test that sources include correct lesson links"""
        tool = CourseSearchTool(populated_vector_store)

        tool.execute(query="Claude", lesson_number=1)

        sources = tool.last_sources
        if sources:
            # Check that link is populated for lesson 1
            for source in sources:
                if "Lesson 1" in source.get("display_text", ""):
                    assert source.get("link") is not None, \
                        "Lesson 1 source should have a link"

    def test_execute_empty_vector_store(self, vector_store):
        """Test behavior with empty vector store"""
        tool = CourseSearchTool(vector_store)

        result = tool.execute(query="any query")

        assert "No relevant content found" in result, \
            "Should return no results message for empty store"

    def test_execute_handles_special_characters(self, populated_vector_store):
        """Test that special characters in query don't cause errors"""
        tool = CourseSearchTool(populated_vector_store)

        result = tool.execute(query="Claude & AI - features!")

        assert result is not None, "Should handle special characters without error"

    def test_execute_multilingual_query(self, populated_vector_store):
        """Test that multilingual queries work"""
        tool = CourseSearchTool(populated_vector_store)

        # Chinese query
        result = tool.execute(query="Claude 功能")

        assert result is not None, "Should handle multilingual queries"


class TestCourseSearchToolFormatting:
    """Tests for result formatting in CourseSearchTool"""

    def test_format_includes_course_header(self, populated_vector_store):
        """Test that formatted results include course header"""
        tool = CourseSearchTool(populated_vector_store)

        result = tool.execute(query="Claude")

        if "No relevant content found" not in result:
            assert "[" in result, "Formatted result should include header brackets"
            assert "Introduction to Claude" in result, "Header should include course title"

    def test_format_includes_lesson_info(self, populated_vector_store):
        """Test that formatted results include lesson information"""
        tool = CourseSearchTool(populated_vector_store)

        result = tool.execute(query="features", lesson_number=2)

        if "No relevant content found" not in result:
            assert "Lesson 2" in result, "Result should include lesson number"

    def test_sources_deduplication(self, populated_vector_store):
        """Test that sources are deduplicated"""
        tool = CourseSearchTool(populated_vector_store)

        # Multiple searches from same lesson
        tool.execute(query="Claude", lesson_number=1)
        tool.execute(query="prompts", lesson_number=1)

        sources = tool.last_sources
        display_texts = [s.get("display_text") for s in sources]

        # Check no duplicates in last sources
        assert len(display_texts) == len(set(display_texts)), \
            "Sources should not have duplicates"


class TestCourseSearchToolIntegration:
    """Integration tests for CourseSearchTool with VectorStore"""

    def test_tool_definition_format(self, search_tool):
        """Test that tool definition is correctly formatted for Anthropic"""
        tool_def = search_tool.get_tool_definition()

        assert tool_def["name"] == "search_course_content"
        assert "input_schema" in tool_def
        assert "properties" in tool_def["input_schema"]
        assert "query" in tool_def["input_schema"]["properties"]
        assert tool_def["input_schema"]["required"] == ["query"]

    def test_execute_with_tool_manager(self, tool_manager):
        """Test that ToolManager can execute the search tool"""
        result = tool_manager.execute_tool(
            "search_course_content",
            query="Claude"
        )

        assert result is not None, "ToolManager should execute tool successfully"

    def test_get_last_sources_through_tool_manager(self, tool_manager):
        """Test retrieving sources through ToolManager"""
        tool_manager.execute_tool("search_course_content", query="Claude")

        sources = tool_manager.get_last_sources()

        assert sources is not None, "ToolManager should return sources"

    def test_reset_sources_clears_sources(self, tool_manager):
        """Test that reset_sources clears the sources"""
        tool_manager.execute_tool("search_course_content", query="Claude")

        # Sources should be populated
        sources_before = tool_manager.get_last_sources()
        assert sources_before is not None

        tool_manager.reset_sources()

        # After reset, should be empty
        sources_after = tool_manager.get_last_sources()
        assert sources_after == [], "Sources should be empty after reset"