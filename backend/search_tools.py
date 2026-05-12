from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from vector_store import SearchResults, VectorStore


class Tool(ABC):
    """Abstract base class for all tools"""

    @abstractmethod
    def get_tool_definition(self) -> Dict[str, Any]:
        """Return Anthropic tool definition for this tool"""
        pass

    @abstractmethod
    def execute(self, **kwargs) -> str:
        """Execute the tool with given parameters"""
        pass


class CourseSearchTool(Tool):
    """Tool for searching course content with semantic course name matching"""

    def __init__(self, vector_store: VectorStore):
        self.store = vector_store
        self.last_sources = []  # Track sources from last search

    def get_tool_definition(self) -> Dict[str, Any]:
        """Return Anthropic tool definition for this tool"""
        return {
            "name": "search_course_content",
            "description": "Search course materials with smart course name matching and lesson filtering",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to search for in the course content",
                    },
                    "course_name": {
                        "type": "string",
                        "description": "Course title (partial matches work, e.g. 'MCP', 'Introduction')",
                    },
                    "lesson_number": {
                        "type": "integer",
                        "description": "Specific lesson number to search within (e.g. 1, 2, 3)",
                    },
                },
                "required": ["query"],
            },
        }

    def execute(
        self,
        query: str,
        course_name: Optional[str] = None,
        lesson_number: Optional[int] = None,
    ) -> str:
        """
        Execute the search tool with given parameters.

        Args:
            query: What to search for
            course_name: Optional course filter
            lesson_number: Optional lesson filter

        Returns:
            Formatted search results or error message
        """

        # Use the vector store's unified search interface
        results = self.store.search(
            query=query, course_name=course_name, lesson_number=lesson_number
        )

        # Handle errors
        if results.error:
            return results.error

        # Handle empty results
        if results.is_empty():
            filter_info = ""
            if course_name:
                filter_info += f" in course '{course_name}'"
            if lesson_number:
                filter_info += f" in lesson {lesson_number}"
            return f"No relevant content found{filter_info}."

        # Format and return results
        return self._format_results(results)

    def _format_results(self, results: SearchResults) -> str:
        """Format search results with course and lesson context"""
        formatted = []
        sources = []  # Track sources for the UI
        seen_display_texts = set()  # Track unique sources to avoid duplicates

        # Collect all (course_title, lesson_number) pairs for batch link lookup
        course_lesson_pairs = []
        for meta in results.metadata:
            course_title = meta.get("course_title", "unknown")
            lesson_num = meta.get("lesson_number")
            if lesson_num is not None:
                course_lesson_pairs.append((course_title, lesson_num))

        # Batch get lesson links
        links_map = self.store.get_lesson_links_batch(course_lesson_pairs)

        for doc, meta in zip(results.documents, results.metadata):
            course_title = meta.get("course_title", "unknown")
            lesson_num = meta.get("lesson_number")

            # Build context header
            header = f"[{course_title}"
            if lesson_num is not None:
                header += f" - Lesson {lesson_num}"
            header += "]"

            # Get lesson link from batch results
            lesson_link = None
            if lesson_num is not None:
                lesson_link = links_map.get((course_title, lesson_num))

            # Build SourceInfo for API response - only add if not already seen
            display_text = course_title
            if lesson_num is not None:
                display_text = f"{course_title} - Lesson {lesson_num}"

            if display_text not in seen_display_texts:
                sources.append({"display_text": display_text, "link": lesson_link})
                seen_display_texts.add(display_text)

            formatted.append(f"{header}\n{doc}")

        # Store sources for retrieval
        self.last_sources = sources

        return "\n\n".join(formatted)


class CourseOutlineTool(Tool):
    """Tool for retrieving course outline with lesson list"""

    def __init__(self, vector_store: VectorStore):
        self.store = vector_store

    def get_tool_definition(self) -> Dict[str, Any]:
        """Return Anthropic tool definition for this tool"""
        return {
            "name": "get_course_outline",
            "description": "Get the complete outline of a course including course link and all lessons with their numbers and titles",
            "input_schema": {
                "type": "object",
                "properties": {
                    "course_title": {
                        "type": "string",
                        "description": "The exact or partial course title to look up (e.g. 'MCP', 'Introduction to Claude')",
                    }
                },
                "required": ["course_title"],
            },
        }

    def execute(self, course_title: str) -> str:
        """
        Execute the outline tool with given course title.

        Args:
            course_title: Course title to look up (partial matches supported)

        Returns:
            Formatted course outline or error message
        """
        # First resolve the course title using vector search if partial
        resolved_title = self.store._resolve_course_name(course_title)
        if not resolved_title:
            return f"No course found matching '{course_title}'."

        # Get the course outline
        outline = self.store.get_course_outline(resolved_title)
        if not outline:
            return f"Could not retrieve outline for course '{resolved_title}'."

        # Format the output
        return self._format_outline(outline)

    def _format_outline(self, outline: Dict[str, Any]) -> str:
        """Format course outline for display"""
        lines = []

        # Course header
        lines.append(f"Course: {outline['course_title']}")
        if outline["course_link"]:
            lines.append(f"Course Link: {outline['course_link']}")

        # Lessons list
        if outline["lessons"]:
            lines.append("\nLessons:")
            for lesson in outline["lessons"]:
                lesson_num = lesson.get("lesson_number")
                lesson_title = lesson.get("lesson_title")
                if lesson_num is not None and lesson_title:
                    lines.append(f"  Lesson {lesson_num}: {lesson_title}")

        return "\n".join(lines)


class ToolManager:
    """Manages available tools for the AI"""

    def __init__(self):
        self.tools = {}

    def register_tool(self, tool: Tool):
        """Register any tool that implements the Tool interface"""
        tool_def = tool.get_tool_definition()
        tool_name = tool_def.get("name")
        if not tool_name:
            raise ValueError("Tool must have a 'name' in its definition")
        self.tools[tool_name] = tool

    def get_tool_definitions(self) -> list:
        """Get all tool definitions for Anthropic tool calling"""
        return [tool.get_tool_definition() for tool in self.tools.values()]

    def execute_tool(self, tool_name: str, **kwargs) -> str:
        """Execute a tool by name with given parameters"""
        if tool_name not in self.tools:
            return f"Tool '{tool_name}' not found"

        return self.tools[tool_name].execute(**kwargs)

    def get_last_sources(self) -> list:
        """Get sources from the last search operation, formatted for API response"""
        # Check all tools for last_sources attribute
        for tool in self.tools.values():
            if hasattr(tool, "last_sources") and tool.last_sources:
                # Return structured sources (list of {display_text, link})
                return tool.last_sources
        return []

    def reset_sources(self):
        """Reset sources from all tools that track sources"""
        for tool in self.tools.values():
            if hasattr(tool, "last_sources"):
                tool.last_sources = []
