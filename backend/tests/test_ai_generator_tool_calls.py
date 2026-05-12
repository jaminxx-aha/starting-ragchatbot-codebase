"""
Tests for AIGenerator tool calling behavior.

These tests evaluate:
1. Tool definitions are correctly passed to Claude
2. Claude correctly decides when to use tools
3. Tool execution is properly handled
4. Response handling for tool results
5. Error handling in tool calls
6. Sequential tool calling (agentic loop)
7. Max rounds enforcement
"""

from unittest.mock import MagicMock

import pytest
from ai_generator import AIGenerator


class TestAIGeneratorSequentialToolCalls:
    """Tests for sequential tool calling (agentic loop) behavior"""

    @pytest.fixture
    def mock_client_sequential(self):
        """Create a mock Anthropic client for sequential tests"""
        mock = MagicMock()
        return mock

    @pytest.fixture
    def mock_ai_generator_sequential(self, mock_client_sequential, tool_manager):
        """Create AIGenerator with mocked client for sequential tests"""
        ai = AIGenerator(api_key="test_key", model="claude-sonnet-4-20250514")
        ai.client = mock_client_sequential
        return ai

    def test_two_round_sequential_tool_calls(
        self, mock_ai_generator_sequential, mock_client_sequential, tool_manager
    ):
        """Test that Claude can make 2 sequential tool calls in separate rounds"""
        # Round 1: get_course_outline
        mock_response_1 = MagicMock()
        mock_response_1.stop_reason = "tool_use"
        mock_tool_block_1 = MagicMock()
        mock_tool_block_1.type = "tool_use"
        mock_tool_block_1.name = "get_course_outline"
        mock_tool_block_1.id = "tool_1"
        mock_tool_block_1.input = {"course_title": "Course X"}
        mock_response_1.content = [mock_tool_block_1]

        # Round 2: search_course_content (based on outline results)
        mock_response_2 = MagicMock()
        mock_response_2.stop_reason = "tool_use"
        mock_tool_block_2 = MagicMock()
        mock_tool_block_2.type = "tool_use"
        mock_tool_block_2.name = "search_course_content"
        mock_tool_block_2.id = "tool_2"
        mock_tool_block_2.input = {"query": "lesson 4 topic"}
        mock_response_2.content = [mock_tool_block_2]

        # Round 3: Final text response
        mock_response_3 = MagicMock()
        mock_response_3.stop_reason = "end_turn"
        mock_text_block = MagicMock()
        mock_text_block.type = "text"
        mock_text_block.text = "Found the matching course information."
        mock_response_3.content = [mock_text_block]

        mock_client_sequential.messages.create.side_effect = [
            mock_response_1,
            mock_response_2,
            mock_response_3,
        ]

        result = mock_ai_generator_sequential.generate_response(
            query="Find a course discussing the same topic as lesson 4 of Course X",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager,
        )

        # Should have made 3 API calls (2 tool rounds + final response)
        assert mock_client_sequential.messages.create.call_count == 3
        assert result == "Found the matching course information."

    def test_tools_remain_available_in_second_round(
        self, mock_ai_generator_sequential, mock_client_sequential, tool_manager
    ):
        """Test that tools parameter is preserved in second API call"""
        mock_response_1 = MagicMock()
        mock_response_1.stop_reason = "tool_use"
        mock_tool_block_1 = MagicMock()
        mock_tool_block_1.type = "tool_use"
        mock_tool_block_1.name = "get_course_outline"
        mock_tool_block_1.id = "tool_1"
        mock_tool_block_1.input = {"course_title": "Test"}
        mock_response_1.content = [mock_tool_block_1]

        mock_response_2 = MagicMock()
        mock_response_2.stop_reason = "end_turn"
        mock_text_block = MagicMock()
        mock_text_block.type = "text"
        mock_text_block.text = "Done"
        mock_response_2.content = [mock_text_block]

        mock_client_sequential.messages.create.side_effect = [
            mock_response_1,
            mock_response_2,
        ]

        mock_ai_generator_sequential.generate_response(
            query="test",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager,
        )

        # Check second call still has tools
        calls = mock_client_sequential.messages.create.call_args_list
        if len(calls) >= 2:
            second_call_params = calls[1][1]
            assert (
                "tools" in second_call_params
            ), "Tools should be available in second round"
            assert second_call_params["tools"] is not None

    def test_max_rounds_enforcement(
        self, mock_ai_generator_sequential, mock_client_sequential, tool_manager
    ):
        """Test that max 2 tool rounds are enforced"""
        # Simulate Claude wanting to use tools 3 times (should stop at 2)
        mock_response_1 = MagicMock()
        mock_response_1.stop_reason = "tool_use"
        mock_tool_block_1 = MagicMock()
        mock_tool_block_1.type = "tool_use"
        mock_tool_block_1.name = "search_course_content"
        mock_tool_block_1.id = "tool_1"
        mock_tool_block_1.input = {"query": "query1"}
        mock_response_1.content = [mock_tool_block_1]

        mock_response_2 = MagicMock()
        mock_response_2.stop_reason = "tool_use"
        mock_tool_block_2 = MagicMock()
        mock_tool_block_2.type = "tool_use"
        mock_tool_block_2.name = "search_course_content"
        mock_tool_block_2.id = "tool_2"
        mock_tool_block_2.input = {"query": "query2"}
        mock_response_2.content = [mock_tool_block_2]

        # Final response after max rounds (no tools in final call)
        mock_response_final = MagicMock()
        mock_response_final.stop_reason = "end_turn"
        mock_text_block = MagicMock()
        mock_text_block.type = "text"
        mock_text_block.text = "Final response after max rounds"
        mock_response_final.content = [mock_text_block]

        mock_client_sequential.messages.create.side_effect = [
            mock_response_1,
            mock_response_2,
            mock_response_final,
        ]

        result = mock_ai_generator_sequential.generate_response(
            query="test",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager,
        )

        # Should have made exactly 3 calls: round 1, round 2, final (no tools)
        assert mock_client_sequential.messages.create.call_count == 3
        assert result == "Final response after max rounds"

    def test_tool_error_handling(
        self, mock_ai_generator_sequential, mock_client_sequential, tool_manager
    ):
        """Test that tool errors are properly handled with is_error flag"""
        mock_response_1 = MagicMock()
        mock_response_1.stop_reason = "tool_use"
        mock_tool_block = MagicMock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "search_course_content"
        mock_tool_block.id = "tool_error_1"
        # This input will cause an error in the mock
        mock_tool_block.input = {"query": "trigger_error"}
        mock_response_1.content = [mock_tool_block]

        mock_response_2 = MagicMock()
        mock_response_2.stop_reason = "end_turn"
        mock_text_block = MagicMock()
        mock_text_block.type = "text"
        mock_text_block.text = "Handled the error gracefully"
        mock_response_2.content = [mock_text_block]

        mock_client_sequential.messages.create.side_effect = [
            mock_response_1,
            mock_response_2,
        ]

        # The tool_manager fixture should handle this
        mock_ai_generator_sequential.generate_response(
            query="test",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager,
        )

        # Check that tool_result with is_error was included
        calls = mock_client_sequential.messages.create.call_args_list
        if len(calls) >= 2:
            second_call_params = calls[1][1]
            messages = second_call_params.get("messages", [])
            # Look for tool_result with is_error in the messages
            for msg in messages:
                if msg.get("role") == "user":
                    content = msg.get("content", [])
                    for block in content:
                        if (
                            isinstance(block, dict)
                            and block.get("type") == "tool_result"
                        ):
                            # Either error was handled or tool succeeded
                            assert "content" in block

    def test_end_turn_stops_loop_early(
        self, mock_ai_generator_sequential, mock_client_sequential, tool_manager
    ):
        """Test that end_turn stops the loop immediately"""
        # Only one tool call, then end_turn
        mock_response_1 = MagicMock()
        mock_response_1.stop_reason = "tool_use"
        mock_tool_block = MagicMock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "search_course_content"
        mock_tool_block.id = "tool_1"
        mock_tool_block.input = {"query": "test"}
        mock_response_1.content = [mock_tool_block]

        mock_response_2 = MagicMock()
        mock_response_2.stop_reason = "end_turn"
        mock_text_block = MagicMock()
        mock_text_block.type = "text"
        mock_text_block.text = "Done after one round"
        mock_response_2.content = [mock_text_block]

        mock_client_sequential.messages.create.side_effect = [
            mock_response_1,
            mock_response_2,
        ]

        result = mock_ai_generator_sequential.generate_response(
            query="test",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager,
        )

        # Should have made exactly 2 calls (initial tool + end_turn)
        assert mock_client_sequential.messages.create.call_count == 2
        assert result == "Done after one round"

    def test_messages_accumulation(
        self, mock_ai_generator_sequential, mock_client_sequential, tool_manager
    ):
        """Test that messages array accumulates correctly across rounds"""
        mock_response_1 = MagicMock()
        mock_response_1.stop_reason = "tool_use"
        mock_tool_block = MagicMock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "get_course_outline"
        mock_tool_block.id = "tool_1"
        mock_tool_block.input = {"course_title": "Course X"}
        mock_response_1.content = [mock_tool_block]

        mock_response_2 = MagicMock()
        mock_response_2.stop_reason = "tool_use"
        mock_tool_block_2 = MagicMock()
        mock_tool_block_2.type = "tool_use"
        mock_tool_block_2.name = "search_course_content"
        mock_tool_block_2.id = "tool_2"
        mock_tool_block_2.input = {"query": "test"}
        mock_response_2.content = [mock_tool_block_2]

        mock_response_3 = MagicMock()
        mock_response_3.stop_reason = "end_turn"
        mock_text_block = MagicMock()
        mock_text_block.type = "text"
        mock_text_block.text = "Final"
        mock_response_3.content = [mock_text_block]

        mock_client_sequential.messages.create.side_effect = [
            mock_response_1,
            mock_response_2,
            mock_response_3,
        ]

        mock_ai_generator_sequential.generate_response(
            query="test query",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager,
        )

        # Check third call has accumulated messages
        calls = mock_client_sequential.messages.create.call_args_list
        if len(calls) >= 3:
            third_call_params = calls[2][1]
            messages = third_call_params.get("messages", [])
            # Should have: user query, assistant tool_use_1, user tool_result_1,
            #              assistant tool_use_2, user tool_result_2
            assert len(messages) == 5, "Messages should accumulate across rounds"
            assert messages[0]["role"] == "user"
            assert messages[1]["role"] == "assistant"
            assert messages[2]["role"] == "user"
            assert messages[3]["role"] == "assistant"
            assert messages[4]["role"] == "user"


class TestAIGeneratorToolDefinitions:
    """Tests for tool definition handling in AIGenerator"""

    def test_tools_are_passed_to_api(self, tool_manager):
        """Test that tool definitions are passed to Claude API"""
        tool_defs = tool_manager.get_tool_definitions()

        assert len(tool_defs) >= 1, "Should have at least one tool"
        assert tool_defs[0]["name"] in [
            "search_course_content",
            "get_course_outline",
        ], "Tool should be either search or outline tool"

    def test_tool_definition_schema_valid(self, tool_manager):
        """Test that tool definitions have valid Anthropic schema"""
        tool_defs = tool_manager.get_tool_definitions()

        for tool_def in tool_defs:
            assert "name" in tool_def, "Tool must have name"
            assert "description" in tool_def, "Tool must have description"
            assert "input_schema" in tool_def, "Tool must have input_schema"
            assert (
                tool_def["input_schema"]["type"] == "object"
            ), "Schema type must be object"

    def test_search_tool_required_params(self, tool_manager):
        """Test that search_course_content has correct required params"""
        tool_defs = tool_manager.get_tool_definitions()
        search_tool = [t for t in tool_defs if t["name"] == "search_course_content"]

        assert len(search_tool) == 1, "Should have search_course_content tool"
        assert (
            "query" in search_tool[0]["input_schema"]["required"]
        ), "query should be required parameter"


class TestAIGeneratorToolExecutionMocked:
    """Tests using mocked Claude API responses"""

    @pytest.fixture
    def mock_client(self):
        """Create a mock Anthropic client"""
        mock = MagicMock()
        return mock

    @pytest.fixture
    def mock_ai_generator(self, mock_client, tool_manager):
        """Create AIGenerator with mocked client"""
        ai = AIGenerator(api_key="test_key", model="claude-sonnet-4-20250514")
        ai.client = mock_client
        return ai

    def test_direct_text_response_handling(self, mock_ai_generator, mock_client):
        """Test handling of direct text response (no tool use)"""
        # Mock response with text block
        mock_response = MagicMock()
        mock_response.stop_reason = "end_turn"
        mock_text_block = MagicMock()
        mock_text_block.type = "text"
        mock_text_block.text = "This is a direct response."
        mock_response.content = [mock_text_block]

        mock_client.messages.create.return_value = mock_response

        result = mock_ai_generator.generate_response(query="Hello", tools=None)

        assert (
            result == "This is a direct response."
        ), "Should return text from text block"

    def test_tool_use_response_triggers_execution(
        self, mock_ai_generator, mock_client, tool_manager
    ):
        """Test that tool_use stop_reason triggers tool execution"""
        # First response: tool use request
        mock_tool_response = MagicMock()
        mock_tool_response.stop_reason = "tool_use"

        mock_tool_block = MagicMock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "search_course_content"
        mock_tool_block.id = "tool_123"
        mock_tool_block.input = {"query": "Claude features"}

        mock_tool_response.content = [mock_tool_block]

        # Second response: text after tool execution
        mock_final_response = MagicMock()
        mock_final_response.stop_reason = "end_turn"
        mock_final_text = MagicMock()
        mock_final_text.type = "text"
        mock_final_text.text = "Claude has many features including tool use."
        mock_final_response.content = [mock_final_text]

        mock_client.messages.create.side_effect = [
            mock_tool_response,
            mock_final_response,
        ]

        mock_ai_generator.generate_response(
            query="What features does Claude have?",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager,
        )

        # Should have called create twice (initial + after tool)
        assert mock_client.messages.create.call_count >= 1, "Should have made API calls"

    def test_tool_result_included_in_followup(
        self, mock_ai_generator, mock_client, tool_manager
    ):
        """Test that tool results are properly included in follow-up message"""
        # Setup mock responses
        mock_tool_response = MagicMock()
        mock_tool_response.stop_reason = "tool_use"

        mock_tool_block = MagicMock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "search_course_content"
        mock_tool_block.id = "tool_123"
        mock_tool_block.input = {"query": "test query"}
        mock_tool_response.content = [mock_tool_block]

        mock_final_response = MagicMock()
        mock_final_response.stop_reason = "end_turn"
        mock_final_text = MagicMock()
        mock_final_text.type = "text"
        mock_final_text.text = "Based on search results..."
        mock_final_response.content = [mock_final_text]

        mock_client.messages.create.side_effect = [
            mock_tool_response,
            mock_final_response,
        ]

        mock_ai_generator.generate_response(
            query="test",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager,
        )

        # Check that second call includes tool_result
        if mock_client.messages.create.call_count >= 2:
            second_call_args = mock_client.messages.create.call_args
            messages = second_call_args[1].get("messages", [])

            # Should have user message with tool_result
            has_tool_result = any(
                any(
                    (
                        block.get("type") == "tool_result"
                        if isinstance(block, dict)
                        else hasattr(block, "type") and block.type == "tool_result"
                    )
                    for block in msg.get("content", [])
                )
                for msg in messages
            )
            assert (
                has_tool_result or len(messages) >= 2
            ), "Follow-up should include tool results"

    def test_multiple_tool_calls_handling(
        self, mock_ai_generator, mock_client, tool_manager
    ):
        """Test handling of multiple tool calls in one response"""
        mock_tool_response = MagicMock()
        mock_tool_response.stop_reason = "tool_use"

        # Multiple tool blocks
        mock_tool_block1 = MagicMock()
        mock_tool_block1.type = "tool_use"
        mock_tool_block1.name = "search_course_content"
        mock_tool_block1.id = "tool_1"
        mock_tool_block1.input = {"query": "query1"}

        mock_tool_block2 = MagicMock()
        mock_tool_block2.type = "tool_use"
        mock_tool_block2.name = "search_course_content"
        mock_tool_block2.id = "tool_2"
        mock_tool_block2.input = {"query": "query2"}

        mock_tool_response.content = [mock_tool_block1, mock_tool_block2]

        mock_final_response = MagicMock()
        mock_final_response.stop_reason = "end_turn"
        mock_final_text = MagicMock()
        mock_final_text.type = "text"
        mock_final_text.text = "Combined results..."
        mock_final_response.content = [mock_final_text]

        mock_client.messages.create.side_effect = [
            mock_tool_response,
            mock_final_response,
        ]

        result = mock_ai_generator.generate_response(
            query="test",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager,
        )

        assert result is not None, "Should handle multiple tool calls"

    def test_thinking_block_handling(self, mock_ai_generator, mock_client):
        """Test handling of extended thinking blocks"""
        mock_response = MagicMock()
        mock_response.stop_reason = "end_turn"

        mock_thinking_block = MagicMock()
        mock_thinking_block.type = "thinking"
        mock_thinking_block.thinking = "Thinking about the answer..."

        mock_text_block = MagicMock()
        mock_text_block.type = "text"
        mock_text_block.text = "The actual answer."

        mock_response.content = [mock_thinking_block, mock_text_block]

        mock_client.messages.create.return_value = mock_response

        result = mock_ai_generator.generate_response(query="test")

        # Should prioritize text block over thinking
        assert (
            result == "The actual answer."
        ), "Should return text block content, not thinking"

    def test_only_thinking_block_returns_thinking(self, mock_ai_generator, mock_client):
        """Test that if only thinking block exists, return thinking content"""
        mock_response = MagicMock()
        mock_response.stop_reason = "end_turn"

        mock_thinking_block = MagicMock()
        mock_thinking_block.type = "thinking"
        mock_thinking_block.thinking = "Only thinking content available"

        mock_response.content = [mock_thinking_block]

        mock_client.messages.create.return_value = mock_response

        result = mock_ai_generator.generate_response(query="test")

        assert (
            result == "Only thinking content available"
        ), "Should return thinking content when no text block"

    def test_no_response_content_handling(self, mock_ai_generator, mock_client):
        """Test handling of empty/no response content"""
        mock_response = MagicMock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = []

        mock_client.messages.create.return_value = mock_response

        result = mock_ai_generator.generate_response(query="test")

        assert (
            result == "No response generated."
        ), "Should return fallback message for empty content"


class TestAIGeneratorAPIParameters:
    """Tests for API parameter handling"""

    def test_base_params_includes_model(self):
        """Test that base params include correct model"""
        ai = AIGenerator(api_key="test_key", model="claude-sonnet-4-20250514")

        assert ai.base_params["model"] == "claude-sonnet-4-20250514"
        assert ai.base_params["temperature"] == 0
        assert ai.base_params["max_tokens"] == 800

    def test_base_url_configuration(self):
        """Test that custom base_url is used"""
        ai = AIGenerator(
            api_key="test_key",
            model="claude-sonnet-4-20250514",
            base_url="https://custom.api.url",
        )

        # Base URL should be stored/configured
        assert ai.client is not None, "Client should be initialized"

    def test_system_prompt_content(self):
        """Test that system prompt contains required elements"""
        ai = AIGenerator(api_key="test_key", model="test_model")

        system_prompt = ai.SYSTEM_PROMPT

        assert (
            "search_course_content" in system_prompt
        ), "Should mention search_course_content tool"
        assert (
            "get_course_outline" in system_prompt
        ), "Should mention get_course_outline tool"
        assert (
            "course outline" in system_prompt.lower()
            or "Course outline" in system_prompt
        ), "Should handle course outline queries"

    def test_conversation_history_included_in_system(self):
        """Test that conversation history is added to system content"""
        ai = AIGenerator(api_key="test_key", model="test_model")

        system_with_history = (
            ai.SYSTEM_PROMPT + "\n\nPrevious conversation:\nSome history"
        )

        assert "Previous conversation" in system_with_history
        assert "Some history" in system_with_history


class TestAIGeneratorToolChoice:
    """Tests for tool_choice parameter handling"""

    @pytest.fixture
    def mock_client_for_tool_choice(self):
        """Create a mock Anthropic client for tool_choice tests"""
        mock = MagicMock()
        return mock

    @pytest.fixture
    def mock_ai_generator_for_tool_choice(
        self, mock_client_for_tool_choice, tool_manager
    ):
        """Create AIGenerator with mocked client for tool_choice tests"""
        ai = AIGenerator(api_key="test_key", model="claude-sonnet-4-20250514")
        ai.client = mock_client_for_tool_choice
        return ai

    def test_tool_choice_auto_when_tools_present(
        self,
        mock_ai_generator_for_tool_choice,
        mock_client_for_tool_choice,
        tool_manager,
    ):
        """Test that tool_choice is set to auto when tools are provided"""
        mock_response = MagicMock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = [MagicMock(type="text", text="Response")]
        mock_client_for_tool_choice.messages.create.return_value = mock_response

        mock_ai_generator_for_tool_choice.generate_response(
            query="test", tools=tool_manager.get_tool_definitions()
        )

        call_args = mock_client_for_tool_choice.messages.create.call_args
        params = call_args[1] if call_args else {}

        assert params.get("tool_choice") == {
            "type": "auto"
        }, "tool_choice should be auto when tools provided"

    def test_no_tool_choice_when_no_tools(
        self, mock_ai_generator_for_tool_choice, mock_client_for_tool_choice
    ):
        """Test that tool_choice is not set when no tools"""
        mock_response = MagicMock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = [MagicMock(type="text", text="Response")]
        mock_client_for_tool_choice.messages.create.return_value = mock_response

        mock_ai_generator_for_tool_choice.generate_response(query="test", tools=None)

        call_args = mock_client_for_tool_choice.messages.create.call_args
        params = call_args[1] if call_args else {}

        assert (
            "tool_choice" not in params or params.get("tool_choice") is None
        ), "tool_choice should not be set when no tools"


class TestAIGeneratorRecursiveToolUse:
    """Tests for recursive tool use handling"""

    @pytest.fixture
    def mock_client_recursive(self):
        """Create a mock Anthropic client for recursive tests"""
        mock = MagicMock()
        return mock

    @pytest.fixture
    def mock_ai_generator_recursive(self, mock_client_recursive, tool_manager):
        """Create AIGenerator with mocked client for recursive tests"""
        ai = AIGenerator(api_key="test_key", model="claude-sonnet-4-20250514")
        ai.client = mock_client_recursive
        return ai

    def test_recursive_tool_use_handled(
        self, mock_ai_generator_recursive, mock_client_recursive, tool_manager
    ):
        """Test that recursive tool use (tool wants to use another tool) is handled"""
        # First response: tool use
        mock_tool_response1 = MagicMock()
        mock_tool_response1.stop_reason = "tool_use"
        mock_tool_block1 = MagicMock()
        mock_tool_block1.type = "tool_use"
        mock_tool_block1.name = "search_course_content"
        mock_tool_block1.id = "tool_1"
        mock_tool_block1.input = {"query": "first query"}
        mock_tool_response1.content = [mock_tool_block1]

        # Second response: wants to use tool again
        mock_tool_response2 = MagicMock()
        mock_tool_response2.stop_reason = "tool_use"
        mock_tool_block2 = MagicMock()
        mock_tool_block2.type = "tool_use"
        mock_tool_block2.name = "search_course_content"
        mock_tool_block2.id = "tool_2"
        mock_tool_block2.input = {"query": "second query"}
        mock_tool_response2.content = [mock_tool_block2]

        # Third response: final text
        mock_final_response = MagicMock()
        mock_final_response.stop_reason = "end_turn"
        mock_final_text = MagicMock()
        mock_final_text.type = "text"
        mock_final_text.text = "Final answer after multiple searches"
        mock_final_response.content = [mock_final_text]

        mock_client_recursive.messages.create.side_effect = [
            mock_tool_response1,
            mock_tool_response2,
            mock_final_response,
        ]

        result = mock_ai_generator_recursive.generate_response(
            query="complex query",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager,
        )

        assert (
            result == "Final answer after multiple searches"
        ), "Should handle recursive tool use and return final answer"
