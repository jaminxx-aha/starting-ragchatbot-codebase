import anthropic
from typing import List, Optional, Dict, Any

class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""

    # Maximum sequential tool calling rounds
    MAX_TOOL_ROUNDS = 2

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to tools for course information.

Available Tools:
1. **search_course_content** - For searching specific content within course materials
   - Use for questions about specific course content or detailed educational materials
   - Parameters: query (required), course_name (optional), lesson_number (optional)
   - One search per query maximum

2. **get_course_outline** - For retrieving course structure and lesson lists
   - Use for questions about course outline, structure, or lesson list
   - Returns: course title, course link, and complete lesson list (lesson number + title for each)
   - Parameters: course_title (required, partial matches supported)

Response Protocol:
- **Multi-step queries**: You may make up to 2 sequential tool calls to gather information
  - Example: First use get_course_outline to find lesson titles, then search_course_content with that title
  - Each tool call round lets you reason about previous results before deciding next steps
- **Course outline queries**: Use get_course_outline tool, then format the response with course title, course link, and lesson list
- **Content queries**: Use search_course_content tool first, then answer based on results
- **General knowledge questions**: Answer using existing knowledge without searching
- **No meta-commentary**:
  - Provide direct answers only — no reasoning process, search explanations, or question-type analysis
  - Do not mention "based on the search results"


All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""
    
    def __init__(self, api_key: str, model: str, base_url: str = ""):
        if base_url:
            self.client = anthropic.Anthropic(api_key=api_key, base_url=base_url)
        else:
            self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        
        # Pre-build base API parameters
        self.base_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800
        }
    
    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:
        """
        Generate AI response with optional tool usage and conversation context.

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools

        Returns:
            Generated response as string
        """

        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        # Initialize messages array
        messages = [{"role": "user", "content": query}]

        # Prepare API call parameters - tools stay available for all rounds
        api_params = {
            **self.base_params,
            "messages": messages,
            "system": system_content
        }

        # Add tools if available - these remain for sequential calls
        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}

        # Handle tool calling via agentic loop if tools and manager provided
        if tools and tool_manager:
            return self._agentic_loop(messages, api_params, tool_manager)

        # Direct API call without tools
        response = self.client.messages.create(**api_params)
        return self._extract_text(response) or "No response generated."
    
    def _agentic_loop(self, messages: List[Dict], api_params: Dict[str, Any],
                       tool_manager, max_rounds: int = MAX_TOOL_ROUNDS) -> str:
        """
        Execute agentic loop for sequential tool calling.

        Args:
            messages: Conversation messages array
            api_params: API parameters including tools
            tool_manager: Manager to execute tools
            max_rounds: Maximum sequential tool calling rounds

        Returns:
            Final response text after all tool rounds
        """
        for round_num in range(max_rounds):
            # Make API call with tools available
            response = self.client.messages.create(**api_params)

            # Check for normal completion - Claude is done
            if response.stop_reason == "end_turn":
                return self._extract_text(response)

            # Handle tool use
            if response.stop_reason == "tool_use":
                # Add assistant's response to messages
                messages.append({"role": "assistant", "content": response.content})

                # Execute all tool calls and collect results
                tool_results = self._execute_tools(response.content, tool_manager)

                # Add tool results as user message
                messages.append({"role": "user", "content": tool_results})

                # Update messages in params for next round
                api_params["messages"] = messages

                # Continue loop for next round
                continue

            # Handle max_tokens or other stop reasons
            if response.stop_reason == "max_tokens":
                # Return whatever text we have
                text = self._extract_text(response)
                if text:
                    return text
                return "Response was truncated due to token limit."

            # Unknown stop reason - try to extract text
            text = self._extract_text(response)
            if text:
                return text
            break

        # Max rounds reached - make one final call without tools to get response
        final_params = {
            **self.base_params,
            "messages": messages,
            "system": api_params["system"]
        }
        final_response = self.client.messages.create(**final_params)
        return self._extract_text(final_response) or "No response generated."

    def _execute_tools(self, content_blocks: List, tool_manager) -> List[Dict]:
        """
        Execute all tool calls from response content blocks.

        Args:
            content_blocks: Response content blocks from Claude
            tool_manager: Manager to execute tools

        Returns:
            List of tool_result blocks
        """
        tool_results = []
        for block in content_blocks:
            if block.type == "tool_use":
                try:
                    result = tool_manager.execute_tool(block.name, **block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result
                    })
                except Exception as e:
                    # Error handling with is_error flag
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": f"Error executing tool: {str(e)}",
                        "is_error": True
                    })
        return tool_results

    def _extract_text(self, response) -> str:
        """
        Extract text from Claude response, handling various block types.

        Args:
            response: Claude API response

        Returns:
            Extracted text or empty string
        """
        # Prioritize text block
        for block in response.content:
            if block.type == 'text' and hasattr(block, 'text'):
                return block.text

        # Fall back to thinking block if no text
        for block in response.content:
            if block.type == 'thinking' and hasattr(block, 'thinking'):
                return block.thinking

        # Last resort: any text attribute
        for block in response.content:
            if hasattr(block, 'text'):
                return block.text

        return ""