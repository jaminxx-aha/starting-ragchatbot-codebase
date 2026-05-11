import anthropic
from typing import List, Optional, Dict, Any

class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""
    
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
        
        # Prepare API call parameters efficiently
        api_params = {
            **self.base_params,
            "messages": [{"role": "user", "content": query}],
            "system": system_content
        }
        
        # Add tools if available
        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}
        
        # Get response from Claude
        response = self.client.messages.create(**api_params)

        # Handle tool execution if needed
        if response.stop_reason == "tool_use" and tool_manager:
            return self._handle_tool_execution(response, api_params, tool_manager)

        # Return direct response - prioritize TextBlock over ThinkingBlock
        for block in response.content:
            if block.type == 'text' and hasattr(block, 'text'):
                return block.text

        # If only thinking block, return thinking content
        for block in response.content:
            if block.type == 'thinking' and hasattr(block, 'thinking'):
                return block.thinking

        # Last resort: return any text content
        for block in response.content:
            if hasattr(block, 'text'):
                return block.text

        return "No response generated."
    
    def _handle_tool_execution(self, initial_response, base_params: Dict[str, Any], tool_manager):
        """
        Handle execution of tool calls and get follow-up response.
        
        Args:
            initial_response: The response containing tool use requests
            base_params: Base API parameters
            tool_manager: Manager to execute tools
            
        Returns:
            Final response text after tool execution
        """
        # Start with existing messages
        messages = base_params["messages"].copy()
        
        # Add AI's tool use response
        messages.append({"role": "assistant", "content": initial_response.content})
        
        # Execute all tool calls and collect results
        tool_results = []
        for content_block in initial_response.content:
            if content_block.type == "tool_use":
                tool_result = tool_manager.execute_tool(
                    content_block.name, 
                    **content_block.input
                )
                
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": content_block.id,
                    "content": tool_result
                })
        
        # Add tool results as single message
        if tool_results:
            messages.append({"role": "user", "content": tool_results})
        
        # Prepare final API call without tools
        final_params = {
            **self.base_params,
            "messages": messages,
            "system": base_params["system"]
        }
        
        # Get final response - handle various block types
        final_response = self.client.messages.create(**final_params)

        # Try to find text block first
        for block in final_response.content:
            if block.type == 'text' and hasattr(block, 'text'):
                return block.text

        # If model wants to use tools again, execute them and get another response
        if final_response.stop_reason == "tool_use":
            return self._handle_tool_execution(final_response, final_params, tool_manager)

        # If only thinking block, extract thinking content as response
        for block in final_response.content:
            if block.type == 'thinking' and hasattr(block, 'thinking'):
                thinking_content = block.thinking
                if thinking_content:
                    return thinking_content

        # Last resort: return any text content
        for block in final_response.content:
            if hasattr(block, 'text'):
                return block.text

        return "No response generated."