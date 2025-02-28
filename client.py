import asyncio
import json
import sys
from typing import Optional
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()  # load environment variables from .env

class MCPClient:
    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        # Replace Anthropic client with OpenAI
        self.openai = OpenAI()  # Make sure you have OPENAI_API_KEY in your env

    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server
        
        Args:
            server_script_path: Path to the server script (.py or .js)
        """
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")
            
        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None
        )
        
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
        
        await self.session.initialize()
        
        # List available tools
        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])

    async def process_query(self, query: str) -> str:
        """Process a query using OpenAI and available tools"""
        messages = [
            {
                "role": "user",
                "content": query
            }
        ]

        response = await self.session.list_tools()
        
        # Convert MCP tools to OpenAI tool format
        available_tools = [{ 
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.inputSchema
            }
        } for tool in response.tools]

        # Initial OpenAI API call
        response = self.openai.chat.completions.create(
            model="gpt-4o",  # Or another OpenAI model that supports tool calling
            messages=messages,
            tools=available_tools,
            tool_choice="auto"
        )

        # Process response and handle tool calls
        tool_results = []
        final_text = []
        assistant_response = response.choices[0].message
        
        # Add the initial response to final text
        if assistant_response.content:
            final_text.append(assistant_response.content)
        
        # Check if there are tool calls
        if hasattr(assistant_response, 'tool_calls') and assistant_response.tool_calls:
            # Add the assistant message to the conversation
            messages.append({
                "role": "assistant",
                "content": assistant_response.content,
                "tool_calls": assistant_response.model_dump().get('tool_calls', [])
            })
            
            # Process each tool call
            for tool_call in assistant_response.tool_calls:
                function_call = tool_call.function
                tool_name = function_call.name
                try:
                    tool_args = json.loads(function_call.arguments)
                except json.JSONDecodeError:
                    tool_args = {}  # Fallback if arguments aren't valid JSON
                
                try:
                    # Execute tool call using MCP
                    result = await self.session.call_tool(tool_name, tool_args)
                    tool_results.append({"call": tool_name, "result": result})
                    final_text.append(f"[Calling tool {tool_name} with args {tool_args}]")
                    
                    # Add the tool response to the conversation
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_name,
                        "content": result.content
                    })
                except Exception as e:
                    error_msg = f"Error executing tool {tool_name}: {str(e)}"
                    print(error_msg)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_name,
                        "content": f"Error: {str(e)}"
                    })
            
            # Get the next response from OpenAI with the tool results
            try:
                response = self.openai.chat.completions.create(
                    model="gpt-4o",
                    messages=messages
                )
                
                # Add the final response
                final_text.append(response.choices[0].message.content)
            except Exception as e:
                final_text.append(f"Error getting response after tool calls: {str(e)}")

        return "\n".join(final_text)

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")
        
        while True:
            try:
                query = input("\nQuery: ").strip()
                
                if query.lower() == 'quit':
                    break
                    
                response = await self.process_query(query)
                print("\n" + response)
                    
            except Exception as e:
                print(f"\nError: {str(e)}")
    
    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()

async def main():
    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_server_script>")
        sys.exit(1)
        
    client = MCPClient()
    try:
        await client.connect_to_server(sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())