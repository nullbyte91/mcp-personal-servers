#!/usr/bin/env python3
import asyncio
import re
import json
import os
import sys
from typing import Optional, Dict, Any, List, Union
from contextlib import AsyncExitStack
import random

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client
from mcp.types import AnyUrl

from backends.base import AIBackend
from backends.ollama_backend import OllamaBackend
from backends.claude_backend import ClaudeBackend

class FolderExplorerClient:
    def __init__(self, backend_type="ollama", backend_options=None):
        """
        Initialize the client with the specified backend
        
        Args:
            backend_type: "ollama" or "claude"
            backend_options: Dictionary of options for the backend
        """
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        
        # Default backend options
        if backend_options is None:
            backend_options = {}
            
        # Initialize the AI backend
        if backend_type.lower() == "ollama":
            model = backend_options.get("model", "deepseek-r1:7b")
            base_url = backend_options.get("url", "http://localhost:11434")
            self.ai_backend = OllamaBackend(base_url=base_url, model=model)
            self.backend_type = "ollama"
        elif backend_type.lower() == "claude":
            api_key = backend_options.get("api_key", os.getenv("ANTHROPIC_API_KEY"))
            model = backend_options.get("model", "claude-3-5-sonnet-20241022")
            self.ai_backend = ClaudeBackend(api_key=api_key, model=model)
            self.backend_type = "claude"
        else:
            raise ValueError(f"Unsupported backend type: {backend_type}")
            
        self.resources = {}
        self.tools = {}
        self.available_prompts = {}
    
    async def connect_to_server_stdio(self, server_script_path: str):
        """Connect to an MCP server by launching it as a subprocess"""
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
        await self._load_resources_and_tools()
        
        # Initialize the AI backend
        await self._initialize_backend()
    
    async def connect_to_server_sse(self, url: str = "http://localhost:8000/sse"):
        """Connect to an MCP server via SSE (for connecting to a separately running server)"""
        try:
            sse_transport = await self.exit_stack.enter_async_context(sse_client(url))
            self.stdio, self.write = sse_transport
            self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

            await self.session.initialize()
            await self._load_resources_and_tools()
            
            print(f"Successfully connected to server at {url}")
            
            # Initialize the AI backend
            await self._initialize_backend()
        except Exception as e:
            print(f"Error connecting to server: {e}")
            raise
    
    async def _initialize_backend(self):
        """Initialize the AI backend and check availability"""
        print(f"\nInitializing {self.backend_type} backend with model: {self.ai_backend.get_model_name()}")
        
        is_available = await self.ai_backend.initialize()
        if is_available:
            print(f"✅ {self.backend_type.capitalize()} is available with model: {self.ai_backend.get_model_name()}")
        else:
            print(f"⚠️ Warning: {self.backend_type.capitalize()} is not available or configuration is incorrect")
            if self.backend_type == "ollama":
                print("Please ensure Ollama is running and you have pulled the model:")
                print(f"  ollama pull {self.ai_backend.get_model_name()}")
            else:
                print("Please check your API key and model name")
            
    async def _load_resources_and_tools(self):
        """Load available resources and tools from the server"""
        if not self.session:
            raise ValueError("Not connected to a server")
            
        # List available resources
        try:
            resources_response = await self.session.list_resources()
            print("\nAvailable files:")
            
            # Store resources for later use
            for resource in resources_response.resources:
                self.resources[str(resource.uri)] = {
                    "name": resource.name,
                    "description": resource.description,
                    "uri": resource.uri,
                    "mime_type": resource.mimeType
                }
                print(f"- {resource.name}: {resource.description}")
                
            # Group files by extension for a summary
            extensions = {}
            for uri, resource in self.resources.items():
                ext = os.path.splitext(resource["name"])[1].lower()
                if ext not in extensions:
                    extensions[ext] = 0
                extensions[ext] += 1
                
            print("\nFile types summary:")
            for ext, count in extensions.items():
                ext_name = ext if ext else "(no extension)"
                print(f"- {ext_name}: {count} files")
                
        except Exception as e:
            print(f"Error listing resources: {e}")
        
        # List available tools
        try:
            tools_response = await self.session.list_tools()
            print("\nAvailable tools:")
            
            # Store tools for later use
            for tool in tools_response.tools:
                self.tools[tool.name] = {
                    "name": tool.name,
                    "description": tool.description,
                    "schema": tool.inputSchema
                }
                print(f"- {tool.name}: {tool.description}")
        except Exception as e:
            print(f"Error listing tools: {e}")

    async def call_tool(self, tool_name: str, args: Dict[str, Any]) -> str:
        """Call a tool on the server"""
        if not self.session:
            return "Not connected to a server. Please connect first."
            
        if tool_name not in self.tools:
            return f"Tool '{tool_name}' not found on the server."
            
        print(f"Calling server tool '{tool_name}' with args: {args}")
        
        try:
            result = await self.session.call_tool(tool_name, args)
            if result and hasattr(result, 'content') and result.content:
                content_text = []
                for content_item in result.content:
                    if hasattr(content_item, 'text'):
                        content_text.append(content_item.text)
                return "\n".join(content_text)
            return "Tool call didn't return any results"
        except Exception as e:
            return f"Error calling tool: {e}"

    async def read_resource(self, uri: str) -> str:
        """Read a resource from the server"""
        if not self.session:
            return "Not connected to a server. Please connect first."
            
        if uri not in self.resources:
            return f"Resource '{uri}' not found on the server."
            
        print(f"Reading resource: {uri}")
        
        try:
            resource_result = await self.session.read_resource(AnyUrl(uri))
            if resource_result and resource_result.contents:
                content = resource_result.contents[0]
                if hasattr(content, 'text'):
                    return content.text
                else:
                    return "Resource content is not text"
            return "Resource read didn't return any content"
        except Exception as e:
            return f"Error reading resource: {e}"

    async def switch_backend(self, backend_type, backend_options=None):
        """Switch to a different AI backend"""
        if backend_options is None:
            backend_options = {}
            
        # Initialize the new AI backend
        if backend_type.lower() == "ollama":
            model = backend_options.get("model", "deepseek-r1:7b")
            base_url = backend_options.get("url", "http://localhost:11434")
            self.ai_backend = OllamaBackend(base_url=base_url, model=model)
            self.backend_type = "ollama"
        elif backend_type.lower() == "claude":
            api_key = backend_options.get("api_key", os.getenv("ANTHROPIC_API_KEY"))
            model = backend_options.get("model", "claude-3-5-sonnet-20241022")
            self.ai_backend = ClaudeBackend(api_key=api_key, model=model)
            self.backend_type = "claude"
        else:
            raise ValueError(f"Unsupported backend type: {backend_type}")
            
        return await self.ai_backend.initialize()

    async def call_ai_backend(self, system_prompt, messages, max_tokens=2000, temperature=0):
        """Call the AI backend with retry logic"""
        # Ensure backend is initialized
        if not await self.ai_backend.initialize():
            raise Exception(f"{self.backend_type} backend could not be initialized")
        
        # Add retry logic directly here to handle all types of backends
        max_retries = 3
        retry_count = 0
        base_delay = 2
        
        while retry_count < max_retries:
            try:
                return await self.ai_backend.generate_response(
                    system_prompt=system_prompt,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
            except Exception as e:
                error_str = str(e)
                
                # Handle retryable errors
                if "529" in error_str or "429" in error_str or "overloaded" in error_str.lower() or "rate_limit" in error_str.lower():
                    retry_count += 1
                    if retry_count >= max_retries:
                        raise
                        
                    delay = base_delay * (2 ** (retry_count - 1))
                    delay = delay + (random.random() * delay * 0.1)  # Add jitter
                    print(f"API issue. Retrying in {delay:.2f} seconds... (Attempt {retry_count}/{max_retries})")
                    await asyncio.sleep(delay)
                else:
                    # Non-retryable error
                    raise

    async def process_query(self, query: str) -> str:
        """Process a query using the AI backend and available resources/tools"""
        if not self.session:
            return "Not connected to a server. Please connect first."
        
        # Define the system prompt
        system_prompt = """You are an assistant that helps analyze files and directories.
        You have access to tools that can search, analyze, and explore files in a directory.
        
        When the user asks questions about files, use the appropriate tools to help answer their questions.
        Always use tools rather than guessing about file contents.
        
        - For finding files, use the find_files tool.
        - For searching within a file, use the search_in_file tool.
        - For getting details about a file, use the get_file_info tool.
        - For comparing files, use the compare_files tool.
        - If files might have changed on disk, suggest using rescan_folder.
        
        Always explain what tools you're using and why, then present the results clearly."""
        
        # Prepare messages
        messages = [
            {
                "role": "user",
                "content": query
            }
        ]

        # Add context about available resources and tools
        context = "\n\nAvailable files:\n"
        # Limit to 10 files to avoid overwhelming context
        file_count = 0
        for uri, resource in list(self.resources.items())[:10]:
            context += f"- {resource['name']}: {resource['description']}\n  URI: {uri}\n"
            file_count += 1
        
        if file_count < len(self.resources):
            context += f"... and {len(self.resources) - file_count} more files.\n"
            
        if self.tools:
            context += "\nAvailable tools:\n"
            for name, tool in self.tools.items():
                context += f"- {name}: {tool['description']}\n"
        
        # Add context information to user query
        messages[0]["content"] = query + context
        
        # Call AI backend and get its response
        try:
            print(f"Processing query using {self.backend_type} ({self.ai_backend.get_model_name()})...")
            response = await self.call_ai_backend(system_prompt, messages)
            
            # Extract response text
            response_text = ""
            if self.backend_type == "claude":
                # Handle Claude response format
                for content in response.content:
                    if content.type == 'text':
                        response_text += content.text
            else:
                # Handle Ollama or other response formats that use get()
                for content in response.get("content", []):
                    if content.get("type") == 'text':
                        response_text += content.get("text", "")
            
            # Check if the response suggests using tools
            tool_calls = self._extract_suggested_tool_calls(response_text)
            
            # If tool calls are identified, execute them and update the response
            if tool_calls:
                tool_results = []
                for tool_call in tool_calls:
                    tool_name = tool_call["tool"]
                    args = tool_call["args"]
                    
                    print(f"\nExecuting suggested tool: {tool_name}")
                    result = await self.call_tool(tool_name, args)
                    tool_results.append(f"Results from {tool_name}:\n{result}")
                    
                # If we got tool results, we need to call the AI again with the results
                if tool_results:
                    # Add the tool results to the conversation
                    messages.append({
                        "role": "assistant",
                        "content": response_text
                    })
                    
                    messages.append({
                        "role": "user",
                        "content": f"Here are the results of the tool calls you suggested:\n\n" + 
                                "\n\n".join(tool_results) +
                                "\n\nPlease analyze these results and provide your final answer."
                    })
                    
                    # Call AI again with the tool results
                    print(f"Processing tool results using {self.backend_type}...")
                    final_response = await self.call_ai_backend(system_prompt, messages)
                    
                    # Extract final response text
                    final_text = ""
                    if self.backend_type == "claude":
                        # Handle Claude response format
                        for content in final_response.content:
                            if content.type == 'text':
                                final_text += content.text
                    else:
                        # Handle Ollama or other response formats
                        for content in final_response.get("content", []):
                            if content.get("type") == 'text':
                                final_text += content.get("text", "")
                            
                    return final_text
            
            # Check if we need to read any resources directly
            resource_uris = self._extract_resource_uris(response_text)
            
            if resource_uris:
                resource_contents = []
                for uri in resource_uris:
                    print(f"\nReading resource: {uri}")
                    content = await self.read_resource(uri)
                    # Truncate very long files
                    if len(content) > 5000:
                        content = content[:5000] + "...[content truncated]"
                    resource_contents.append(f"Content of {self.resources.get(uri, {}).get('name', uri)}:\n{content}")
                    
                # Add the resource contents to the conversation
                messages.append({
                    "role": "assistant",
                    "content": response_text
                })
                
                messages.append({
                    "role": "user",
                    "content": f"Here are the contents of the files you requested:\n\n" + 
                            "\n\n".join(resource_contents) +
                            "\n\nPlease analyze these files and provide your final answer."
                })
                
                # Call AI again with the resource contents
                print(f"Processing file contents using {self.backend_type}...")
                final_response = await self.call_ai_backend(system_prompt, messages)
                
                # Extract final response text
                final_text = ""
                if self.backend_type == "claude":
                    # Handle Claude response format
                    for content in final_response.content:
                        if content.type == 'text':
                            final_text += content.text
                else:
                    # Handle Ollama or other response formats
                    for content in final_response.get("content", []):
                        if content.get("type") == 'text':
                            final_text += content.get("text", "")
                    
                return final_text
            
            # If no tools or resources were used, return the original response
            return response_text
            
        except Exception as e:
            print(f"Error during AI API call: {str(e)}")
            # Return a fallback message when all retries fail
            return f"""I'm having trouble connecting to the AI model.

    You asked: "{query}"

    Please try again in a few minutes, or try using one of these simple commands:
    - "What files are in this folder?" - to list files
    - "Find files with [extension]" - to find files by type
    - "Search for [term]" - to search within files

    Error details: {type(e).__name__}: {str(e)}"""
    
    def _extract_suggested_tool_calls(self, response_text: str) -> List[Dict[str, Any]]:
        """Extract suggested tool calls from the AI's response"""
        tool_calls = []
        
        # Check for specific tool mentions
        for tool_name in self.tools.keys():
            # Check for explicit mentions of using the tool
            if f"use the {tool_name}" in response_text.lower() or f"using the {tool_name}" in response_text.lower():
                args = {}
                
                # Extract arguments based on the tool
                if tool_name == "get_file_info":
                    # Look for URIs
                    uri_match = re.search(r'file://([\w\d/\._+-]+)', response_text)
                    if uri_match:
                        full_uri = f"file://{uri_match.group(1)}"
                        if full_uri in self.resources:
                            args["file_uri"] = full_uri
                
                elif tool_name == "search_in_file":
                    # Look for URIs and search terms
                    uri_match = re.search(r'file://([\w\d/\._+-]+)', response_text)
                    search_term_match = re.search(r'search[^"\']*["\']([^"\']+)["\']', response_text, re.IGNORECASE)
                    
                    if uri_match and search_term_match:
                        full_uri = f"file://{uri_match.group(1)}"
                        if full_uri in self.resources:
                            args["file_uri"] = full_uri
                            args["search_term"] = search_term_match.group(1)
                
                elif tool_name == "find_files":
                    # Determine search type and term
                    if "by name" in response_text.lower():
                        search_type = "name"
                    elif "by extension" in response_text.lower():
                        search_type = "extension"
                    elif "by content" in response_text.lower():
                        search_type = "content"
                    else:
                        search_type = "name"  # Default
                    
                    search_term_match = re.search(r'search[^"\']*["\']([^"\']+)["\']', response_text, re.IGNORECASE)
                    if not search_term_match:
                        search_term_match = re.search(r'for ["\']?([^"\']+)["\']?', response_text, re.IGNORECASE)
                    
                    if search_term_match:
                        args["search_type"] = search_type
                        args["search_term"] = search_term_match.group(1)
                
                elif tool_name == "compare_files":
                    # Look for two URIs
                    uri_matches = re.findall(r'file://([\w\d/\._+-]+)', response_text)
                    if len(uri_matches) >= 2:
                        uri1 = f"file://{uri_matches[0]}"
                        uri2 = f"file://{uri_matches[1]}"
                        if uri1 in self.resources and uri2 in self.resources:
                            args["file_uri1"] = uri1
                            args["file_uri2"] = uri2
                
                elif tool_name == "rescan_folder":
                    # No arguments needed
                    pass
                
                # Add the tool call if we have valid arguments or it's a no-arg tool
                if args or tool_name == "rescan_folder":
                    tool_calls.append({
                        "tool": tool_name,
                        "args": args
                    })
        
        return tool_calls
    
    def _extract_resource_uris(self, response_text: str) -> List[str]:
        """Extract resource URIs from the AI's response that might need to be read"""
        uris = []
        
        # Look for references to reading files with more comprehensive patterns
        if "read the file" in response_text.lower() or "check the content" in response_text.lower() or \
        "looking at the file" in response_text.lower() or "examine the code" in response_text.lower() or \
        "analyze the file" in response_text.lower():
            uri_matches = re.findall(r'file://([\w\d/\._+-]+)', response_text)
            for match in uri_matches:
                full_uri = f"file://{match}"
                if full_uri in self.resources:
                    uris.append(full_uri)
        
        return uris
        
    async def chat_loop(self):
        """Run an interactive chat loop that supports both prompt-based and natural language approaches"""
        print(f"\n========= Folder Explorer Client with {self.backend_type.capitalize()} =========")
        print(f"Using {self.backend_type} with model: {self.ai_backend.get_model_name()}")
        print("Type your queries about files or use one of these commands:")
        print("  help               - Show this help message")
        print("  prompts            - List all available prompts")
        print("  tools              - List all available tools")
        print("  use <prompt_name>  - Start using a specific prompt")
        print("  model <model_name> - Switch to a different model")
        print("  backend <name>     - Switch AI backend (ollama/claude)")
        print("  quit               - Exit the application")
        print("=========================================================")

        # Store available prompts for quick access
        try:
            prompts = await self.list_prompts()
            for prompt in prompts:
                self.available_prompts[prompt["name"]] = prompt
        except Exception as e:
            print(f"Failed to load prompts initially: {str(e)}")
            print("You can try 'prompts' command later.")
        
        while True:
            try:
                query = input("\nQuery: ").strip()

                if query.lower() in ['quit', 'exit']:
                    break
                    
                # Handle help command
                if query.lower() == 'help':
                    print("\n========= Command Help =========")
                    print("  help               - Show this help message")
                    print("  prompts            - List all available prompts")
                    print("  tools              - List all available tools")
                    print("  use <prompt_name>  - Start using a specific prompt")
                    print("  model <model_name> - Switch to a different model")
                    print("  backend <name>     - Switch AI backend (ollama/claude)")
                    print("  quit               - Exit the application")
                    print("\nSample queries:")
                    print("  What files are in this folder?")
                    print("  Find all Python files")
                    print("  Search for 'example' in all text files")
                    print("  Compare file1.py and file2.py")
                    print("  Analyze the complexity of main.py")
                    print("\nAdvanced usage:")
                    print("  use <prompt_name> <arg_name>=<value>  - Use prompt with direct arguments")
                    print("  Example: use code-complexity-analysis file_uri=file:///path/to/file.py")
                    print("================================")
                    continue
                    
                # Handle tools listing command
                if query.lower() == 'tools':
                    print("\nAvailable tools:")
                    for name, tool in self.tools.items():
                        print(f"- {name}: {tool['description']}")
                    continue
                    
                # Handle prompt listing
                if query.lower() == 'prompts':
                    prompts = await self.list_prompts()
                    self.available_prompts = {p["name"]: p for p in prompts}
                    continue
                
                # Handle model switching
                if query.lower().startswith('model '):
                    new_model = query[6:].strip()
                    if not new_model:
                        print(f"Current model: {self.ai_backend.get_model_name()}")
                        continue
                        
                    print(f"Switching to model: {new_model}")
                    if await self.ai_backend.set_model(new_model):
                        print(f"✅ Successfully switched to model: {new_model}")
                    else:
                        print(f"⚠️ Warning: Model '{new_model}' not found or not available")
                        print(f"Using current model: {self.ai_backend.get_model_name()}")
                    continue
                
                # Handle backend switching
                if query.lower().startswith('backend '):
                    new_backend = query[8:].strip()
                    if new_backend.lower() not in ["ollama", "claude"]:
                        print(f"\nUnsupported backend: {new_backend}")
                        print("Available backends: ollama, claude")
                        continue
                        
                    # Get backend options
                    if new_backend.lower() == "ollama":
                        options = {"model": input("Enter Ollama model name [deepseek-r1:7b]: ") or "deepseek-r1:7b"}
                    else:  # claude
                        api_key = os.getenv("ANTHROPIC_API_KEY")
                        if not api_key:
                            api_key = input("Enter Anthropic API key: ")
                        options = {
                            "api_key": api_key,
                            "model": input("Enter Claude model [claude-3-5-sonnet-20241022]: ") or "claude-3-5-sonnet-20241022"
                        }
                        
                    print(f"Switching to {new_backend} backend...")
                    try:
                        if await self.switch_backend(new_backend, options):
                            print(f"✅ Successfully switched to {new_backend} backend with model: {self.ai_backend.get_model_name()}")
                        else:
                            print(f"⚠️ Failed to initialize {new_backend} backend")
                    except Exception as e:
                        print(f"Error switching backend: {str(e)}")
                    continue
                
                # Check for direct arguments in prompt usage
                # Format: use prompt_name arg1=value1 arg2=value2
                if query.lower().startswith('use '):
                    parts = query[4:].strip().split(' ')
                    prompt_name = parts[0]
                    
                    if prompt_name not in self.available_prompts:
                        print(f"\nPrompt '{prompt_name}' not found. Use 'prompts' to see available options.")
                        continue
                    
                    # Check if direct arguments are provided
                    args = {}
                    if len(parts) > 1:
                        # Parse arguments in format arg=value
                        for arg_part in parts[1:]:
                            if '=' in arg_part:
                                key, value = arg_part.split('=', 1)
                                args[key] = value
                    
                    # If we have all required arguments, execute directly
                    prompt = self.available_prompts[prompt_name]
                    required_args = [arg["name"] for arg in prompt.get("arguments", []) 
                                    if arg.get("required", False)]
                    
                    if all(arg in args for arg in required_args):
                        # All required args are provided, execute directly
                        print(f"\n=== Using prompt: {prompt_name} ===")
                        print(f"Description: {prompt['description']}")
                        print("Using provided arguments:")
                        for key, value in args.items():
                            print(f"- {key}: {value}")
                        
                        # Execute the prompt
                        print("\nExecuting prompt...")
                        try:
                            response = await self.use_prompt(prompt_name, args)
                            print("\n=== Prompt Result ===")
                            print(response)
                        except Exception as e:
                            error_str = str(e)
                            print(f"\nError executing prompt: {error_str}")
                        continue
                    
                    # Otherwise fall back to interactive mode
                    print(f"\n=== Using prompt: {prompt_name} ===")
                    print(f"Description: {prompt['description']}")
                    
                    # Collect arguments that weren't provided
                    for arg in prompt.get("arguments", []):
                        arg_name = arg["name"]
                        arg_desc = arg["description"]
                        arg_required = arg.get("required", False)
                        
                        # Skip arguments that were already provided
                        if arg_name in args:
                            print(f"Using provided {arg_name}: {args[arg_name]}")
                            continue
                        
                        suffix = " (required)" if arg_required else " (optional, press Enter to skip)"
                        while True:
                            arg_value = input(f"Enter {arg_name} - {arg_desc}{suffix}: ").strip()
                            
                            if not arg_value and arg_required:
                                print(f"  {arg_name} is required. Please provide a value.")
                            else:
                                break
                        
                        if arg_value:
                            args[arg_name] = arg_value
                    
                    # Execute the prompt
                    print("\nExecuting prompt...")
                    try:
                        response = await self.use_prompt(prompt_name, args)
                        print("\n=== Prompt Result ===")
                        print(response)
                    except Exception as e:
                        error_str = str(e)
                        print(f"\nError executing prompt: {error_str}")
                    continue

                # Process a regular query (natural language approach)
                try:
                    print(f"Processing query using {self.backend_type} ({self.ai_backend.get_model_name()})...")
                    response = await self.process_query(query)
                    print("\n" + response)
                except Exception as e:
                    error_str = str(e)
                    print(f"\nError processing query: {error_str}")
                    
                    # Try fallback for basic queries
                    try:
                        # Check if query is asking for specific file information
                        file_mention = re.search(r'(file|document|code)\s+([/\w\-\d\.]+)', query, re.IGNORECASE)
                        if file_mention:
                            filename = file_mention.group(2)
                            # Try to find the file
                            result = await self.call_tool("find_files", {"search_type": "name", "search_term": filename})
                            print(f"\nLooking for file '{filename}':\n{result}")
                    except Exception:
                        pass

            except KeyboardInterrupt:
                print("\nOperation cancelled. Type 'quit' to exit.")
            except Exception as e:
                print(f"\nError: {str(e)}")

    async def cleanup(self):
        """Clean up resources"""
        try:
            await self.exit_stack.aclose()
        except Exception as e:
            print(f"Error during cleanup: {str(e)}")

    async def list_prompts(self) -> List[Dict[str, Any]]:
        """List available prompts from the server"""
        if not self.session:
            print("Not connected to a server. Please connect first.")
            return []
            
        try:
            prompts_response = await self.session.list_prompts()
            prompts = []
            
            print("\nAvailable prompts:")
            for prompt in prompts_response.prompts:
                prompt_info = {
                    "name": prompt.name,
                    "description": prompt.description,
                    "arguments": []
                }
                
                if prompt.arguments:
                    for arg in prompt.arguments:
                        prompt_info["arguments"].append({
                            "name": arg.name,
                            "description": arg.description,
                            "required": arg.required
                        })
                
                prompts.append(prompt_info)
                
                # Print prompt information
                print(f"- {prompt.name}: {prompt.description}")
                if prompt.arguments:
                    for arg in prompt.arguments:
                        req = "(required)" if arg.required else "(optional)"
                        print(f"  - {arg.name}: {arg.description} {req}")
            
            # Add the code-complexity-analysis prompt if it doesn't exist
            if not any(p["name"] == "code-complexity-analysis" for p in prompts):
                code_complexity_prompt = {
                    "name": "code-complexity-analysis",
                    "description": "Analyze code complexity using various metrics appropriate for the programming language",
                    "arguments": [
                        {
                            "name": "file_uri",
                            "description": "URI of the file to analyze",
                            "required": True
                        }
                    ]
                }
                prompts.append(code_complexity_prompt)
                
                # Print the additional prompt
                print(f"- {code_complexity_prompt['name']}: {code_complexity_prompt['description']}")
                for arg in code_complexity_prompt["arguments"]:
                    req = "(required)" if arg.get("required") else "(optional)"
                    print(f"  - {arg['name']}: {arg['description']} {req}")
                
            return prompts
        except Exception as e:
            print(f"Error listing prompts: {e}")
            return []

    async def get_prompt(self, prompt_name: str, arguments: Dict[str, str] = None) -> Dict[str, Any]:
        """Get a specific prompt with the provided arguments"""
        if not self.session:
            print("Not connected to a server. Please connect first.")
            return {"error": "Not connected to a server"}
            
        try:
            prompt_result = await self.session.get_prompt(prompt_name, arguments)
            return {
                "messages": [
                    {
                        "role": msg.role,
                        "content": msg.content.text if hasattr(msg.content, 'text') else str(msg.content)
                    }
                    for msg in prompt_result.messages
                ]
            }
        except Exception as e:
            print(f"Error getting prompt {prompt_name}: {e}")
            return {"error": f"Error: {str(e)}"}

    async def use_prompt(self, prompt_name: str, arguments: Dict[str, str] = None) -> str:
        """Use a prompt and get the AI's response to it"""
        if not self.session:
            return "Not connected to a server. Please connect first."
            
        try:
            # Special handling for code-complexity-analysis if it's not on the server
            if prompt_name == "code-complexity-analysis":
                return await self._handle_code_complexity_analysis(arguments)
                
            # Get the prompt from the server
            print(f"Getting prompt: {prompt_name}")
            prompt_result = await self.get_prompt(prompt_name, arguments)
            
            if "error" in prompt_result:
                return f"Error getting prompt: {prompt_result['error']}"
                
            # Extract messages to send to the AI
            messages = prompt_result.get("messages", [])
            if not messages:
                return "The prompt didn't generate any messages"
                
            # Define system prompt
            system_prompt = """You are an assistant that helps analyze files and source code.
            You provide thorough analysis of code, documentation, and other files.
            
            For code:
            - Explain the purpose and structure
            - Identify key components and algorithms
            - Note potential bugs or improvements
            - Suggest better practices where appropriate
            
            For documentation:
            - Summarize key points
            - Extract main concepts and relationships
            - Identify missing or unclear information
            
            Be specific, technical, and helpful in your analysis."""
            
            formatted_messages = []
            for msg in messages:
                formatted_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
                
            # Call the AI with the prompt's messages
            print(f"Calling {self.backend_type} with prompt...")
            response = await self.call_ai_backend(system_prompt, formatted_messages)
            
            # Extract the AI's response
            response_text = ""
            if self.backend_type == "claude":
                # Handle Claude response format
                for content in response.content:
                    if content.type == 'text':
                        response_text += content.text
            else:
                # Handle Ollama or other response formats
                for content in response.get("content", []):
                    if content.get("type") == 'text':
                        response_text += content.get("text", "")
                    
            return response_text
        except Exception as e:
            print(f"Error using prompt: {e}")
            return f"Error: {str(e)}"
            
    async def _handle_code_complexity_analysis(self, arguments: Dict[str, str] = None) -> str:
        """Custom handler for code complexity analysis prompt"""
        if not arguments or "file_uri" not in arguments:
            return "Error: file_uri is required for code complexity analysis"
            
        file_uri = arguments["file_uri"]
        
        # Validate the file URI
        if not file_uri.startswith("file://"):
            print(f"Invalid file URI format: {file_uri}")
            return "Error: File URI must start with 'file://'"
        
        # Check if the file exists in resources
        if file_uri not in self.resources:
            print(f"File not found in resources: {file_uri}")
            all_files = "\n".join([f"- {uri}" for uri in list(self.resources.keys())[:10]])
            return f"Error: File not found: {file_uri}\n\nAvailable files (first 10):\n{all_files}"
        
        print(f"Analyzing complexity of file: {file_uri}")
        
        # Get file info
        print("Getting file info...")
        try:
            file_info_result = await self.call_tool("get_file_info", {"file_uri": file_uri})
            print("✓ File info retrieved")
        except Exception as e:
            print(f"Error getting file info: {str(e)}")
            return f"Error retrieving file information: {str(e)}"
        
        # Read the file content
        print("Reading file content...")
        try:
            file_content = await self.read_resource(file_uri)
            print("✓ File content retrieved")
        except Exception as e:
            print(f"Error reading file: {str(e)}")
            return f"Error reading file content: {str(e)}"
        
        # Truncate if too large
        original_length = len(file_content)
        if len(file_content) > 5000:
            file_content = file_content[:5000]
            print(f"Note: File content truncated from {original_length} to 5000 characters for analysis")
        
        # Extract file extension to determine language
        file_name = self.resources[file_uri]["name"]
        _, ext = os.path.splitext(file_name)
        language_map = {
            '.py': 'Python',
            '.js': 'JavaScript',
            '.ts': 'TypeScript',
            '.java': 'Java',
            '.c': 'C',
            '.cpp': 'C++',
            '.go': 'Go',
            '.rs': 'Rust'
        }
        language = language_map.get(ext.lower(), "code")
        print(f"Detected language: {language}")
        
        # Create metrics prompt based on language
        metrics_prompt = ""
        if language:
            metrics_prompt = f"\n\nPlease calculate and report these complexity metrics for this {language} code:\n"
            if language == "Python":
                metrics_prompt += (
                    "1. Cyclomatic complexity for each function (number of decision points + 1)\n"
                    "2. Nesting depth in conditional statements and loops\n"
                    "3. Number of imported modules\n"
                    "4. Length of functions (in lines)\n"
                    "5. Cognitive complexity (estimation of how difficult the code is to understand)"
                )
            elif language in ["JavaScript", "TypeScript"]:
                metrics_prompt += (
                    "1. Cyclomatic complexity for each function\n"
                    "2. Nesting depth in conditional statements and loops\n"
                    "3. Number of dependencies\n"
                    "4. Function length and parameter count\n"
                    "5. Cognitive complexity estimation"
                )
            else:
                metrics_prompt += (
                    "1. Function/method complexity\n"
                    "2. Nesting depth\n"
                    "3. Dependencies/imports\n"
                    "4. Function length\n"
                    "5. Overall readability assessment"
                )
        
        # Create message to send to the AI
        system_prompt = """You are an assistant specialized in analyzing code complexity.
        You provide detailed metrics and insights about code quality and complexity.
        Focus on identifying complex areas, potential issues, and suggesting improvements.
        Be sure to back up your analysis with specific line numbers and code examples."""
        
        print("Creating analysis prompt...")
        file_info_section = ""
        if isinstance(file_info_result, str):
            file_info_section = f"File info:\n{file_info_result}\n\n"
        
        message = {
            "role": "user",
            "content": f"Please analyze the complexity of this code file: {file_name}\n\n"
                    f"{file_info_section}"
                    f"Code content ({original_length} bytes, {'truncated ' if original_length > 5000 else ''}showing first {len(file_content)} characters):\n```{ext[1:] if ext else ''}\n{file_content}\n```"
                    f"{metrics_prompt}\n\n"
                    f"After analyzing the metrics, please suggest specific ways to reduce the complexity and improve maintainability."
        }
        
        # Call the AI with the message
        print(f"Calling {self.backend_type} for complexity analysis...")
        try:
            response = await self.call_ai_backend(system_prompt, [message])
            print("✓ Analysis completed successfully")
        except Exception as e:
            print(f"Error during analysis: {str(e)}")
            return f"Error during complexity analysis: {str(e)}"
        
        # Extract the AI's response
        response_text = ""
        if self.backend_type == "claude":
            # Handle Claude response format
            for content in response.content:
                if content.type == 'text':
                    response_text += content.text
        else:
            # Handle Ollama or other response formats
            for content in response.get("content", []):
                if content.get("type") == 'text':
                    response_text += content.get("text", "")
        
        print("Analysis complete. Displaying results...")
        return response_text