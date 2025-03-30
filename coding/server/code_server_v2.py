import asyncio
import os
import sys
import glob
from pathlib import Path
from typing import List, Dict, Any, Union, Literal, Optional
import mcp.server.stdio
import mcp.types as types
from mcp.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions
from pydantic import AnyUrl, FileUrl
import uvicorn
from starlette.applications import Starlette
from starlette.routing import Mount, Route
from mcp.server.sse import SseServerTransport

import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Then use logger instead of print
logger.debug("Server will provide access to files in: %s", "./output/")

# Initialize the server
server = Server("folder-explorer")

# Configuration
DEFAULT_FOLDER = os.path.abspath(os.path.expanduser("~/documents"))  # Default folder to explore
MAX_FILE_SIZE = 1024 * 1024  # 1MB limit for file size

# Define available prompts
FILE_PROMPTS = {
    "summarize-file": types.Prompt(
        name="summarize-file",
        description="Generate a summary of a file's contents",
        arguments=[
            types.PromptArgument(
                name="file_uri",
                description="URI of the file to summarize",
                required=True
            )
        ],
    ),
    "find-code-pattern": types.Prompt(
        name="find-code-pattern",
        description="Find usage patterns in code files",
        arguments=[
            types.PromptArgument(
                name="pattern",
                description="Description of the pattern to find (e.g., 'error handling', 'API calls')",
                required=True
            ),
            types.PromptArgument(
                name="file_extension",
                description="File extension to search (e.g., '.py', '.js')",
                required=False
            )
        ],
    ),
    "git-commit-message": types.Prompt(
        name="git-commit-message",
        description="Generate a commit message based on file changes",
        arguments=[
            types.PromptArgument(
                name="file_uris",
                description="Comma-separated list of file URIs that were changed",
                required=True
            ),
            types.PromptArgument(
                name="change_description",
                description="Brief description of the changes made",
                required=False
            )
        ],
    ),
    "explain-code": types.Prompt(
        name="explain-code",
        description="Explain how a piece of code works",
        arguments=[
            types.PromptArgument(
                name="file_uri",
                description="URI of the code file to explain",
                required=True
            ),
            types.PromptArgument(
                name="start_line",
                description="Starting line number (optional)",
                required=False
            ),
            types.PromptArgument(
                name="end_line",
                description="Ending line number (optional)",
                required=False
            )
        ],
    ),
    "code-review": types.Prompt(
        name="code-review",
        description="Perform a code review on a file",
        arguments=[
            types.PromptArgument(
                name="file_uri",
                description="URI of the file to review",
                required=True
            ),
            types.PromptArgument(
                name="focus",
                description="Focus of the review (e.g., 'security', 'performance', 'readability')",
                required=False
            )
        ],
    )
}

# Initialize folder to scan from command line argument if provided
target_folder = DEFAULT_FOLDER
if len(sys.argv) > 1:
    target_folder = os.path.abspath(os.path.expanduser(sys.argv[1]))
    if not os.path.isdir(target_folder):
        print(f"Error: {target_folder} is not a valid directory")
        sys.exit(1)

print(f"Server will provide access to files in: {target_folder}")

# Store found files
files_by_uri = {}
file_extensions = {}  # Group files by extension

def scan_folder(folder_path: str) -> Dict[str, Dict[str, Any]]:
    """Scan folder and return a dictionary of files with their metadata"""
    result = {}
    
    for file_path in glob.glob(os.path.join(folder_path, "**"), recursive=True):
        if os.path.isfile(file_path):
            try:
                rel_path = os.path.relpath(file_path, folder_path)
                file_size = os.path.getsize(file_path)
                _, file_ext = os.path.splitext(file_path)
                
                # Skip files that are too large
                if file_size > MAX_FILE_SIZE:
                    continue
                    
                # Create URI for this file
                file_uri = f"file://{file_path}"
                
                result[file_uri] = {
                    "path": file_path,
                    "relative_path": rel_path,
                    "size": file_size,
                    "extension": file_ext.lower(),
                    "name": os.path.basename(file_path)
                }
                
                # Group by extension
                if file_ext.lower() not in file_extensions:
                    file_extensions[file_ext.lower()] = []
                file_extensions[file_ext.lower()].append(file_uri)
                
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
    
    return result

# Scan the folder when server starts
files_by_uri = scan_folder(target_folder)
print(f"Found {len(files_by_uri)} files")
for ext, count in [(ext, len(files)) for ext, files in file_extensions.items()]:
    print(f"  {ext or '(no extension)'}: {count} files")

def get_mime_type(file_path: str) -> str:
    """Determine MIME type based on file extension"""
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    
    mime_mapping = {
        '.txt': 'text/plain',
        '.md': 'text/markdown',
        '.html': 'text/html',
        '.htm': 'text/html',
        '.css': 'text/css',
        '.csv': 'text/csv',
        '.json': 'application/json',
        '.js': 'application/javascript',
        '.py': 'text/x-python',
        '.java': 'text/x-java',
        '.c': 'text/x-c',
        '.cpp': 'text/x-c++',
        '.h': 'text/x-c',
        '.pdf': 'application/pdf',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.svg': 'image/svg+xml',
        '.xml': 'application/xml',
        '.zip': 'application/zip',
        '.doc': 'application/msword',
        '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        '.xls': 'application/vnd.ms-excel',
        '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        '.ppt': 'application/vnd.ms-powerpoint',
        '.pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
    }
    
    return mime_mapping.get(ext, 'application/octet-stream')

def is_text_file(file_path: str) -> bool:
    """Check if a file is a text file based on extension"""
    text_extensions = [
        '.txt', '.md', '.py', '.js', '.html', '.htm', '.css', '.csv', '.json',
        '.xml', '.java', '.c', '.cpp', '.h', '.sh', '.bat', '.ps1', '.yaml', '.yml',
        '.ini', '.cfg', '.conf', '.log', '.sql', '.r', '.go', '.rs', '.ts', '.tsx'
    ]
    
    _, ext = os.path.splitext(file_path)
    return ext.lower() in text_extensions

def read_file_content(file_path: str) -> Union[str, bytes]:
    """Read file content as string or bytes depending on the file type"""
    if is_text_file(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            # Fall back to binary if UTF-8 decoding fails
            with open(file_path, 'rb') as f:
                return f.read()
    else:
        with open(file_path, 'rb') as f:
            return f.read()

@server.list_resources()
async def handle_list_resources() -> List[types.Resource]:
    """List all files as resources"""
    resources = []
    
    for uri, file_info in files_by_uri.items():
        resources.append(
            types.Resource(
                uri=uri,
                name=file_info["name"],
                description=f"Path: {file_info['relative_path']} (Size: {file_info['size']} bytes)",
                mimeType=get_mime_type(file_info["path"])
            )
        )
    
    return resources

@server.read_resource()
async def handle_read_resource(uri: AnyUrl) -> Union[str, bytes]:
    """Handle reading a specific file resource"""
    uri_str = str(uri)
    
    if uri_str in files_by_uri:
        file_path = files_by_uri[uri_str]["path"]
        print(f"SERVER: Reading file {file_path}")
        return read_file_content(file_path)
    
    # If we get here, the resource wasn't found
    raise ValueError(f"Resource not found: {uri_str}")

# Define tools for file analysis
def get_file_info(file_path: str) -> Dict[str, Any]:
    """Get detailed information about a file"""
    if not os.path.exists(file_path):
        return {"error": f"File {file_path} does not exist"}
        
    try:
        stat_info = os.stat(file_path)
        return {
            "name": os.path.basename(file_path),
            "path": file_path,
            "size": stat_info.st_size,
            "created": stat_info.st_ctime,
            "modified": stat_info.st_mtime,
            "is_text": is_text_file(file_path),
            "mime_type": get_mime_type(file_path),
        }
    except Exception as e:
        return {"error": str(e)}

def search_in_file(file_path: str, search_term: str) -> Dict[str, Any]:
    """Search for a term in a text file and return matching lines with context"""
    if not os.path.exists(file_path):
        return {"error": f"File {file_path} does not exist"}
        
    if not is_text_file(file_path):
        return {"error": f"File {file_path} is not a text file"}
    
    try:
        result = {"matches": [], "file": os.path.basename(file_path)}
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        for i, line in enumerate(lines):
            if search_term.lower() in line.lower():
                # Get context (2 lines before and after)
                context_start = max(0, i - 2)
                context_end = min(len(lines), i + 3)
                
                result["matches"].append({
                    "line_number": i + 1,
                    "line": line.strip(),
                    "context": "".join(lines[context_start:context_end]).strip()
                })
                
        return result
    except Exception as e:
        return {"error": str(e)}

def find_files(search_type: str, search_term: str) -> Dict[str, Any]:
    """Find files by name, extension, or content"""
    results = {"matches": []}
    
    if search_type == "name":
        # Search by filename
        for uri, file_info in files_by_uri.items():
            if search_term.lower() in file_info["name"].lower():
                results["matches"].append({
                    "uri": uri,
                    "name": file_info["name"],
                    "path": file_info["relative_path"]
                })
    
    elif search_type == "extension":
        # Search by extension
        search_term = search_term.lower()
        if not search_term.startswith('.'):
            search_term = '.' + search_term
            
        if search_term in file_extensions:
            for uri in file_extensions[search_term]:
                file_info = files_by_uri[uri]
                results["matches"].append({
                    "uri": uri,
                    "name": file_info["name"],
                    "path": file_info["relative_path"]
                })
    
    elif search_type == "content":
        # Search by content (only in text files)
        for uri, file_info in files_by_uri.items():
            if is_text_file(file_info["path"]):
                try:
                    with open(file_info["path"], 'r', encoding='utf-8') as f:
                        content = f.read()
                        if search_term.lower() in content.lower():
                            results["matches"].append({
                                "uri": uri,
                                "name": file_info["name"],
                                "path": file_info["relative_path"]
                            })
                except Exception:
                    # Skip files that can't be read
                    pass
    
    return results

def compare_files(file_path1: str, file_path2: str) -> Dict[str, Any]:
    """Compare two text files and show differences"""
    if not os.path.exists(file_path1):
        return {"error": f"File {file_path1} does not exist"}
    if not os.path.exists(file_path2):
        return {"error": f"File {file_path2} does not exist"}
        
    if not is_text_file(file_path1) or not is_text_file(file_path2):
        return {"error": "Both files must be text files"}
    
    try:
        import difflib
        
        with open(file_path1, 'r', encoding='utf-8') as f1:
            lines1 = f1.readlines()
        
        with open(file_path2, 'r', encoding='utf-8') as f2:
            lines2 = f2.readlines()
            
        diff = difflib.unified_diff(
            lines1, lines2,
            fromfile=os.path.basename(file_path1),
            tofile=os.path.basename(file_path2)
        )
        
        diff_text = ''.join(diff)
        if not diff_text:
            return {"result": "Files are identical"}
        
        return {
            "result": "Files differ",
            "diff": diff_text,
            "file1": os.path.basename(file_path1),
            "file2": os.path.basename(file_path2)
        }
    except Exception as e:
        return {"error": str(e)}

def find_file_by_uri(uri: str) -> Optional[str]:
    """Find file path by URI"""
    if uri in files_by_uri:
        return files_by_uri[uri]["path"]
    return None

@server.list_tools()
async def handle_list_tools() -> List[types.Tool]:
    """List all available tools"""
    tools = [
        types.Tool(
            name="get_file_info",
            description="Get detailed information about a file by URI",
            inputSchema={
                "type": "object",
                "required": ["file_uri"],
                "properties": {
                    "file_uri": {"type": "string", "description": "URI of the file to get information about"}
                }
            }
        ),
        types.Tool(
            name="search_in_file",
            description="Search for a term in a text file and return matching lines with context",
            inputSchema={
                "type": "object",
                "required": ["file_uri", "search_term"],
                "properties": {
                    "file_uri": {"type": "string", "description": "URI of the file to search in"},
                    "search_term": {"type": "string", "description": "Term to search for in the file"}
                }
            }
        ),
        types.Tool(
            name="find_files",
            description="Find files by name, extension, or content",
            inputSchema={
                "type": "object",
                "required": ["search_type", "search_term"],
                "properties": {
                    "search_type": {
                        "type": "string", 
                        "enum": ["name", "extension", "content"],
                        "description": "Type of search to perform: by name, extension, or content"
                    },
                    "search_term": {"type": "string", "description": "Term to search for"}
                }
            }
        ),
        types.Tool(
            name="compare_files",
            description="Compare two text files and show differences",
            inputSchema={
                "type": "object",
                "required": ["file_uri1", "file_uri2"],
                "properties": {
                    "file_uri1": {"type": "string", "description": "URI of the first file to compare"},
                    "file_uri2": {"type": "string", "description": "URI of the second file to compare"}
                }
            }
        ),
        types.Tool(
            name="rescan_folder",
            description="Rescan the folder to update the list of available files",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        )
    ]
    return tools

@server.call_tool()
async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> List[types.TextContent]:
    """Handle tool calls"""
    print(f"SERVER: Tool call received: {name} with args {arguments}")
    
    result: Any = None
    
    if name == "get_file_info":
        file_uri = arguments["file_uri"]
        file_path = find_file_by_uri(file_uri)
        if file_path:
            result = get_file_info(file_path)
        else:
            result = {"error": f"File not found: {file_uri}"}
    
    elif name == "search_in_file":
        file_uri = arguments["file_uri"]
        search_term = arguments["search_term"]
        file_path = find_file_by_uri(file_uri)
        if file_path:
            result = search_in_file(file_path, search_term)
        else:
            result = {"error": f"File not found: {file_uri}"}
    
    elif name == "find_files":
        search_type = arguments["search_type"]
        search_term = arguments["search_term"]
        result = find_files(search_type, search_term)
    
    elif name == "compare_files":
        file_uri1 = arguments["file_uri1"]
        file_uri2 = arguments["file_uri2"]
        file_path1 = find_file_by_uri(file_uri1)
        file_path2 = find_file_by_uri(file_uri2)
        
        if not file_path1:
            result = {"error": f"File not found: {file_uri1}"}
        elif not file_path2:
            result = {"error": f"File not found: {file_uri2}"}
        else:
            result = compare_files(file_path1, file_path2)
    
    elif name == "rescan_folder":
        global files_by_uri, file_extensions
        file_extensions = {}
        files_by_uri = scan_folder(target_folder)
        result = {
            "result": "Folder rescanned",
            "total_files": len(files_by_uri),
            "extensions": {ext: len(files) for ext, files in file_extensions.items()}
        }
    
    else:
        return [types.TextContent(type="text", text=f"Unknown tool: {name}")]
    
    # Format result as string
    if isinstance(result, dict):
        import json
        result_str = json.dumps(result, indent=2)
    else:
        result_str = str(result)
    
    return [types.TextContent(type="text", text=result_str)]

@server.list_prompts()
async def handle_list_prompts() -> list[types.Prompt]:
    """List all available prompts"""
    return list(FILE_PROMPTS.values())


@server.get_prompt()
async def handle_get_prompt(name: str, arguments: dict[str, str] | None = None) -> types.GetPromptResult:
    """Handle retrieving a specific prompt with arguments and automatically call relevant tools"""
    if name not in FILE_PROMPTS:
        raise ValueError(f"Prompt not found: {name}")
    
    if arguments is None:
        arguments = {}
    
    # Create a helper function to call tools and format results
    async def call_tool_and_format(tool_name: str, tool_args: dict) -> str:
        """Call a tool and format the results as text"""
        try:
            tool_result = await handle_call_tool(tool_name, tool_args)
            if tool_result:
                content_text = []
                for content_item in tool_result:
                    if hasattr(content_item, 'text'):
                        content_text.append(content_item.text)
                return "\n".join(content_text)
            return "Tool call didn't return any results"
        except Exception as e:
            return f"Error calling tool {tool_name}: {str(e)}"
    
    if name == "find-files":
        # This prompt is a direct wrapper around the find_files tool
        search_type = arguments.get("search_type", "name")
        search_term = arguments.get("search_term", "")
        
        # Call the find_files tool directly
        tool_result = await call_tool_and_format("find_files", {
            "search_type": search_type,
            "search_term": search_term
        })
        
        return types.GetPromptResult(
            messages=[
                types.PromptMessage(
                    role="user",
                    content=types.TextContent(
                        type="text",
                        text=f"I used the find_files tool to search for files with {search_type}='{search_term}'. Here are the results:\n\n{tool_result}\n\n"
                             f"Please analyze these results and provide insights about the files found."
                    )
                )
            ]
        )
    
    elif name == "search-in-files":
        # This prompt searches for a term across multiple files
        search_term = arguments.get("search_term", "")
        file_extension = arguments.get("file_extension", "")
        
        # First find files with the given extension
        matching_files = []
        if file_extension:
            files_result = await call_tool_and_format("find_files", {
                "search_type": "extension",
                "search_term": file_extension
            })
            
            # Parse the result to extract file URIs
            import json
            try:
                result_json = json.loads(files_result)
                if "matches" in result_json:
                    matching_files = [match.get("uri") for match in result_json.get("matches", [])]
            except:
                # Fallback if JSON parsing fails
                import re
                uri_matches = re.findall(r'"uri":\s*"([^"]+)"', files_result)
                matching_files = uri_matches
        
        # Search in each file
        search_results = []
        for uri in matching_files[:5]:  # Limit to first 5 files to avoid overwhelming
            file_path = find_file_by_uri(uri)
            if file_path and is_text_file(file_path):
                search_result = await call_tool_and_format("search_in_file", {
                    "file_uri": uri,
                    "search_term": search_term
                })
                search_results.append(f"Results for {os.path.basename(file_path)}:\n{search_result}")
        
        all_results = "\n\n".join(search_results) if search_results else "No matching files found or no occurrences of the search term."
        
        return types.GetPromptResult(
            messages=[
                types.PromptMessage(
                    role="user",
                    content=types.TextContent(
                        type="text",
                        text=f"I searched for '{search_term}' in files with extension '{file_extension}'. Here are the results:\n\n{all_results}\n\n"
                             f"Please analyze these search results and explain the context and usage of '{search_term}' in these files."
                    )
                )
            ]
        )
    
    elif name == "compare-two-files":
        # This prompt compares two files using the compare_files tool
        file_uri1 = arguments.get("file_uri1", "")
        file_uri2 = arguments.get("file_uri2", "")
        
        # Call the compare_files tool
        compare_result = await call_tool_and_format("compare_files", {
            "file_uri1": file_uri1,
            "file_uri2": file_uri2
        })
        
        file1_name = os.path.basename(find_file_by_uri(file_uri1) or "unknown_file1")
        file2_name = os.path.basename(find_file_by_uri(file_uri2) or "unknown_file2")
        
        return types.GetPromptResult(
            messages=[
                types.PromptMessage(
                    role="user",
                    content=types.TextContent(
                        type="text",
                        text=f"I compared the files {file1_name} and {file2_name}. Here's the comparison result:\n\n{compare_result}\n\n"
                             f"Please analyze this comparison and explain the key differences between these files, "
                             f"focusing on functionality changes, improvements, or potential issues introduced."
                    )
                )
            ]
        )
    
    elif name == "folder-overview":
        # This prompt gives an overview of the folder structure by calling multiple tools
        
        # Get a list of all files
        all_files_result = await call_tool_and_format("find_files", {
            "search_type": "name",
            "search_term": ""
        })
        
        # Get file counts by extension
        file_counts = {}
        import json
        try:
            all_files_json = json.loads(all_files_result)
            if "matches" in all_files_json:
                for match in all_files_json.get("matches", []):
                    if "name" in match:
                        ext = os.path.splitext(match["name"])[1].lower() or "(no extension)"
                        file_counts[ext] = file_counts.get(ext, 0) + 1
        except:
            # Fallback if JSON parsing fails
            file_counts = {"(unknown)": "Unable to parse file list"}
        
        extensions_summary = "\n".join([f"- {ext}: {count} files" for ext, count in file_counts.items()])
        
        # Try to identify important files (like README, configuration files, etc.)
        important_files = []
        for name_pattern in ["README", "config", "setup", "package.json", "requirements.txt", "Makefile", "Dockerfile", ".env"]:
            pattern_result = await call_tool_and_format("find_files", {
                "search_type": "name",
                "search_term": name_pattern
            })
            
            try:
                pattern_json = json.loads(pattern_result)
                if "matches" in pattern_json:
                    for match in pattern_json.get("matches", []):
                        if "name" in match and "uri" in match:
                            important_files.append(f"- {match['name']}: {match['uri']}")
            except:
                pass
        
        important_files_text = "\n".join(important_files) if important_files else "No common important files found."
        
        return types.GetPromptResult(
            messages=[
                types.PromptMessage(
                    role="user",
                    content=types.TextContent(
                        type="text",
                        text=f"Please provide an overview of the folder structure I'm exploring.\n\n"
                             f"File type summary:\n{extensions_summary}\n\n"
                             f"Important files found:\n{important_files_text}\n\n"
                             f"Based on this information, please:\n"
                             f"1. Characterize what kind of project or codebase this appears to be\n"
                             f"2. Suggest which files might be most important to examine first\n"
                             f"3. Provide guidance on how I should explore this codebase effectively"
                    )
                )
            ]
        )
    
    elif name == "code-complexity-analysis":
        # This prompt analyzes code complexity by examining file contents
        file_uri = arguments.get("file_uri", "")
        
        file_path = find_file_by_uri(file_uri)
        if not file_path or not os.path.exists(file_path) or not is_text_file(file_path):
            return types.GetPromptResult(
                messages=[
                    types.PromptMessage(
                        role="user",
                        content=types.TextContent(
                            type="text",
                            text=f"Please analyze the code complexity of {file_uri}, but I can't find this file or it's not a text file. Can you confirm the URI is correct?"
                        )
                    )
                ]
            )
        
        # Get file info
        file_info_result = await call_tool_and_format("get_file_info", {
            "file_uri": file_uri
        })
        
        # Read file content
        content = ""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # Truncate if too large
                if len(content) > 5000:
                    content = content[:5000] + "\n\n[Content truncated due to size]"
        except Exception as e:
            content = f"Error reading file: {str(e)}"
        
        # Determine language based on file extension
        _, ext = os.path.splitext(file_path)
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
        language = language_map.get(ext.lower(), "")
        
        metrics_prompt = ""
        if language:
            metrics_prompt = f"\n\nPlease calculate and report these complexity metrics for this {language} code:\n"
            if language == "Python":
                metrics_prompt += (
                    "1. Cyclomatic complexity for each function\n"
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
        
        return types.GetPromptResult(
            messages=[
                types.PromptMessage(
                    role="user",
                    content=types.TextContent(
                        type="text",
                        text=f"Please analyze the complexity of this code file:\n\n"
                             f"File info:\n{file_info_result}\n\n"
                             f"Code content:\n```{ext[1:] if ext else ''}\n{content}\n```"
                             f"{metrics_prompt}\n\n"
                             f"After analyzing the metrics, please suggest specific ways to reduce the complexity and improve maintainability."
                    )
                )
            ]
        )
    
    elif name == "summarize-file":
        file_uri = arguments.get("file_uri", "")
        file_path = find_file_by_uri(file_uri)
        
        if not file_path or not os.path.exists(file_path):
            return types.GetPromptResult(
                messages=[
                    types.PromptMessage(
                        role="user",
                        content=types.TextContent(
                            type="text",
                            text=f"Please generate a summary for the file at {file_uri}, but I can't find this file. Can you confirm the URI is correct?"
                        )
                    )
                ]
            )
        
        # Read file content if it's a text file
        content = ""
        if is_text_file(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Truncate if too large
                    if len(content) > 5000:
                        content = content[:5000] + "\n\n[Content truncated due to size]"
            except Exception as e:
                content = f"Error reading file: {str(e)}"
        else:
            content = "[Binary file - content not displayed]"
        
        return types.GetPromptResult(
            messages=[
                types.PromptMessage(
                    role="user",
                    content=types.TextContent(
                        type="text",
                        text=f"Please generate a concise summary of the following file: {os.path.basename(file_path)}\n\n"
                             f"File path: {file_path}\n"
                             f"File type: {get_mime_type(file_path)}\n\n"
                             f"Content:\n```\n{content}\n```\n\n"
                             f"Please focus on the main purpose of the file, key functions/components, and important patterns or structures."
                    )
                )
            ]
        )
    
    elif name == "find-code-pattern":
        pattern = arguments.get("pattern", "")
        file_extension = arguments.get("file_extension", "")
        
        # Find matching files
        search_results = find_files("extension", file_extension) if file_extension else {"matches": []}
        
        # Format file list
        file_list = ""
        for idx, match in enumerate(search_results.get("matches", [])[:10], 1):
            file_list += f"{idx}. {match.get('name')}: {match.get('uri')}\n"
        
        if not file_list:
            file_list = "No files found with the specified extension."
        
        return types.GetPromptResult(
            messages=[
                types.PromptMessage(
                    role="user",
                    content=types.TextContent(
                        type="text",
                        text=f"Please help me find usage patterns related to '{pattern}' in my code files.\n\n"
                             f"Files with extension '{file_extension}':\n{file_list}\n\n"
                             f"For each file, I'd like you to suggest what to look for to identify this pattern, "
                             f"and which of these files are most likely to contain the pattern based on their names."
                    )
                )
            ]
        )
    
    elif name == "git-commit-message":
        file_uris = arguments.get("file_uris", "").split(",")
        change_description = arguments.get("change_description", "")
        
        file_contents = []
        for uri in file_uris:
            uri = uri.strip()
            file_path = find_file_by_uri(uri)
            if file_path and os.path.exists(file_path) and is_text_file(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        # Only include a sample of the content to avoid overwhelming
                        if len(content) > 1000:
                            content = content[:1000] + "\n\n[Content truncated...]"
                    
                    file_contents.append(f"File: {os.path.basename(file_path)}\n```\n{content}\n```")
                except Exception as e:
                    file_contents.append(f"File: {os.path.basename(file_path)}\nError reading file: {str(e)}")
            elif file_path:
                file_contents.append(f"File: {os.path.basename(file_path)} (binary or non-text file)")
            else:
                file_contents.append(f"File URI not found: {uri}")
        
        joined_file_contents = '---\n\n'.join(file_contents)

        return types.GetPromptResult(
            messages=[
                types.PromptMessage(
                    role="user",
                    content=types.TextContent(
                        type="text",
                        text=(
                            f"Please generate a good git commit message for the following changes:\n\n"
                            f"Files changed:\n- {', '.join([os.path.basename(find_file_by_uri(uri.strip())) for uri in file_uris if find_file_by_uri(uri.strip())])}\n\n"
                            f"Description provided by user: {change_description}\n\n"
                            f"File contents:\n\n{joined_file_contents}\n\n"
                            f"Please generate a concise, descriptive commit message following best practices (subject line + optional description)."
                        )
                    )
                )
            ]
        )
    
    elif name == "explain-code":
        file_uri = arguments.get("file_uri", "")
        start_line = arguments.get("start_line", "")
        end_line = arguments.get("end_line", "")
        
        file_path = find_file_by_uri(file_uri)
        if not file_path or not os.path.exists(file_path) or not is_text_file(file_path):
            return types.GetPromptResult(
                messages=[
                    types.PromptMessage(
                        role="user",
                        content=types.TextContent(
                            type="text",
                            text=f"Please explain the code at {file_uri}, but I can't find this file or it's not a text file. Can you confirm the URI is correct?"
                        )
                    )
                ]
            )
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Extract specific lines if provided
            try:
                start = int(start_line) - 1 if start_line else 0
                end = int(end_line) if end_line else len(lines)
                selected_lines = lines[start:end]
                line_range = f"lines {start+1}-{end}" if start_line or end_line else "entire file"
            except (ValueError, IndexError):
                selected_lines = lines
                line_range = "entire file"
            
            content = ''.join(selected_lines)
        except Exception as e:
            content = f"Error reading file: {str(e)}"
            line_range = "N/A"
        
        # Determine language based on file extension
        _, ext = os.path.splitext(file_path)
        language_map = {
            '.py': 'Python',
            '.js': 'JavaScript',
            '.ts': 'TypeScript',
            '.html': 'HTML',
            '.css': 'CSS',
            '.java': 'Java',
            '.c': 'C',
            '.cpp': 'C++',
            '.go': 'Go',
            '.rs': 'Rust',
            '.rb': 'Ruby',
            '.php': 'PHP',
            '.sh': 'Shell',
            '.ps1': 'PowerShell',
            '.sql': 'SQL'
        }
        language = language_map.get(ext.lower(), "code")
        
        return types.GetPromptResult(
            messages=[
                types.PromptMessage(
                    role="user",
                    content=types.TextContent(
                        type="text",
                        text=f"Please explain the following {language} code ({line_range}) from {os.path.basename(file_path)}:\n\n"
                             f"```{ext[1:] if ext else ''}\n{content}\n```\n\n"
                             f"Please explain:\n"
                             f"1. What this code does at a high level\n"
                             f"2. The purpose and behavior of key functions, classes, or sections\n"
                             f"3. Any important patterns, algorithms, or techniques used\n"
                             f"4. Any potential issues or areas for improvement"
                    )
                )
            ]
        )
    
    elif name == "code-review":
        file_uri = arguments.get("file_uri", "")
        focus = arguments.get("focus", "general").lower()
        
        file_path = find_file_by_uri(file_uri)
        if not file_path or not os.path.exists(file_path) or not is_text_file(file_path):
            return types.GetPromptResult(
                messages=[
                    types.PromptMessage(
                        role="user",
                        content=types.TextContent(
                            type="text",
                            text=f"Please review the code at {file_uri}, but I can't find this file or it's not a text file. Can you confirm the URI is correct?"
                        )
                    )
                ]
            )
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # Truncate if too large
                if len(content) > 5000:
                    content = content[:5000] + "\n\n[Content truncated due to size]"
        except Exception as e:
            content = f"Error reading file: {str(e)}"
        
        focus_instructions = {
            "security": "Please focus on security issues like injection vulnerabilities, authentication problems, improper error handling, etc.",
            "performance": "Please focus on performance issues like inefficient algorithms, unnecessary operations, memory leaks, etc.",
            "readability": "Please focus on code clarity, naming conventions, comment quality, and overall maintainability.",
            "general": "Please provide a general review covering correctness, style, performance, and potential bugs."
        }
        
        review_focus = focus_instructions.get(focus, focus_instructions["general"])
        
        return types.GetPromptResult(
            messages=[
                types.PromptMessage(
                    role="user",
                    content=types.TextContent(
                        type="text",
                        text=f"Please review the following code from {os.path.basename(file_path)}:\n\n"
                             f"```\n{content}\n```\n\n"
                             f"{review_focus}\n\n"
                             f"Format your review as:\n"
                             f"1. Overall assessment\n"
                             f"2. Specific issues (with line numbers when possible)\n"
                             f"3. Suggested improvements\n"
                             f"4. Positive aspects worth keeping"
                    )
                )
            ]
        )
    
    # If we get here, it's a prompt we know about but haven't implemented
    raise ValueError(f"Prompt implementation not found for {name}")

# @server.get_prompt()
# async def handle_get_prompt(name: str, arguments: dict[str, str] | None = None) -> types.GetPromptResult:
#     """Handle retrieving a specific prompt with arguments"""
#     if name not in FILE_PROMPTS:
#         raise ValueError(f"Prompt not found: {name}")
    
#     if arguments is None:
#         arguments = {}
    
#     if name == "summarize-file":
#         file_uri = arguments.get("file_uri", "")
#         file_path = find_file_by_uri(file_uri)
        
#         if not file_path or not os.path.exists(file_path):
#             return types.GetPromptResult(
#                 messages=[
#                     types.PromptMessage(
#                         role="user",
#                         content=types.TextContent(
#                             type="text",
#                             text=f"Please generate a summary for the file at {file_uri}, but I can't find this file. Can you confirm the URI is correct?"
#                         )
#                     )
#                 ]
#             )
        
#         # Read file content if it's a text file
#         content = ""
#         if is_text_file(file_path):
#             try:
#                 with open(file_path, 'r', encoding='utf-8') as f:
#                     content = f.read()
#                     # Truncate if too large
#                     if len(content) > 5000:
#                         content = content[:5000] + "\n\n[Content truncated due to size]"
#             except Exception as e:
#                 content = f"Error reading file: {str(e)}"
#         else:
#             content = "[Binary file - content not displayed]"
        
#         return types.GetPromptResult(
#             messages=[
#                 types.PromptMessage(
#                     role="user",
#                     content=types.TextContent(
#                         type="text",
#                         text=f"Please generate a concise summary of the following file: {os.path.basename(file_path)}\n\n"
#                              f"File path: {file_path}\n"
#                              f"File type: {get_mime_type(file_path)}\n\n"
#                              f"Content:\n```\n{content}\n```\n\n"
#                              f"Please focus on the main purpose of the file, key functions/components, and important patterns or structures."
#                     )
#                 )
#             ]
#         )
    
#     elif name == "find-code-pattern":
#         pattern = arguments.get("pattern", "")
#         file_extension = arguments.get("file_extension", "")
        
#         # Find matching files
#         search_results = find_files("extension", file_extension) if file_extension else {"matches": []}
        
#         # Format file list
#         file_list = ""
#         for idx, match in enumerate(search_results.get("matches", [])[:10], 1):
#             file_list += f"{idx}. {match.get('name')}: {match.get('uri')}\n"
        
#         if not file_list:
#             file_list = "No files found with the specified extension."
        
#         return types.GetPromptResult(
#             messages=[
#                 types.PromptMessage(
#                     role="user",
#                     content=types.TextContent(
#                         type="text",
#                         text=f"Please help me find usage patterns related to '{pattern}' in my code files.\n\n"
#                              f"Files with extension '{file_extension}':\n{file_list}\n\n"
#                              f"For each file, I'd like you to suggest what to look for to identify this pattern, "
#                              f"and which of these files are most likely to contain the pattern based on their names."
#                     )
#                 )
#             ]
#         )
    
#     elif name == "git-commit-message":
#         file_uris = arguments.get("file_uris", "").split(",")
#         change_description = arguments.get("change_description", "")
        
#         file_contents = []
#         for uri in file_uris:
#             uri = uri.strip()
#             file_path = find_file_by_uri(uri)
#             if file_path and os.path.exists(file_path) and is_text_file(file_path):
#                 try:
#                     with open(file_path, 'r', encoding='utf-8') as f:
#                         content = f.read()
#                         # Only include a sample of the content to avoid overwhelming
#                         if len(content) > 1000:
#                             content = content[:1000] + "\n\n[Content truncated...]"
                    
#                     file_contents.append(f"File: {os.path.basename(file_path)}\n```\n{content}\n```")
#                 except Exception as e:
#                     file_contents.append(f"File: {os.path.basename(file_path)}\nError reading file: {str(e)}")
#             elif file_path:
#                 file_contents.append(f"File: {os.path.basename(file_path)} (binary or non-text file)")
#             else:
#                 file_contents.append(f"File URI not found: {uri}")
        
#         joined_file_contents = '---\n\n'.join(file_contents)

#         return types.GetPromptResult(
#             messages=[
#                 types.PromptMessage(
#                     role="user",
#                     content=types.TextContent(
#                         type="text",
#                         text=(
#                             f"Please generate a good git commit message for the following changes:\n\n"
#                             f"Files changed:\n- {', '.join([os.path.basename(find_file_by_uri(uri.strip())) for uri in file_uris if find_file_by_uri(uri.strip())])}\n\n"
#                             f"Description provided by user: {change_description}\n\n"
#                             f"File contents:\n\n{joined_file_contents}\n\n"
#                             f"Please generate a concise, descriptive commit message following best practices (subject line + optional description)."
#                         )
#                     )
#                 )
#             ]
#         )
    
#     elif name == "explain-code":
#         file_uri = arguments.get("file_uri", "")
#         start_line = arguments.get("start_line", "")
#         end_line = arguments.get("end_line", "")
        
#         file_path = find_file_by_uri(file_uri)
#         if not file_path or not os.path.exists(file_path) or not is_text_file(file_path):
#             return types.GetPromptResult(
#                 messages=[
#                     types.PromptMessage(
#                         role="user",
#                         content=types.TextContent(
#                             type="text",
#                             text=f"Please explain the code at {file_uri}, but I can't find this file or it's not a text file. Can you confirm the URI is correct?"
#                         )
#                     )
#                 ]
#             )
        
#         try:
#             with open(file_path, 'r', encoding='utf-8') as f:
#                 lines = f.readlines()
            
#             # Extract specific lines if provided
#             try:
#                 start = int(start_line) - 1 if start_line else 0
#                 end = int(end_line) if end_line else len(lines)
#                 selected_lines = lines[start:end]
#                 line_range = f"lines {start+1}-{end}" if start_line or end_line else "entire file"
#             except (ValueError, IndexError):
#                 selected_lines = lines
#                 line_range = "entire file"
            
#             content = ''.join(selected_lines)
#         except Exception as e:
#             content = f"Error reading file: {str(e)}"
#             line_range = "N/A"
        
#         # Determine language based on file extension
#         _, ext = os.path.splitext(file_path)
#         language_map = {
#             '.py': 'Python',
#             '.js': 'JavaScript',
#             '.ts': 'TypeScript',
#             '.html': 'HTML',
#             '.css': 'CSS',
#             '.java': 'Java',
#             '.c': 'C',
#             '.cpp': 'C++',
#             '.go': 'Go',
#             '.rs': 'Rust',
#             '.rb': 'Ruby',
#             '.php': 'PHP',
#             '.sh': 'Shell',
#             '.ps1': 'PowerShell',
#             '.sql': 'SQL'
#         }
#         language = language_map.get(ext.lower(), "code")
        
#         return types.GetPromptResult(
#             messages=[
#                 types.PromptMessage(
#                     role="user",
#                     content=types.TextContent(
#                         type="text",
#                         text=f"Please explain the following {language} code ({line_range}) from {os.path.basename(file_path)}:\n\n"
#                              f"```{ext[1:] if ext else ''}\n{content}\n```\n\n"
#                              f"Please explain:\n"
#                              f"1. What this code does at a high level\n"
#                              f"2. The purpose and behavior of key functions, classes, or sections\n"
#                              f"3. Any important patterns, algorithms, or techniques used\n"
#                              f"4. Any potential issues or areas for improvement"
#                     )
#                 )
#             ]
#         )
    
#     elif name == "code-review":
#         file_uri = arguments.get("file_uri", "")
#         focus = arguments.get("focus", "general").lower()
        
#         file_path = find_file_by_uri(file_uri)
#         if not file_path or not os.path.exists(file_path) or not is_text_file(file_path):
#             return types.GetPromptResult(
#                 messages=[
#                     types.PromptMessage(
#                         role="user",
#                         content=types.TextContent(
#                             type="text",
#                             text=f"Please review the code at {file_uri}, but I can't find this file or it's not a text file. Can you confirm the URI is correct?"
#                         )
#                     )
#                 ]
#             )
        
#         try:
#             with open(file_path, 'r', encoding='utf-8') as f:
#                 content = f.read()
#                 # Truncate if too large
#                 if len(content) > 5000:
#                     content = content[:5000] + "\n\n[Content truncated due to size]"
#         except Exception as e:
#             content = f"Error reading file: {str(e)}"
        
#         focus_instructions = {
#             "security": "Please focus on security issues like injection vulnerabilities, authentication problems, improper error handling, etc.",
#             "performance": "Please focus on performance issues like inefficient algorithms, unnecessary operations, memory leaks, etc.",
#             "readability": "Please focus on code clarity, naming conventions, comment quality, and overall maintainability.",
#             "general": "Please provide a general review covering correctness, style, performance, and potential bugs."
#         }
        
#         review_focus = focus_instructions.get(focus, focus_instructions["general"])
        
#         return types.GetPromptResult(
#             messages=[
#                 types.PromptMessage(
#                     role="user",
#                     content=types.TextContent(
#                         type="text",
#                         text=f"Please review the following code from {os.path.basename(file_path)}:\n\n"
#                              f"```\n{content}\n```\n\n"
#                              f"{review_focus}\n\n"
#                              f"Format your review as:\n"
#                              f"1. Overall assessment\n"
#                              f"2. Specific issues (with line numbers when possible)\n"
#                              f"3. Suggested improvements\n"
#                              f"4. Positive aspects worth keeping"
#                     )
#                 )
#             ]
#         )
    
#     # If we get here, it's a prompt we know about but haven't implemented
#     raise ValueError(f"Prompt implementation not found for {name}")

async def run_stdio_server():
    """Run the server using stdio streams"""
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="folder-explorer",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )

async def run_sse_server(port: int = 8000):
    """Run the server using SSE transport"""
    sse = SseServerTransport("/messages/")

    async def handle_sse(request):
        async with sse.connect_sse(
            request.scope,
            request.receive,
            request._send,  # type: ignore[reportPrivateUsage]
        ) as streams:
            await server.run(
                streams[0],
                streams[1],
                InitializationOptions(
                    server_name="folder-explorer",
                    server_version="0.1.0",
                    capabilities=server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={},
                    ),
                ),
            )

    app = Starlette(
        debug=True,
        routes=[
            Route("/sse", endpoint=handle_sse),
            Mount("/messages/", app=sse.handle_post_message),
        ],
    )

    print(f"Starting SSE server on port {port}")
    print(f"Connect with: python separate_client.py --sse --url http://localhost:{port}/sse")
    
    config = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="info")
    server_instance = uvicorn.Server(config)
    await server_instance.serve()

async def main(transport: Literal["stdio", "sse"] = "stdio", port: int = 8000):
    """Run the server with the specified transport"""
    if transport == "stdio":
        await run_stdio_server()
    elif transport == "sse":
        await run_sse_server(port)
    else:
        raise ValueError(f"Unknown transport: {transport}")

if __name__ == "__main__":
    asyncio.run(main())