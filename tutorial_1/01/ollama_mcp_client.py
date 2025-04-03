import argparse
import asyncio
import json
import os
import re
import sys
from typing import Dict, List, Optional

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
import httpx


async def call_ollama(
    prompt: str,
    model: str = "llama3.1:8b",
    server_url: str = "http://localhost:11434/api/generate",
):
    """
    Call Ollama API with error handling
    """
    try:
        async with httpx.AsyncClient() as client:
            print(f"Calling Ollama API (model: {model})...")
            response = await client.post(
                server_url,
                json={"model": model, "prompt": prompt, "stream": False},
                timeout=60,
            )
            response.raise_for_status()
            result = response.json()
            return result.get("response", "")
    except Exception as e:
        print(f"Error calling Ollama API: {str(e)}")
        return f"Error: Could not connect to Ollama: {str(e)}"


async def simple_ollama_mcp(
    mcp_command: str,
    mcp_args: List[str],
    file_path: Optional[str] = None,
    model: str = "llama3",
    ollama_url: str = "http://localhost:11434/api/generate",
):
    """
    Run a simple MCP session with Ollama for analysis.
    """
    # Connect to the MCP server
    server_params = StdioServerParameters(
        command=mcp_command,
        args=mcp_args,
    )
    
    async with stdio_client(server_params) as (read_stream, write_stream):
        try:
            # Create client session
            async with ClientSession(read_stream, write_stream) as session:
                # Initialize the session
                init_result = await session.initialize()
                print(f"Connected to MCP server: {init_result.serverInfo.name} v{init_result.serverInfo.version}")
                print(f"Server description: {init_result.instructions or 'No description provided'}")
                print("\nInitializing environment...")
                
                # Create sample code if needed and file_path is not provided
                if not file_path:
                    create_result = await session.call_tool("create_sample_slow_code")
                    print(f"Created sample code: {create_result.content[0].text}")
                    current_file = "sample_slow_code.py"
                else:
                    current_file = file_path
                
                # Get list of code files
                list_files_result = await session.call_tool("list_code_files")
                code_files_text = list_files_result.content[0].text
                files_list = json.loads(code_files_text)
                
                print("\nAvailable code files:")
                for i, file in enumerate(files_list):
                    print(f"{i+1}. {file}")
                
                # Allow file selection if not specified
                if not file_path:
                    file_choice = input("\nEnter the number or name of the file to analyze (default: sample_slow_code.py): ")
                    
                    if file_choice.strip():
                        try:
                            # Check if input is a number (index)
                            if file_choice.isdigit() and 1 <= int(file_choice) <= len(files_list):
                                current_file = files_list[int(file_choice) - 1]
                            # Check if input is a file name
                            elif file_choice in files_list:
                                current_file = file_choice
                            else:
                                print(f"File not found. Using default: {current_file}")
                        except (ValueError, IndexError):
                            print(f"Invalid selection. Using default: {current_file}")
                
                print(f"\nAnalyzing file: {current_file}")
                
                # Get the code content
                read_result = await session.read_resource(f"code://{current_file}")
                code_content = read_result.contents[0].text
                print("\nFile content:")
                print("----------------")
                print(code_content)
                print("----------------")
                
                # Store original performance data for comparison
                original_performance = None
                
                # Main loop
                while True:
                    print("\nOptions:")
                    print("1. Analyze with Ollama")
                    print("2. Optimize with Ollama")
                    print("3. Measure performance")
                    print("4. Run tests")
                    print("5. Switch file")
                    print("6. Exit")
                    
                    choice = input("\nEnter option: ")
                    
                    if choice == "1":  # Analyze with Ollama
                        # Send code to Ollama for analysis
                        prompt = f"""
                        Please analyze this Python code for inefficiencies and suggest optimizations:
                        
                        ```python
                        {code_content}
                        ```
                        
                        Focus on identifying:
                        1. Algorithmic inefficiencies
                        2. Unnecessary operations
                        3. Better data structures
                        4. More Pythonic approaches
                        
                        Don't provide a full rewrite unless necessary, just analysis.
                        """
                        
                        ollama_response = await call_ollama(prompt, model, ollama_url)
                        
                        print("\n=== Ollama Analysis ===\n")
                        print(ollama_response)
                        print("\n=======================")
                    
                    elif choice == "2":  # Optimize with Ollama
                        # Send code to Ollama for optimization
                        prompt = f"""
                        Please optimize this Python code to improve its performance:
                        
                        ```python
                        {code_content}
                        ```
                        
                        Make it more efficient while maintaining the same functionality.
                        Please provide the complete optimized code wrapped in ```python and ``` markers.
                        """
                        
                        ollama_response = await call_ollama(prompt, model, ollama_url)
                        
                        print("\n=== Ollama Optimization ===\n")
                        print(ollama_response)
                        print("\n===========================")
                        
                        # Extract code from Ollama's response
                        code_blocks = re.findall(r"```python\s*(.*?)\s*```", ollama_response, re.DOTALL)
                        
                        if code_blocks:
                            optimized_code = code_blocks[0]
                            update_confirm = input("\nUpdate the code with this optimized version? (yes/no): ")
                            
                            if update_confirm.lower() in ["yes", "y", "sure", "ok", "okay"]:
                                # Measure original performance first
                                try:
                                    # Try to extract function name
                                    function_match = re.search(r"def\s+(\w+)", code_content)
                                    function_name = function_match.group(1) if function_match else "main"
                                    
                                    if original_performance is None:
                                        print(f"\nMeasuring original performance...")
                                        perf_result = await session.call_tool(
                                            "measure_performance",
                                            {
                                                "filepath": current_file,
                                                "function_name": function_name
                                            }
                                        )
                                        perf_data = json.loads(perf_result.content[0].text)
                                        if perf_data["success"]:
                                            original_performance = perf_data["performance"]["average"]
                                            print(f"Original average time: {original_performance:.6f} seconds")
                                    
                                    # Update the code
                                    print(f"\nUpdating {current_file}...")
                                    update_result = await session.call_tool(
                                        "update_code_file", 
                                        {"filepath": current_file, "content": optimized_code}
                                    )
                                    print(update_result.content[0].text)
                                    
                                    # Run tests
                                    potential_test_files = [
                                        f"test_{current_file}",
                                        f"test{current_file}",
                                        "test_sample.py" if current_file == "sample_slow_code.py" else None
                                    ]
                                    
                                    test_file = None
                                    for test_file_name in [f for f in potential_test_files if f]:
                                        if test_file_name in files_list:
                                            test_file = test_file_name
                                            break
                                    
                                    if test_file:
                                        print(f"\nRunning tests: {test_file}")
                                        test_result = await session.call_tool("run_tests", {"test_path": test_file})
                                        test_output = json.loads(test_result.content[0].text)
                                        
                                        if test_output["success"]:
                                            print("✅ Tests are passing with the optimized code.")
                                        else:
                                            print("❌ Tests are failing with the optimized code.")
                                            print(f"Restoring original code...")
                                            # Revert to original code if tests fail
                                            await session.call_tool(
                                                "update_code_file", 
                                                {"filepath": current_file, "content": code_content}
                                            )
                                            continue
                                    
                                    # Measure new performance
                                    print(f"\nMeasuring new performance...")
                                    perf_result = await session.call_tool(
                                        "measure_performance",
                                        {
                                            "filepath": current_file,
                                            "function_name": function_name
                                        }
                                    )
                                    perf_data = json.loads(perf_result.content[0].text)
                                    
                                    if perf_data["success"] and original_performance is not None:
                                        current_avg = perf_data["performance"]["average"]
                                        improvement = original_performance / current_avg
                                        
                                        print("\n=== Performance Comparison ===")
                                        print(f"Original average time: {original_performance:.6f} seconds")
                                        print(f"Optimized average time: {current_avg:.6f} seconds")
                                        print(f"Performance improvement: {improvement:.2f}x faster")
                                except Exception as e:
                                    print(f"Error during optimization process: {str(e)}")
                            else:
                                print("Update cancelled.")
                        else:
                            print("No code blocks found in Ollama's response.")
                    
                    elif choice == "3":  # Measure performance
                        # Try to extract function name
                        function_match = re.search(r"def\s+(\w+)", code_content)
                        function_name = function_match.group(1) if function_match else "main"
                        
                        func_name = input(f"Enter function name to measure (default: {function_name}): ") or function_name
                        
                        try:
                            print(f"Measuring performance of {func_name} in {current_file}...")
                            perf_result = await session.call_tool(
                                "measure_performance", 
                                {
                                    "filepath": current_file,
                                    "function_name": func_name
                                }
                            )
                            perf_data_str = perf_result.content[0].text
                            
                            perf_data = json.loads(perf_data_str)
                            if perf_data["success"]:
                                current_avg = perf_data["performance"]["average"]
                                
                                if original_performance is None:
                                    original_performance = current_avg
                                
                                print(f"\nPerformance:")
                                print(f"- Average execution time: {current_avg:.6f} seconds")
                                print(f"- Minimum time: {perf_data['performance']['min']:.6f} seconds")
                                print(f"- Maximum time: {perf_data['performance']['max']:.6f} seconds")
                                print(f"- Iterations: {perf_data['performance']['iterations']}")
                            else:
                                print(f"Error measuring performance: {perf_data.get('error')}")
                        except Exception as e:
                            print(f"Error measuring performance: {str(e)}")
                    
                    elif choice == "4":  # Run tests
                        test_path = input("Enter test file path (default: test_sample.py): ") or "test_sample.py"
                        
                        try:
                            print(f"Running tests: {test_path}")
                            test_result = await session.call_tool("run_tests", {"test_path": test_path})
                            test_output_str = test_result.content[0].text
                            
                            test_output = json.loads(test_output_str)
                            if test_output["success"]:
                                print("\n✅ Tests passed successfully!")
                            else:
                                print("\n❌ Tests failed!")
                                
                            if test_output.get("stdout"):
                                print("\nStandard output:")
                                print(test_output["stdout"])
                                
                            if test_output.get("stderr"):
                                print("\nError output:")
                                print(test_output["stderr"])
                        except Exception as e:
                            print(f"Error running tests: {str(e)}")
                    
                    elif choice == "5":  # Switch file
                        print("\nAvailable code files:")
                        for i, file in enumerate(files_list):
                            print(f"{i+1}. {file}")
                        
                        file_choice = input("\nEnter the number or name of the file to analyze: ")
                        
                        if file_choice.strip():
                            try:
                                # Check if input is a number (index)
                                if file_choice.isdigit() and 1 <= int(file_choice) <= len(files_list):
                                    new_file = files_list[int(file_choice) - 1]
                                # Check if input is a file name
                                elif file_choice in files_list:
                                    new_file = file_choice
                                else:
                                    print(f"File '{file_choice}' not found.")
                                    continue
                                
                                # Switch to new file
                                current_file = new_file
                                read_result = await session.read_resource(f"code://{current_file}")
                                code_content = read_result.contents[0].text
                                
                                print(f"\nSwitched to file: {current_file}")
                                print("\nFile content:")
                                print("----------------")
                                print(code_content)
                                print("----------------")
                                
                                # Reset performance data
                                original_performance = None
                            except Exception as e:
                                print(f"Error switching files: {str(e)}")
                    
                    elif choice == "6":  # Exit
                        print("Exiting...")
                        break
                    
                    else:
                        print("Invalid option")
                
        except Exception as e:
            print(f"Error in MCP session: {str(e)}")
            if hasattr(e, "__traceback__"):
                import traceback
                traceback.print_exception(type(e), e, e.__traceback__)


def main():
    parser = argparse.ArgumentParser(description="Simple Ollama MCP Client")
    parser.add_argument(
        "--mcp-cmd", 
        default="python",
        help="Command to start the MCP server"
    )
    parser.add_argument(
        "--mcp-args",
        nargs="+",
        default=["code_optimizer_server.py"],
        help="Arguments for the MCP server command"
    )
    parser.add_argument(
        "--file",
        help="Path to the file to analyze"
    )
    parser.add_argument(
        "--model", 
        default="llama3.1:8b", 
        help="Ollama model to use"
    )
    parser.add_argument(
        "--ollama-url", 
        default="http://localhost:11434/api/generate",
        help="URL of the Ollama API server"
    )
    
    args = parser.parse_args()
    
    try:
        asyncio.run(simple_ollama_mcp(
            args.mcp_cmd,
            args.mcp_args,
            args.file,
            args.model,
            args.ollama_url
        ))
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exception(type(e), e, e.__traceback__)


if __name__ == "__main__":
    main()