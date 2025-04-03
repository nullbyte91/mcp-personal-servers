import os
import sys
import time
from pathlib import Path
import subprocess
import json
import re
import inspect

from mcp.server.fastmcp import FastMCP, Context

# Initialize the FastMCP server
server = FastMCP(
    name="OptimizedMCPServer",
    instructions="This server provides tools to analyze, optimize, and test code with function-specific testing."
)

# Define the project root directory
PROJECT_ROOT = Path(os.getcwd())


@server.tool(description="List available code files in the project")
def list_code_files(extension: str = "py", ctx: Context = None) -> str:
    """List all code files with the given extension."""
    files = []
    for path in PROJECT_ROOT.glob(f"*.{extension}"):
        if path.is_file():
            files.append(str(path.name))
    
    return json.dumps(files)


@server.resource("code://{filepath}")
def get_code_file(filepath: str) -> str:
    """Return the content of a code file."""
    file_path = PROJECT_ROOT / filepath
    if not file_path.exists():
        raise ValueError(f"File not found: {filepath}")
    
    return file_path.read_text()


@server.tool(description="Update a code file with new content")
def update_code_file(filepath: str, content: str, ctx: Context = None) -> str:
    """Update a code file with new content."""
    file_path = PROJECT_ROOT / filepath
    
    # Create a backup if file exists
    if file_path.exists():
        backup_path = file_path.with_suffix(f"{file_path.suffix}.bak")
        backup_path.write_text(file_path.read_text())
    
    # Write the new content
    file_path.write_text(content)
    
    return f"Updated {filepath}"


@server.tool(description="Create a sample slow code file for demonstration")
def create_sample_slow_code(filepath: str = "sample_slow_code.py", ctx: Context = None) -> str:
    """Create a sample Python file with inefficient code for optimization demo."""
    file_path = PROJECT_ROOT / filepath
    
    # Sample inefficient code
    code = """
def find_duplicates(numbers):
    '''
    Find all numbers that appear more than once in the input list.
    This implementation is inefficient and can be optimized.
    '''
    duplicates = []
    
    # Inefficient O(n²) implementation
    for i in range(len(numbers)):
        for j in range(len(numbers)):
            if i != j and numbers[i] == numbers[j] and numbers[i] not in duplicates:
                duplicates.append(numbers[i])
    
    return duplicates

def main():
    # Create a test list with duplicates
    test_list = list(range(1000)) + list(range(500))
    
    # Find duplicates
    result = find_duplicates(test_list)
    print(f"Found {len(result)} duplicates")
    
if __name__ == "__main__":
    main()
"""
    
    # Write the file
    file_path.write_text(code)
    
    # Create a test file
    test_path = PROJECT_ROOT / "test_sample.py"
    test_code = """
import unittest
from sample_slow_code import find_duplicates

class TestFindDuplicates(unittest.TestCase):
    def test_find_duplicates(self):
        # Test with an empty list
        self.assertEqual(find_duplicates([]), [])
        
        # Test with no duplicates
        self.assertEqual(sorted(find_duplicates([1, 2, 3, 4])), [])
        
        # Test with duplicates
        self.assertEqual(sorted(find_duplicates([1, 2, 2, 3, 4, 4])), [2, 4])
        
        # Test with all duplicates
        self.assertEqual(sorted(find_duplicates([1, 1, 1])), [1])

if __name__ == "__main__":
    unittest.main()
"""
    test_path.write_text(test_code)
    
    return f"Created sample slow code at {filepath} and test file at test_sample.py."


@server.tool(description="Run tests for the project")
async def run_tests(test_path: str = "test_sample.py", ctx: Context = None) -> str:
    """Run tests for the project."""
    if ctx:
        await ctx.info("Running tests...")
    
    test_file = PROJECT_ROOT / test_path
    
    try:
        result = subprocess.run(
            [sys.executable, str(test_file)],
            capture_output=True,
            text=True,
            check=False,
        )
        
        return json.dumps({
            "success": result.returncode == 0,
            "exit_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        })
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e),
        })


@server.tool(description="Measure code performance")
async def measure_performance(
    filepath: str = "sample_slow_code.py", 
    function_name: str = "find_duplicates",
    iterations: int = 5,
    ctx: Context = None
) -> str:
    """Measure the performance of a Python function with appropriate test data."""
    if ctx:
        await ctx.info(f"Measuring performance...")
    
    file_path = PROJECT_ROOT / filepath
    if not file_path.exists():
        return json.dumps({"success": False, "error": f"File not found: {filepath}"})
    
    # Simple timing approach - import the module and time the function execution
    try:
        module_name = filepath.replace(".py", "")
        
        # Use importlib to import the module
        import importlib.util
        spec = importlib.util.spec_from_file_location(module_name, str(file_path))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Get the function
        func = getattr(module, function_name)
        
        # Analyze function signature
        sig = inspect.signature(func)
        param_count = len(sig.parameters)
        
        # Create appropriate test data based on function name and signature
        test_data = None
        
        # Known function types
        if function_name == "find_duplicates":
            test_data = list(range(1000)) + list(range(500))
        elif function_name == "factorial":
            test_data = 10  # Use a reasonable size that won't take too long
        # Inference based on parameter names and function name
        elif "factorial" in function_name.lower():
            test_data = 10
        elif param_count == 0:
            test_data = None
        elif param_count == 1:
            # Try to infer type from parameter name
            param_name = list(sig.parameters.keys())[0]
            if any(x in param_name.lower() for x in ["n", "num", "int", "digit"]):
                test_data = 10
            elif any(x in param_name.lower() for x in ["str", "text", "name"]):
                test_data = "test"
            elif any(x in param_name.lower() for x in ["list", "array", "items", "elements"]):
                test_data = list(range(100))
            else:
                # Default to a list for unknown single parameter
                test_data = list(range(100))
        else:
            # For multiple parameters, use defaults if available or generate reasonable values
            test_data = {}
            for name, param in sig.parameters.items():
                if param.default is not inspect.Parameter.empty:
                    # Use default value
                    continue
                
                # Infer type from name
                if any(x in name.lower() for x in ["n", "num", "int", "digit", "count", "size"]):
                    test_data[name] = 10
                elif any(x in name.lower() for x in ["str", "text", "name"]):
                    test_data[name] = "test"
                elif any(x in name.lower() for x in ["list", "array", "items", "elements"]):
                    test_data[name] = list(range(100))
                else:
                    # Default to int for unknown
                    test_data[name] = 10
        
        # Print test data for debugging
        if ctx:
            await ctx.info(f"Using test data: {test_data}")
        
        # Measure performance
        durations = []
        for i in range(iterations):
            start_time = time.time()
            if test_data is None:
                func()
            elif isinstance(test_data, dict):
                func(**test_data)
            else:
                func(test_data)
            end_time = time.time()
            durations.append(end_time - start_time)
        
        # Calculate statistics
        avg_duration = sum(durations) / len(durations)
        min_duration = min(durations)
        max_duration = max(durations)
        
        return json.dumps({
            "success": True,
            "performance": {
                "iterations": iterations,
                "durations": durations,
                "average": avg_duration,
                "min": min_duration,
                "max": max_duration,
                "test_data": str(test_data)
            }
        })
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        return json.dumps({
            "success": False,
            "error": str(e),
            "traceback": tb
        })


if __name__ == "__main__":
    # Run the server
    server.run()