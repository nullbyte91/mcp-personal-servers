#!/usr/bin/env python3
import asyncio
import argparse
import os
import sys

# Add parent directory to sys.path if needed
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from client.folder_explorer import FolderExplorerClient

async def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Folder Explorer with AI Integration")
    parser.add_argument("--sse", action="store_true", help="Connect via SSE (for separate server)")
    parser.add_argument("--url", default="http://localhost:8000/sse", help="SSE URL (default: http://localhost:8000/sse)")
    parser.add_argument("--backend", default="ollama", choices=["ollama", "claude"], help="AI backend to use (default: ollama)")
    parser.add_argument("--model", help="Model to use with the selected backend")
    parser.add_argument("--api-key", help="API key for Claude backend")
    parser.add_argument("--ollama-url", default="http://localhost:11434", help="Ollama API URL (default: http://localhost:11434)")
    parser.add_argument("script", nargs="?", help="Path to server script when not using SSE")
    parser.add_argument("folder", nargs="?", help="Folder to explore (passed to server)")
    
    args = parser.parse_args()
    
    # Configure backend options
    backend_options = {}
    if args.backend == "ollama":
        backend_options = {
            "model": args.model or os.getenv("OLLAMA_MODEL", "deepseek-r1:7b"),
            "url": args.ollama_url or os.getenv("OLLAMA_URL", "http://localhost:11434")
        }
    else:  # claude
        backend_options = {
            "api_key": args.api_key or os.getenv("ANTHROPIC_API_KEY"),
            "model": args.model or os.getenv("CLAUDE_MODEL", "claude-3-5-sonnet-20241022")
        }
    
    # Initialize the client
    client = FolderExplorerClient(backend_type=args.backend, backend_options=backend_options)
    
    try:
        # Connect either via SSE or by launching a subprocess
        if args.sse:
            print(f"Connecting to folder explorer server at {args.url}")
            await client.connect_to_server_sse(args.url)
        else:
            if not args.script:
                print("Error: When not using --sse, you must provide a server script path")
                return
            
            print(f"Starting folder explorer server from script: {args.script}")
            await client.connect_to_server_stdio(args.script)
            
        # Start the chat loop
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())