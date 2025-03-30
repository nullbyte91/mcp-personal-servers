```bash
#server
python -c "import asyncio, code_server_v2; asyncio.run(code_server_v2.main('sse'))" /home/nullbyte/Desktop/mcp_dev/mcp_server/mcp-pinecone/src

# client
python main.py --sse --backend claude --api-key api_key --model claude-3-5-sonnet-20241022

python main.py --backend ollama server/code_server_v2.py

```