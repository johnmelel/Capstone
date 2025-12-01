# Medical Agent Workers

Worker agents that handle specific data access tasks for the multi-agent medical EMR assistant.

## Overview

This package contains two worker classes that form the data access layer:

1. **EMRWorker**: Queries EMR database via MCP server
   - Converts natural language to SQL
   - Executes read-only queries
   - Summarizes results for clinical context

2. **ResearchWorker**: Searches medical literature via MCP server
   - Refines queries for better retrieval
   - Performs semantic search on vector store
   - Synthesizes findings with evidence and citations

Both workers use:
- **MCP client** for data access (no direct DB/API calls)
- **Gemini LLM** for reasoning and summarization
- **Async/await** for proper integration with LangSmith (future)

## Architecture

```
User Query
    ↓
[Future: host_assistant → planner → router]
    ↓
EMRWorker.handle_task()              ResearchWorker.handle_task()
    ↓                                        ↓
MCP Client                                MCP Client
    ↓                                        ↓
MCP Server (unified-medical-mcp-server)
    ↓                                        ↓
SQLite Database                          Milvus Vector Store
```

## File Structure

```
agents/
├── __init__.py           # Package exports
├── emr_worker.py         # EMRWorker class (220 lines)
├── research_worker.py    # ResearchWorker class (190 lines)
├── test_workers.py       # Pytest suite (10 tests)
├── requirements.txt      # Dependencies
└── README.md            # This file
```

## Setup

### 1. Install Dependencies

```bash
cd agents
pip install -r requirements.txt
```

Dependencies:
- `mcp>=0.9.0` - MCP client SDK
- `google-generativeai>=0.3.0` - Gemini LLM
- `python-dotenv>=1.0.0` - Environment config
- `pytest>=7.4.0` - Testing framework
- `pytest-asyncio>=0.21.0` - Async test support

### 2. Configure Environment

Ensure `.env` file exists in project root with:

```bash
# Gemini API Key (required)
GEMINI_API_KEY=your_gemini_api_key_here

# Milvus Configuration (for Research Worker)
MILVUS_URI=https://your-milvus-instance.cloud
MILVUS_API_KEY=your_milvus_api_key
MILVUS_COLLECTION_NAME=your_collection_name
```

### 3. Verify MCP Server

Ensure the MCP server is available:

```bash
cd ../mcp-servers
python server.py  # Should run without errors
```

## Usage

### EMRWorker

```python
import asyncio
from emr_worker import EMRWorker

async def query_emr():
    worker = EMRWorker(
        mcp_server_command="python ../mcp-servers/server.py",
        gemini_api_key="your_api_key"
    )
    
    result = await worker.handle_task(
        task="What were patient 10000032's most recent lab results?",
        patient_id="10000032"
    )
    
    print(result["summary"])
    print(f"SQL used: {result['sql_used']}")
    print(f"Found {result['row_count']} rows")

asyncio.run(query_emr())
```

**EMRWorker API**:

```python
async def handle_task(
    task: str,           # Natural language query
    patient_id: str = None  # Optional patient filter
) -> Dict[str, Any]
```

Returns:
```python
{
    "success": True,
    "summary": "Human-readable clinical summary",
    "data": {
        "rows": [...],      # Query results
        "columns": [...],   # Column names
        "row_count": 10
    },
    "sql_used": "SELECT ...",  # Generated SQL
    "row_count": 10
}
```

### ResearchWorker

```python
import asyncio
from research_worker import ResearchWorker

async def search_literature():
    worker = ResearchWorker(
        mcp_server_command="python ../mcp-servers/server.py",
        gemini_api_key="your_api_key"
    )
    
    result = await worker.handle_task(
        task="What does the literature say about amphetamine monitoring?",
        top_k=5
    )
    
    print(result["narrative"])
    print(f"Evidence: {result['evidence']}")
    print(f"Sources: {result['sources']}")

asyncio.run(search_literature())
```

**ResearchWorker API**:

```python
async def handle_task(
    task: str,           # Natural language query
    context: str = "",   # Optional EMR context
    top_k: int = 5       # Number of documents to retrieve
) -> Dict[str, Any]
```

Returns:
```python
{
    "success": True,
    "summary": "Clinical summary (same as narrative)",
    "narrative": "2-3 paragraph prose summary",
    "evidence": [
        "Evidence point 1",
        "Evidence point 2",
        ...
    ],
    "sources": [
        "[1] Source name",
        "[2] Source name",
        ...
    ],
    "documents": [...],      # Raw retrieved documents
    "refined_query": "...",  # Query used for search
    "total_found": 5
}
```

## Testing

Run all tests:

```bash
cd agents
pytest test_workers.py -v
```

Run specific test class:

```bash
pytest test_workers.py::TestEMRWorker -v
pytest test_workers.py::TestResearchWorker -v
pytest test_workers.py::TestIntegration -v
```

Run with output:

```bash
pytest test_workers.py -v -s
```

### Test Coverage

**TestEMRWorker** (5 tests):
1. Worker initialization
2. Query patient lab results (with dictionary JOIN)
3. Query patient diagnoses (with ICD descriptions)
4. Query patient medications
5. Handle queries with no results

**TestResearchWorker** (4 tests):
6. Worker initialization
7. Basic literature search
8. Search with EMR context
9. Verify synthesis quality

**TestIntegration** (2 tests):
10. EMR → Research workflow (get medication, then search guidelines)
11. Research → EMR workflow (research condition, then find patients)

**Expected Results**:
- EMR tests: Always pass (uses local database)
- Research tests: Skip gracefully if Milvus not connected
- Integration tests: Demonstrate full workflow

## Key Features

### EMRWorker Features

1. **Natural Language to SQL**: Uses Gemini to generate appropriate SQL queries
2. **Dictionary Table JOINs**: Always includes d_labitems, d_icd_diagnoses, d_icd_procedures for human-readable output
3. **Clinical Summarization**: Converts query results into clinician-friendly summaries
4. **Schema-Aware**: Retrieves and uses database schema for accurate query generation
5. **Read-Only Safety**: Only SELECT queries executed via MCP server

### ResearchWorker Features

1. **Query Refinement**: Converts conversational queries to medical terminology
2. **Semantic Search**: Uses vector similarity on medical literature
3. **Evidence Synthesis**: Produces narrative + evidence bullets + citations
4. **Context Integration**: Can incorporate EMR context for better retrieval
5. **Relevance Ranking**: Returns most relevant documents first

## Design Patterns

### MCP Client Pattern

Both workers use the same MCP client pattern:

```python
server_params = StdioServerParameters(
    command="python",
    args=["../mcp-servers/server.py"],
    env=None
)

async with stdio_client(server_params) as (read, write):
    async with ClientSession(read, write) as session:
        await session.initialize()
        
        # Call MCP tools
        result = await session.call_tool(
            "tool_name",
            {"arg1": "value1"}
        )
```

This pattern:
- Starts MCP server as subprocess
- Uses stdio for communication
- Properly manages session lifecycle
- Parses JSON responses from tools

### LLM Integration Pattern

Both workers use Gemini for reasoning:

```python
# Configure once in __init__
genai.configure(api_key=gemini_api_key)
self.model = genai.GenerativeModel('gemini-2.0-flash-exp')

# Use for reasoning
response = self.model.generate_content(prompt)
result = response.text.strip()
```

## Integration with Future Parts

### Part 4: Router and Planner

The workers will be called by LangSmith assistants:

```python
# In emr_assistant (LangSmith)
async def handle_a2a_message(message):
    worker = EMRWorker(...)
    result = await worker.handle_task(
        task=message.content,
        patient_id=message.metadata.get("patient_id")
    )
    return A2AResponse(content=result["summary"])
```

### Part 5: Personalization

ResearchWorker context parameter supports EMR data:

```python
# Router combines EMR + Research
emr_summary = await emr_worker.handle_task(...)
research_result = await research_worker.handle_task(
    task="Research this medication",
    context=emr_summary["summary"]  # EMR context
)
```

### Part 6: LangSmith Tracing

Workers are instrumented for tracing:

```python
from langsmith import trace

@trace(name="emr_worker")
async def handle_task(...):
    # Worker logic here
    # Tool calls automatically traced
```

## Error Handling

Both workers return structured error responses:

```python
{
    "success": False,
    "error": "Error message here",
    "summary": "Failed to process query: [error]"
}
```

Common errors:
- **MCP server not running**: "Connection refused"
- **Gemini API key invalid**: "API key not valid"
- **Milvus not connected**: "Milvus is not connected" (Research only)
- **Invalid SQL**: "Only SELECT queries are allowed"

## Performance Considerations

- **EMR queries**: Typically 3-5 seconds (schema retrieval + SQL generation + execution)
- **Research queries**: Typically 5-8 seconds (query refinement + search + synthesis)
- **Schema caching**: Could be added for repeated queries
- **Connection pooling**: MCP server manages SQLite connections

## Limitations

1. **Single MCP server instance**: Each worker starts a new server process
   - Future: Shared server instance for better performance
2. **Gemini API rate limits**: May hit limits with many concurrent requests
   - Future: Add rate limiting and retry logic
3. **No query caching**: Same query re-executes fully
   - Future: Add result caching layer
4. **Basic query refinement**: Not yet using RL-tuned rewriter
   - Future: Part 4 will add RL query optimization

## Next Steps

After Part 3 completion:

1. **Part 4**: Wrap workers in LangSmith assistants
2. **Part 5**: Add Memento personalization agent
3. **Part 6**: Enable full LangSmith tracing
4. **Part 7**: End-to-end integration testing

## Troubleshooting

**Import Error: No module named 'mcp'**
```bash
pip install mcp>=0.9.0
```

**Import Error: No module named 'google.generativeai'**
```bash
pip install google-generativeai>=0.3.0
```

**MCP Server Not Found**
```bash
# Verify path in test_workers.py
MCP_SERVER_PATH = PROJECT_ROOT / "mcp-servers" / "server.py"
```

**Gemini API Key Not Set**
```bash
# Add to .env in project root
GEMINI_API_KEY=your_key_here
```

**Tests Skip (Milvus)**
- Research tests skip if Milvus not connected
- This is expected behavior
- EMR tests should still pass

## Contributing

When adding new worker functionality:

1. Add method to appropriate worker class
2. Add test case to test_workers.py
3. Update this README with usage example
4. Ensure MCP-only data access (no direct DB calls)
5. Use Gemini for all reasoning/summarization

## License

Part of the Capstone multi-agent medical EMR assistant project.
