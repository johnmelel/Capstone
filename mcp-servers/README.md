# Unified MCP Server for Medical Agent

A single MCP (Model Context Protocol) server providing unified access to both EMR database and medical literature vector store.

## Overview

This MCP server exposes 4 tools:
1. **list_tables** - List all tables in the EMR database
2. **get_schema** - Get schema information for a specific table
3. **run_sql** - Execute read-only SQL queries on the EMR database
4. **semantic_search** - Search medical literature using vector search

## Architecture

```
mcp-servers/
├── server.py           # Unified MCP server implementation
├── test_server.py      # Pytest test suite (12 tests)
├── requirements.txt    # Python dependencies
├── .env               # Configuration (Milvus, Gemini API keys)
└── README.md          # This file
```

## Setup

### 1. Install Dependencies

```bash
cd mcp-servers
pip install -r requirements.txt
```

### 2. Configure Environment

The `.env` file should already be present with these variables:

```bash
# Milvus Configuration
MILVUS_URI=https://in03-e3ea905d2dd3b9f.serverless.gcp-us-west1.cloud.zilliz.com
MILVUS_API_KEY=a3cd9c4a2ca860119038db2bc1a19099339883b02f6d7303791bdefd3dd024f85955df5b8a93fb3c4b9e52fe4f21a2319ff66f0b
MILVUS_COLLECTION_NAME=capstone_group_2

# Gemini Embedding Configuration
GEMINI_API_KEY=AIzaSyDVcKmROn7oDrBwGednZuz7tXOKNZ0fjzo
EMBEDDING_DIMENSION=768
```

### 3. Verify Database

Ensure the EMR database exists:
```bash
ls -lh ../database/mimic_emr.db
# Should show ~30MB file
```

## Usage

### Running as MCP Server (for Agent Integration)

```bash
python server.py
```

The server runs in stdio mode and follows the MCP protocol.

### Testing

Run all tests with pytest:

```bash
pytest test_server.py -v
```

Run specific test class:

```bash
pytest test_server.py::TestEMRDatabaseTools -v
pytest test_server.py::TestVectorSearchTools -v
pytest test_server.py::TestIntegration -v
```

Run a specific test:

```bash
pytest test_server.py::TestEMRDatabaseTools::test_list_tables -v
```

## Tool Specifications

### 1. list_tables()

Lists all tables in the EMR database.

**Parameters:** None

**Returns:**
```json
{
  "tables": ["patients", "admissions", "labevents", ...],
  "count": 15
}
```

### 2. get_schema(table_name)

Gets schema information for a specific table.

**Parameters:**
- `table_name` (string): Name of the table

**Returns:**
```json
{
  "table_name": "patients",
  "columns": [
    {"name": "subject_id", "type": "INTEGER", "not_null": false, "primary_key": true},
    {"name": "gender", "type": "TEXT", "not_null": false, "primary_key": false},
    ...
  ],
  "indexes": [
    {"name": "idx_patients_gender", "columns": ["gender"]}
  ]
}
```

### 3. run_sql(query)

Executes a read-only SQL query.

**Parameters:**
- `query` (string): SQL SELECT query

**Returns:**
```json
{
  "query": "SELECT ...",
  "columns": ["subject_id", "gender", ...],
  "rows": [
    {"subject_id": 10000032, "gender": "F", ...},
    ...
  ],
  "row_count": 100
}
```

**Safety:**
- Only SELECT queries are allowed
- INSERT, UPDATE, DELETE, DROP operations are rejected
- Returns clear error messages for invalid queries

**IMPORTANT for Agents:**
Always JOIN with dictionary tables to get human-readable output:
- `d_labitems` for lab test names (not itemid codes)
- `d_icd_diagnoses` for diagnosis descriptions (not ICD codes)
- `d_icd_procedures` for procedure descriptions (not procedure codes)

Example:
```sql
-- BAD: Returns codes
SELECT itemid FROM labevents WHERE subject_id = 10000032;

-- GOOD: Returns names
SELECT d.label 
FROM labevents l 
JOIN d_labitems d ON l.itemid = d.itemid 
WHERE l.subject_id = 10000032;
```

### 4. semantic_search(query, top_k)

Searches medical literature using vector similarity.

**Parameters:**
- `query` (string): Search query
- `top_k` (integer, default=5): Number of results to return

**Returns:**
```json
{
  "query": "amphetamine monitoring",
  "documents": [
    {
      "text": "Amphetamine monitoring requires...",
      "source": "clinical_guidelines.pdf",
      "relevance": 0.95,
      "doc_id": "12345",
      "file_hash": "abc...",
      "chunk_index": 5,
      "total_chunks": 20
    },
    ...
  ],
  "total_found": 5
}
```

**Note:** Requires Milvus connection. If not connected, tool will raise an exception.

## Database Schema

The EMR database contains 15 tables with ~377K rows:

**Core Tables:**
- `patients` (100 rows) - Patient demographics
- `admissions` (275 rows) - Hospital admissions
- `labevents` (107,727 rows) - Laboratory test results
- `prescriptions` (18,087 rows) - Medication orders
- `diagnoses_icd` (4,506 rows) - Diagnosis codes
- `procedures_icd` (722 rows) - Procedure codes

**Dictionary Tables (CRITICAL):**
- `d_labitems` (1,622 rows) - Lab test definitions
- `d_icd_diagnoses` (109,775 rows) - ICD diagnosis descriptions
- `d_icd_procedures` (85,257 rows) - ICD procedure descriptions

**Additional Tables:**
- `emar`, `transfers`, `drgcodes`, `services`, `pharmacy`, `microbiologyevents`

See `../database/schema/SCHEMA.md` for complete documentation.

## Test Coverage

The test suite includes 12 tests organized into 3 classes:

### TestEMRDatabaseTools (7 tests)
1. List all database tables (verifies 15 tables)
2. Get schema for patients table (verifies 6 columns)
3. Count patients (verifies 100 patients)
4. Query with dictionary JOIN (verifies human-readable output)
5. Reject INSERT queries (security)
6. Reject UPDATE queries (security)
7. Reject DELETE queries (security)

### TestVectorSearchTools (3 tests)
8. Basic semantic search (verifies document retrieval)
9. Contextual semantic search (verifies relevance sorting)
10. Verify top_k parameter (tests different k values)

### TestIntegration (2 tests)
11. Get patient meds and search literature (combines SQL + vector search)
12. Use schema to construct query (schema introspection + query)

**Test Status:**
- Tests 1-7: Always run (EMR database)
- Tests 8-10: Skip if Milvus not connected
- Tests 11-12: Partial skip for vector search portion if Milvus unavailable

## Error Handling

The server provides clear error messages:

- **Database not found:** "Database not found at {path}"
- **Invalid SQL:** "Only SELECT queries are allowed"
- **Table not found:** SQLite error with table name
- **Milvus not connected:** "Milvus is not connected. Cannot perform semantic search."
- **Embedding failure:** "Failed to generate query embedding"

## Performance Considerations

- **Database:** SQLite is optimized with 35+ indexes for fast queries
- **Vector Search:** Uses COSINE similarity with nprobe=16
- **Connection Pooling:** Each request creates a new SQLite connection (lightweight)
- **Milvus:** Single persistent connection, collection loaded at startup

## Integration with Agents

Agents will use this MCP server to access both data sources:

1. **EMR Worker Agent** will use:
   - `list_tables()` - Discover available data
   - `get_schema()` - Understand table structure
   - `run_sql()` - Query patient data

2. **Research Worker Agent** will use:
   - `semantic_search()` - Find relevant medical literature

3. **Router Agent** coordinates both workers based on task requirements

## Development Notes

- Server follows MCP protocol (stdio-based communication)
- Async/await throughout for proper MCP integration
- Comprehensive logging for debugging
- Type hints for better IDE support
- Docstrings for all public methods

## Troubleshooting

**Import Error: No module named 'mcp'**
```bash
pip install mcp>=0.9.0
```

**Database Not Found**
```bash
# Verify database location
ls ../database/mimic_emr.db

# If missing, run setup
cd ../Part1_SQLite_EMR
python setup_database.py
```

**Milvus Connection Failed**
- Check MILVUS_URI and MILVUS_API_KEY in .env
- Verify collection name matches (capstone_group_2)
- Tests will skip vector search if Milvus unavailable

**Gemini API Key Invalid**
- Verify GEMINI_API_KEY in .env
- Embeddings required for semantic_search tool

## Next Steps

After Part 2 completion, proceed to:
- **Part 3:** Build worker agents that consume these MCP tools
- **Part 4:** Build router and planner agents
- **Part 5:** Add personalization with Memento
- **Part 6:** Enable LangSmith tracing
- **Part 7:** End-to-end integration testing
