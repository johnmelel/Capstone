# A2A Clinical Retrieval System

An Agent-to-Agent (A2A) system that answers clinical queries by combining structured EMR data with unstructured medical knowledge using real MCP protocol implementations.

## Architecture

```
User Query
   │
   ▼
[Orchestrator + Gemini]
   ├─ A2A → Structured Worker → MCP-SQL → MySQL (MIMIC EMR)
   └─ A2A → Unstructured Worker → MCP-Vector → Medical Knowledge
   │
   ▼
Final JSON Answer
```

### Components

1. **Orchestrator** (`orchestrator/main.py`): Coordinates workers and uses Gemini for reasoning
2. **Structured Worker** (`agents/structured_worker/`): Queries MySQL via MCP-SQL
3. **Unstructured Worker** (`agents/unstructured_worker/`): Searches medical knowledge via MCP-Vector
4. **MCP Servers** (`common/mcp_*_server.py`): Protocol implementations for data access
5. **A2A Protocol** (`common/a2a_*.py`): Message passing between agents

## Quick Start

### 1. Setup Environment

```bash
# Clone and enter directory
cd medical-agent

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Setup Google Cloud Credentials

Your service account JSON file `adsp-34002-ip09-team-2-e0cca2d396a9.json` is already in place. 

### 3. Run the System

```bash
# Start both workers (in one terminal)
source venv/bin/activate
./scripts/start_workers.sh

# In another terminal, run queries
source venv/bin/activate
python scripts/query_system.py "Get me all patients who took Amphetamine in the last 24 hours"
```

## Example Queries

The system currently supports these types of queries:

```bash
# Drug-related queries
python scripts/query_system.py "Get patients who took amphetamine in last 24 hours"
python scripts/query_system.py "Show patients on insulin in past 12 hours"

# Admission queries  
python scripts/query_system.py "Recent admissions in last 48 hours"

# With verbose output
python scripts/query_system.py "amphetamine patients" --verbose

# JSON output
python scripts/query_system.py "insulin monitoring" --output json
```

### Data Sources

**Structured (MySQL)**:
- `patients` - Patient demographics
- `prescriptions` - Medication prescriptions  
- `emar` - Electronic medication administration
- `admissions` - Hospital admissions
- `pharmacy` - Pharmacy orders

**Unstructured (Mock)**:
- Drug monitoring guidelines
- Safety warnings and contraindications
- Clinical practice standards

## Testing

Comprehensive test suite with TDD approach:

```bash
# Run all tests
source venv/bin/activate
python -m pytest tests/ -v

# Run specific test files
python -m pytest tests/test_a2a_messages.py -v
python -m pytest tests/test_orchestrator.py -v
```

## Database Setup (Optional)

To use real MIMIC data instead of mocked responses:

```bash
# Install and start MySQL locally
brew install mysql
brew services start mysql

# Create database and load MIMIC data
python scripts/setup_database.py

# Update connection settings in common/mcp_sql_server.py if needed
```

**Note**: Currently works with dummy/mock data for demonstration.

## API Endpoints

When workers are running:

**Structured Worker (port 8001)**:
- `POST /a2a/task` - Process A2A tasks
- `GET /health` - Health check
- `GET /tools` - List available MCP tools

**Unstructured Worker (port 8002)**:
- Same endpoints as structured worker

## Output Format

```json
{
  "answer": "Found 3 patients took amphetamine in the last 24 hours. Monitor cardiac function and blood pressure.",
  "structured_source": ["prescriptions", "emar", "patients"],
  "unstructured_source": ["pharm_guide_001", "safety_manual_034"]
}
```

## Development

### Adding New Query Types

1. **Extend SQL Query Builder** (`common/mcp_sql_server.py`):
   ```python
   @staticmethod
   def find_patients_with_condition(condition: str, hours_back: int = 24) -> str:
       # Add new SQL query template
   ```

2. **Update Structured Worker** (`agents/structured_worker/main.py`):
   ```python
   def _generate_sql_from_query(self, query: str, parameters: Dict[str, Any]) -> str:
       # Add condition detection logic
   ```

3. **Add Medical Knowledge** (`common/mcp_vector_server.py`):
   ```python
   self.medical_knowledge["new_condition"] = [
       {"text": "Clinical guidelines...", "source": "...", "doc_id": "..."}
   ]
   ```

### Adding Tests

```python
def test_new_feature(self):
    """Test new functionality"""
    # Arrange
    input_data = "test input"
    
    # Act  
    result = system.process(input_data)
    
    # Assert
    assert result.success
    assert "expected" in result.answer
```


## File Structure

```
medical-agent/
├── common/                 # Shared A2A and MCP components
│   ├── a2a_messages.py    # Task/Artifact message classes
│   ├── a2a_transport.py   # HTTP transport layer
│   ├── mcp_sql_server.py  # MCP SQL server implementation
│   └── mcp_vector_server.py # MCP Vector server (dummy)
├── agents/
│   ├── structured_worker/  # Database query worker
│   └── unstructured_worker/ # Medical knowledge worker
├── orchestrator/           # Main coordinator + Gemini
├── scripts/               # Utilities and CLI
├── tests/                 # Comprehensive test suite
├── MIMIC_data/           # Sample EMR data
└── venv/                 # Python virtual environment
```


Built as part of UChicago ADSP Capstone project demonstrating A2A and MCP protocols for clinical data retrieval.
