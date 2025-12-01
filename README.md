# Capstone Project: Iris AI Medical Assistant

## Overview

Welcome to the repository for our capstone project, part of the Master of Science in Applied Data Science program at the University of Chicago. Our team has developed **Iris AI**, a multi-agent medical assistant that combines **Retrieval-Augmented Generation (RAG)** with a **multi-agent architecture** to provide intelligent medical insights from both structured (EMR) and unstructured (medical literature) data sources.

## Team Members

- **Bruna Medeiros**  
- **John Melel**
- **Kyler Rosen**  
- **Samuel Martinez Koss**

## System Architecture

Iris AI implements a sophisticated multi-agent workflow using:

- **LangGraph** - Multi-agent coordination framework
- **MCP (Model Context Protocol)** - Standardized data access layer
- **Gemini 2.5 Flash** - LLM for reasoning and generation
- **FastAPI + React** - Modern web interface
- **LangSmith** - Full observability and tracing

### Agent Flow

```
User Query → Planner → Router → {EMR Worker, Research Worker} → Aggregator → Clinical Report
```

## Quick Start

### Prerequisites

1. Python 3.10+ with virtual environment
2. Node.js 16+ with npm
3. Environment variables configured in `.env`:
   - `GOOGLE_API_KEY` (Gemini API)
   - `LANGSMITH_API_KEY` (tracing)
   - `MILVUS_URI` (vector store)
   - `MILVUS_API_KEY` (vector auth)

### Run the Application

```bash
# Start everything at once (macOS/Linux)
./start_all.sh

# Or manually:
# Terminal 1 - Backend
./start_api.sh

# Terminal 2 - Frontend
./start_frontend.sh
```

Then open http://localhost:5173

For complete documentation, see [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md)

## Project Structure

```
Capstone/
├── agents/                  # LangGraph multi-agent system
│   ├── graph.py            # Main orchestration
│   ├── graph_nodes.py      # Agent implementations
│   ├── emr_worker.py       # EMR database queries
│   └── research_worker.py  # Medical literature search
├── mcp-servers/            # Model Context Protocol server
│   └── server.py          # Unified data access layer
├── front_end/             # Web interface
│   ├── api/              # FastAPI backend
│   └── client/           # React frontend
├── emr-database-setup/   # EMR database (MIMIC-IV)
├── data_ingestion/       # Vector store ingestion pipeline
└── memory-bank/          # Project documentation
```

## Key Features

### Multi-Agent Coordination
- **Planner Agent**: Analyzes queries and determines routing
- **EMR Worker**: Queries structured patient data from SQLite database
- **Research Worker**: Searches medical literature in vector store
- **Aggregator**: Generates professional clinical reports

### Data Access via MCP
- All data access abstracted through MCP protocol
- Read-only SQL queries with safety checks
- Semantic search over medical literature
- Query rewriting for better retrieval

### Clinical Report Generation
- Professional three-section format:
  - CLINICAL SUMMARY
  - KEY FINDINGS
  - CLINICAL RECOMMENDATIONS
- Medically accurate terminology
- Reference range interpretation
- Actionable insights for clinicians

### Web Interface
- Real-time chat interface
- Conversation history with localStorage persistence
- Pinned questions with auto-execution
- Patient selection UI (integration in progress)
- Resizable sidebars

### Observability
- Full LangSmith tracing
- Visual workflow in LangSmith Studio
- MCP tool call tracking
- Performance metrics

## Sample Queries

### EMR Queries
```
What are the lab results for patient 10000032?
Show me diagnoses for patient 10000032
What medications was patient 10000032 prescribed?
```

### Research Queries
```
What are treatment guidelines for hypertension?
Explain macrocytic anemia
What causes thrombocytopenia?
```

### Combined Queries
```
Patient 10000032 has elevated creatinine. What does the literature say?
Patient 10000032 has low platelets. What are the causes?
```

## Technical Stack

### Backend
- **LangGraph** - Multi-agent orchestration
- **FastAPI** - REST API
- **MCP** - Data access protocol
- **SQLite** - EMR database (MIMIC-IV subset)
- **Milvus** - Vector database for medical literature
- **Google Gemini 2.5 Flash** - LLM
- **LangSmith** - Tracing and observability

### Frontend
- **React 19** - UI framework
- **Vite** - Build tool
- **React Markdown** - Formatted responses

### Data
- **MIMIC-IV** - Clinical database (de-identified)
- **Medical Literature** - Vector store corpus
- **Dictionary Tables** - Human-readable medical codes

## Development Status

### ✅ Completed
- Part 1: SQLite EMR database from MIMIC-IV
- Part 2: MCP server for SQL and vector access
- Part 3: EMR and research worker agents
- Part 4: Multi-agent coordination with LangGraph
- Part 6: LangSmith tracing and clinical reports
- **NEW**: Frontend-backend integration

### 🚧 In Progress
- Patient selection integration
- Enhanced error handling
- Performance optimization

### 📋 Planned (Future)
- Part 5: Memento personalization integration
- Authentication and authorization
- Multi-patient query support
- Expanded medical literature corpus
- Real-time streaming responses

## Testing

```bash
# Test backend agents
cd agents
python -m pytest test_graph.py -v

# Test MCP server
cd mcp-servers
python -m pytest test_server.py -v

# Test API endpoint
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What are lab results for patient 10000032?"}'
```

## Known Limitations

### MVP Scope
- No authentication/authorization
- Patient selection UI not fully integrated
- Basic error messages
- No retry logic for failed queries

### Data Limitations
- EMR database: MIMIC-IV subset (limited patients)
- Vector store: Limited medical literature corpus
- Research queries may return "no relevant documents"

### Technical Limitations
- Rate limits: 10 RPM Gemini, 250 RPD (free tier)
- No persistent conversation storage on backend
- No streaming responses
- Single-threaded processing

## Documentation

- [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md) - Complete setup and usage guide
- [agents/README.md](agents/README.md) - Multi-agent system documentation
- [mcp-servers/README.md](mcp-servers/README.md) - MCP server documentation
- [emr-database-setup/README.md](emr-database-setup/README.md) - Database documentation
- [memory-bank/](memory-bank/) - Project decisions and learnings

## Observability

### LangSmith Traces
View execution traces at https://smith.langchain.com

Every query shows:
- Planner reasoning
- Routing decisions
- SQL generation and execution
- Vector search queries
- Report generation
- Timing and performance

### LangSmith Studio
Visual workflow at https://smith.langchain.com/studio

Interactive flowchart showing:
- Agent coordination
- Conditional routing
- Parallel execution
- State transitions

## Acknowledgments

This project builds upon:
- **MIMIC-IV** clinical database (MIT LCP)
- **LangChain ecosystem** (LangGraph, LangSmith)
- **Model Context Protocol** (Anthropic)
- **University of Chicago Medicine** domain expertise

## License

This project is developed as part of academic research at the University of Chicago.

## Contact

For questions or collaboration:
- Submit an issue on GitHub
- Contact the team through University of Chicago MSADS program
