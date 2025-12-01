# Baseline Systems for Paper Evaluation

This directory contains three baseline orchestrator implementations designed to demonstrate the performance advantages of the MCP-based multi-agent architecture.

## Overview

Each baseline strategically removes or modifies a key architectural feature to demonstrate its value:

1. **No-MCP Orchestrator** - Tests MCP protocol benefits
2. **Single-Agent Orchestrator** - Tests multi-agent specialization benefits
3. **Vector-Only Orchestrator** - Tests multi-modal data integration benefits

## Baseline Descriptions

### 1. No-MCP Orchestrator (`orchestrator_no_mcp.py`)
**Purpose**: Demonstrate the value of MCP protocol abstraction

**Key Differences from Main System**:
- Uses direct SQLite connections instead of MCP SQL server
- Uses direct Milvus connections instead of MCP vector server
- Bypasses protocol standardization layer
- Direct database access without tool abstraction

**Expected Impact**:
- Similar functionality but less maintainable code
- No protocol-level error handling benefits
- Direct coupling to specific database implementations
- More complex integration logic

### 2. Single-Agent Orchestrator (`orchestrator_single_agent.py`)
**Purpose**: Demonstrate the value of agent specialization and separation of concerns

**Key Differences from Main System**:
- Monolithic architecture - orchestrator does everything
- No A2A communication (no HTTP between agents)
- SQL generation logic embedded in orchestrator
- Vector search logic embedded in orchestrator
- Single component handles all responsibilities

**Expected Impact**:
- Less modular and harder to maintain
- Cannot scale individual components independently
- All logic tightly coupled in one class
- No specialization benefits

### 3. Vector-Only Orchestrator (`orchestrator_vector_only.py`)
**Purpose**: Demonstrate the value of multi-modal data integration

**Key Differences from Main System**:
- Only uses medical literature (vector search)
- No access to patient-specific EMR data
- Cannot answer patient-specific queries
- Limited to general medical guidance

**Expected Impact**:
- Significantly limited functionality
- Cannot provide patient-specific insights
- Lower reasoning depth on clinical queries
- Missing structured data context

## Testing Framework

Use the enhanced `test_reasoning_categories.py` script to compare all systems:

```bash
# Test all architectures with timing
python scripts/test_reasoning_categories.py --compare-all

# Test specific baseline
python scripts/test_reasoning_categories.py --orchestrator-type single_agent

# Available orchestrator types:
# - mcp_multi_agent (default - main system)
# - no_mcp
# - single_agent
# - vector_only
```

## Metrics Collected

For each baseline vs. the main MCP system, we measure:

1. **Reasoning Steps** - Number of logical reasoning steps (depth of analysis)
2. **Success Rate** - Percentage of queries successfully answered
3. **Response Length** - Character count (proxy for completeness)
4. **Latency** - End-to-end query processing time in seconds
5. **Multi-Modal Utilization** - Percentage using both data sources

## Expected Results

Based on system architecture, we anticipate:

| Metric | MCP Multi-Agent | No-MCP | Single-Agent | Vector-Only |
|--------|-----------------|---------|--------------|-------------|
| Reasoning Steps | 8.1 | ~8.0 | ~7.5 | ~4.0 |
| Success Rate | 100% | ~95% | ~90% | ~60% |
| Response Length | High | High | Medium | Low |
| Latency | Low | Medium | Low | Lowest |
| Multi-Modal Use | 90%+ | 90%+ | 85%+ | 0% |

## Architecture Comparison

### Main MCP Multi-Agent System (Control)
```
Orchestrator → A2A → Specialized Workers → MCP → Databases
```
- Clean separation of concerns
- Protocol-level abstraction
- Independently scalable components
- Specialized worker logic

### No-MCP Baseline
```
Orchestrator → A2A → Workers → Direct DB Connections
```
- Multi-agent architecture preserved
- MCP abstraction removed
- Direct database coupling

### Single-Agent Baseline
```
Monolithic Orchestrator → Direct MCP → Databases
```
- Agent specialization removed
- All logic in one component
- Still uses MCP protocol

### Vector-Only Baseline
```
Orchestrator → A2A → Unstructured Worker Only
```
- Multi-modal integration removed
- Only medical literature available
- Missing patient-specific data

## Running the Baselines

Each baseline can be run standalone:

```bash
# No-MCP baseline
python baselines/orchestrator_no_mcp.py "What medications was the patient taking?"

# Single-Agent baseline
python baselines/orchestrator_single_agent.py "What medications was the patient taking?"

# Vector-Only baseline
python baselines/orchestrator_vector_only.py "What are the side effects of insulin?"
```

## Academic Contribution

These baselines enable us to quantitatively demonstrate:

1. **MCP Protocol Value**: By comparing No-MCP vs MCP systems
2. **Agent Specialization Value**: By comparing Single-Agent vs Multi-Agent
3. **Multi-Modal Integration Value**: By comparing Vector-Only vs Full System

The results will provide empirical evidence for architectural decisions in the academic paper.
