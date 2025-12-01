# Database Path Fix - RESOLVED

## Date: November 17, 2025

## Problem Summary
All database queries were failing despite the database being populated with data. The baseline comparison tests showed error messages in answer previews instead of actual medical data.

## Root Cause
**Database Location Mismatch:**
- Database was created at: `medical-agent/mimic.db` (by `scripts/setup_database.py`)
- All components were looking at: `medical-agent/data/mimic.db` (wrong location)

## Files Changed

### 1. `common/mcp_sql_server.py` (Line 18)
**Before:**
```python
def __init__(self, db_path='data/mimic.db'):
```

**After:**
```python
def __init__(self, db_path='mimic.db'):
```

**Impact:** Fixes MCP multi-agent system, single-agent baseline

### 2. `baselines/orchestrator_no_mcp.py` (Line 51)
**Before:**
```python
self.db_path = 'data/mimic.db'
```

**After:**
```python
self.db_path = 'mimic.db'
```

**Impact:** Fixes no-MCP baseline

## Verification Results

### Database Content Confirmed
```bash
sqlite3 medical-agent/mimic.db "SELECT COUNT(*) FROM admissions"
# Result: 275 rows

sqlite3 medical-agent/mimic.db "SELECT COUNT(*) FROM prescriptions"
# Result: 18,087 rows

sqlite3 medical-agent/mimic.db "SELECT COUNT(*) FROM emar"
# Result: 35,835 rows
```

### Single Query Test
```bash
python3 scripts/query_system.py "Show me recent admissions in the last 24 hours"
```

**Result:** Returns actual patient data:
- Subject ID 10020306
- 74-year-old female
- AMBULATORY OBSERVATION admission
- Timestamp: 2034-10-29 08:00:00

### Baseline Comparison Results
All 4 orchestrators now functioning correctly:

| Orchestrator | Success Rate | Avg Steps | Avg Latency |
|-------------|--------------|-----------|-------------|
| MCP Multi-Agent | 100.0% | 8.7 | 27.89s |
| No-MCP | 100.0% | 6.0 | 20.34s |
| Single-Agent | 100.0% | 6.7 | 24.71s |
| Vector-Only | 100.0% | 4.0 | 4.17s |

### Answer Preview Examples (NOW WORKING)

**Before Fix:**
```
"answer_preview": "Based on the provided information, specific possible diagnoses for a patient 
with elevated troponin levels and recent chest pain cannot be determined directly. 
The structured database query failed, pr"
```

**After Fix:**
```
"answer_preview": "Based on the structured EMR data, patients presenting with elevated troponin 
levels and recent chest pain have been diagnosed with a wide variety of conditions. 
These diagnoses range from malignant ne"
```

## Components Affected

### Fixed Components
1. **MCP SQL Server** - Now points to correct database
2. **Structured Worker** - Inherits correct path from MCP SQL Server
3. **Single-Agent Baseline** - Inherits correct path from MCP SQL Server
4. **No-MCP Baseline** - Direct path fix applied

### Unaffected Components
- **Unstructured Worker** - Uses Milvus (vector database), not SQLite
- **Vector-Only Baseline** - No structured data component
- **Orchestrator** - Only coordinates workers, doesn't access database directly

## Testing Commands

### Restart Workers
```bash
cd medical-agent && bash scripts/start_workers.sh
```

### Test Single Query
```bash
cd medical-agent && python3 scripts/query_system.py "Get patients who took insulin"
```

### Run Baseline Comparison
```bash
cd medical-agent && python3 scripts/compare_baselines.py --quick
```

### Direct Database Query
```bash
cd medical-agent && sqlite3 mimic.db "SELECT * FROM prescriptions WHERE drug LIKE '%insulin%' LIMIT 5"
```

## Status: RESOLVED ✓

All database queries now return actual medical data. Baseline comparison shows 100% success rate across all architectures. The system is ready for academic paper publication benchmarks.
