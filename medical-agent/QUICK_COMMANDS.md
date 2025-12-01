# Quick Command Reference

## Start Workers

```bash
cd medical-agent && source venv/bin/activate && ./scripts/start_workers.sh
```

## Test Custom Queries

### Basic Query
```bash
cd medical-agent && source venv/bin/activate && python3 scripts/query_system.py "YOUR QUERY HERE"
```

### Query with Verbose Output (shows reasoning steps)
```bash
cd medical-agent && source venv/bin/activate && python3 scripts/query_system.py "YOUR QUERY HERE" --verbose
```

## Example Queries

### Patient Medications
```bash
cd medical-agent && source venv/bin/activate && python3 scripts/query_system.py "What medications were prescribed in the last 24 hours?"
```

### Drug Side Effects
```bash
cd medical-agent && source venv/bin/activate && python3 scripts/query_system.py "What are the side effects of insulin?"
```

### Patient Admissions
```bash
cd medical-agent && source venv/bin/activate && python3 scripts/query_system.py "Show me recent hospital admissions"
```

### Lab Results
```bash
cd medical-agent && source venv/bin/activate && python3 scripts/query_system.py "What patients have abnormal lab results?"
```

## Test Reasoning Categories

```bash
cd medical-agent && source venv/bin/activate && python3 scripts/test_reasoning_categories.py
```

## Compare Baselines

### Quick Test (3 queries)
```bash
cd medical-agent && source venv/bin/activate && python3 scripts/compare_baselines.py --quick
```

### Full Test (10 queries)
```bash
cd medical-agent && source venv/bin/activate && python3 scripts/compare_baselines.py --orchestrator-type all
```

### Test Specific Baseline
```bash
cd medical-agent && source venv/bin/activate && python3 scripts/compare_baselines.py --orchestrator-type single_agent
```

## Stop Workers

```bash
pkill -f "python3 agents"
```

Or use Ctrl+C if running in the start_workers.sh terminal.

## Check Worker Status

```bash
curl http://localhost:8001/health
curl http://localhost:8002/health
```

Should both return: `{"status":"healthy","worker":"..."}`

## Troubleshooting

### If workers won't start:
```bash
# Kill existing processes
lsof -ti:8001 | xargs kill -9 2>/dev/null
lsof -ti:8002 | xargs kill -9 2>/dev/null

# Try starting again
cd medical-agent && source venv/bin/activate && ./scripts/start_workers.sh
```

### If queries fail with "connection error":
1. Make sure workers are running (check worker status above)
2. Restart workers if needed
3. Check that you're in the correct directory and venv is activated

## Most Common Command (For Quick Testing)

```bash
cd medical-agent && source venv/bin/activate && python3 scripts/query_system.py "What medications were prescribed?" --verbose
```

This is your go-to command for testing any custom query with full reasoning output!
