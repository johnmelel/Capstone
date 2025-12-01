# Baseline Comparison - Quick Start Guide

## Overview
This guide walks you through running the baseline comparison experiments for your academic paper. The entire process takes about 40-60 minutes to complete.

## Prerequisites

### 1. System Requirements
- Workers running on localhost:8001 and localhost:8002
- Google Cloud credentials configured
- Milvus Cloud connection in .env
- MIMIC-IV database populated

### 2. Verify System is Working
```bash
# Make sure you're in the medical-agent directory
cd medical-agent

# Start workers (if not already running)
./scripts/start_workers.sh

# In another terminal, test the main system
python scripts/test_reasoning_categories.py
```

If this completes successfully, you're ready to run baseline comparisons!

## Running Baseline Comparisons

### Option 1: Quick Test (Recommended First)
Test with just 3 queries to verify everything works (~10 minutes):

```bash
python scripts/compare_baselines.py --quick
```

This will:
- Test all 4 architectures (MCP, No-MCP, Single-Agent, Vector-Only)
- Run 3 queries per architecture
- Generate comparison table
- Save results to `baseline_comparison_results.json`

### Option 2: Full Comparison (For Paper)
Run complete evaluation with all 10 queries (~40 minutes):

```bash
python scripts/compare_baselines.py --orchestrator-type all
```

### Option 3: Test Individual Baselines
Test specific architectures:

```bash
# Test MCP multi-agent (control)
python scripts/compare_baselines.py --orchestrator-type mcp_multi_agent

# Test No-MCP baseline
python scripts/compare_baselines.py --orchestrator-type no_mcp

# Test Single-Agent baseline
python scripts/compare_baselines.py --orchestrator-type single_agent

# Test Vector-Only baseline
python scripts/compare_baselines.py --orchestrator-type vector_only
```

## Understanding the Output

### Console Output
You'll see real-time progress for each query:
```
Testing: MCP MULTI AGENT
==========================================

Test 1/10: 1. Diagnostic Pattern Reasoning
Query: What are the possible diagnoses for a patient with...
  Latency: 3.45s | Steps: 10 | Length: 456 chars
  Success: Yes

...

BASELINE COMPARISON RESULTS
====================================================================================================

Metric                         | MCP Multi    | No-MCP       | Single       | Vector-Only  
----------------------------------------------------------------------------------------------------
Success Rate (%)               | 100.0        | 90.0         | 80.0         | 60.0         
Avg Reasoning Steps            | 8.1          | 7.8          | 7.2          | 4.5          
Avg Latency (s)                | 3.50         | 3.80         | 3.20         | 2.10         
Avg Answer Length              | 520          | 490          | 410          | 280          
Multi-Modal Usage (%)          | 90.0         | 85.0         | 80.0         | 0.0
```

### Results File
Detailed JSON results saved to `baseline_comparison_results.json`:
- Complete query-by-query breakdown
- Timing information
- Success/failure status
- Answer previews
- All metrics per architecture

## Analyzing Results for Your Paper

### Step 1: Review Comparison Table
The console output shows key metrics side-by-side. Look for:
- Success rate differences (demonstrates reliability)
- Reasoning depth differences (demonstrates sophistication)
- Multi-modal usage (demonstrates data integration)

### Step 2: Extract Key Statistics
Open `baseline_comparison_results.json` and extract:
- Exact percentages for success rates
- Average reasoning steps per category
- Latency comparisons
- Data source utilization patterns

### Step 3: Fill in Results Template
Open `RESULTS_TEMPLATE.md` and fill in the [DATA] placeholders with your actual results.

### Step 4: Document Key Findings
For each comparison, write 2-3 sentences explaining:
- What the metric shows
- Why the difference matters
- What this demonstrates about the architecture

## Common Issues and Solutions

### Issue: Workers Not Running
**Error**: "Connection refused" or "Failed to initialize"
**Solution**:
```bash
# Start workers in separate terminal
./scripts/start_workers.sh

# Verify they're running
curl http://localhost:8001/health
curl http://localhost:8002/health
```

### Issue: No-MCP Baseline Fails
**Error**: "Milvus not connected"
**Solution**: Check your .env file has MILVUS_URI and MILVUS_TOKEN set

### Issue: Single-Agent is Slow
**Explanation**: This is expected - it's doing more work in one component. Document this as a finding!

### Issue: Vector-Only Has Low Success Rate
**Explanation**: This is also expected - it demonstrates the value of multi-modal data. This is a positive finding for your paper!

## Creating Visualizations

### Generate Comparison Charts (Python)
```python
import json
import matplotlib.pyplot as plt

# Load results
with open('baseline_comparison_results.json', 'r') as f:
    data = json.load(f)

# Extract metrics
architectures = []
reasoning_steps = []
success_rates = []

for result in data['results']:
    architectures.append(result['orchestrator_type'])
    reasoning_steps.append(result['avg_reasoning_steps'])
    success_rates.append(result['success_rate'])

# Create bar charts
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.bar(architectures, reasoning_steps)
ax1.set_title('Average Reasoning Steps by Architecture')
ax1.set_ylabel('Reasoning Steps')

ax2.bar(architectures, success_rates)
ax2.set_title('Success Rate by Architecture')
ax2.set_ylabel('Success Rate (%)')

plt.tight_layout()
plt.savefig('baseline_comparison.png')
```

## Timeline for Complete Evaluation

1. **Initial Quick Test** (10 min): Verify everything works
2. **Full Baseline Run** (40 min): Complete evaluation
3. **Results Analysis** (30 min): Fill in template, extract insights
4. **Visualization** (20 min): Create charts if needed
5. **Documentation** (30 min): Write findings for paper

**Total Time**: ~2 hours for complete baseline evaluation and analysis

## What to Include in Your Paper

### Methods Section
```
We evaluated our MCP-based multi-agent architecture against three 
strategic baselines:

1. No-MCP: Multi-agent architecture with direct database connections 
   instead of MCP protocol
2. Single-Agent: Monolithic architecture without agent specialization
3. Vector-Only: Single data source without multi-modal integration

Each system was tested on 10 queries spanning diverse medical reasoning 
categories. We measured reasoning depth (step count), success rate, 
latency, and data integration effectiveness.
```

### Results Section
Use the comparison table directly from the console output, and add narrative:
```
The MCP-based multi-agent system achieved a [X]% success rate with an 
average reasoning depth of [Y] steps. Compared to the No-MCP baseline, 
this represents a [Z]% improvement in reasoning sophistication, 
demonstrating the value of protocol-level abstraction...
```

### Discussion Section
Explain what each comparison demonstrates:
- **MCP vs No-MCP**: Value of protocol abstraction
- **Multi-Agent vs Single-Agent**: Value of specialization
- **Full vs Vector-Only**: Value of multi-modal integration

## Next Steps

After running the comparison:

1. ✓ Fill in RESULTS_TEMPLATE.md with actual data
2. ✓ Create visualizations if needed
3. ✓ Write key findings section for paper
4. ✓ Document any interesting failure cases
5. ✓ Calculate statistical significance if needed

## Questions or Issues?

- Check README.md for architecture details
- Review RESULTS_TEMPLATE.md for paper structure
- Examine baseline_comparison_results.json for raw data

Good luck with your paper!
