# Part 1: MIMIC EMR SQLite Database

Complete, production-ready SQLite database for MIMIC-IV EMR data with proper indexing, dictionary tables for human-readable output, and comprehensive documentation.

## Overview

This database provides:
- **15 tables** including core clinical data and dictionary tables
- **Proper indexing** on frequently queried columns
- **Dictionary table joins** for human-readable lab names, diagnoses, and procedures
- **Date transformation** from MIMIC privacy dates (2100s) to 2025-2034
- **Foreign key relationships** for data integrity
- **Sample queries** demonstrating common EMR patterns

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Only requirement: `pandas>=2.0.0`

### 2. Set Up Database

```bash
python setup_database.py
```

This will:
- Create `mimic_emr.db` in the current directory
- Load data from `../legacy-implementation/medical-agent/MIMIC_data/data/`
- Transform dates to 2025-2034 range
- Create all indexes
- Verify data integrity

**Expected runtime:** 2-5 minutes depending on data size (labevents is large)

### 3. Verify Database

```bash
python test_database.py
```

This runs 6 test categories:
1. Database structure (all tables present)
2. Table row counts (data loaded)
3. Dictionary table integrity (coverage checks)
4. Foreign key relationships
5. Index coverage
6. Sample query execution

All tests should pass.

### 4. Explore Sample Queries

Open `sample_queries/emr_sample_queries.sql` in any SQL editor or use:

```bash
sqlite3 mimic_emr.db < sample_queries/emr_sample_queries.sql
```

## Database Structure

### Core Clinical Tables (6)
- **patients** - Demographics (subject_id is primary key)
- **admissions** - Hospital admissions (hadm_id is primary key)
- **labevents** - Laboratory results (highest query volume)
- **prescriptions** - Medication orders
- **diagnoses_icd** - ICD diagnosis codes
- **procedures_icd** - ICD procedure codes

### Dictionary Tables (3) - CRITICAL
- **d_labitems** - Lab test names (itemid → human-readable label)
- **d_icd_diagnoses** - Diagnosis descriptions (icd_code → long_title)
- **d_icd_procedures** - Procedure descriptions

**Always JOIN with these to get readable output!**

### Additional Clinical Tables (6)
- **emar** - Electronic medication administration records
- **pharmacy** - Pharmacy orders
- **transfers** - Patient location changes
- **microbiologyevents** - Culture results
- **drgcodes** - DRG severity indicators
- **services** - Service transfers

## Key Features

### 1. Dictionary Table Integration

**Bad Query (returns codes):**
```sql
SELECT itemid, valuenum FROM labevents WHERE subject_id = 10000032;
-- Returns: 50912, 15.2
```

**Good Query (returns names):**
```sql
SELECT d.label, l.valuenum, l.valueuom
FROM labevents l
JOIN d_labitems d ON l.itemid = d.itemid
WHERE l.subject_id = 10000032;
-- Returns: "Creatinine", 15.2, "mg/dL"
```

### 2. Comprehensive Indexing

Indexes on:
- All foreign keys (subject_id, hadm_id)
- Temporal columns (admittime, charttime, starttime)
- Frequently queried fields (drug, icd_code, label)
- Composite indexes for common patterns (subject_id + charttime)

### 3. Date Handling

All dates transformed from MIMIC's privacy dates:
- Original: 2100-2190 range
- Transformed: 2025-2034 range
- Format: `YYYY-MM-DD HH:MM:SS` (TEXT)
- Use SQLite date functions: `date('now', '-30 days')`

## Usage Examples

### Get Recent Labs with Names
```sql
SELECT 
    d.label as test_name,
    l.valuenum as value,
    l.valueuom as unit,
    l.charttime
FROM labevents l
JOIN d_labitems d ON l.itemid = d.itemid
WHERE l.subject_id = ?
ORDER BY l.charttime DESC
LIMIT 10;
```

### Get Diagnoses with Descriptions
```sql
SELECT 
    d.long_title as diagnosis,
    a.admittime
FROM diagnoses_icd di
JOIN d_icd_diagnoses d ON di.icd_code = d.icd_code 
    AND di.icd_version = d.icd_version
JOIN admissions a ON di.hadm_id = a.hadm_id
WHERE di.subject_id = ?
ORDER BY a.admittime DESC;
```

### Current Medications
```sql
SELECT drug, dose_val_rx, dose_unit_rx, route
FROM prescriptions
WHERE subject_id = ?
  AND (stoptime IS NULL OR stoptime >= date('now'))
ORDER BY starttime DESC;
```

## File Structure

```
Part1_SQLite_EMR/
├── setup_database.py          # Database creation and loading script
├── test_database.py            # Comprehensive test suite
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── mimic_emr.db               # Generated database (after setup)
├── schema/
│   └── SCHEMA.md              # Detailed schema documentation
└── sample_queries/
    └── emr_sample_queries.sql # 20 example queries
```

## Command Line Options

### Setup Database
```bash
# Default (creates mimic_emr.db in current directory)
python setup_database.py

# Custom database path
python setup_database.py --db-path /path/to/custom.db

# Custom data directory
python setup_database.py --data-dir /path/to/mimic/csvs/
```

### Test Database
```bash
# Default
python test_database.py

# Custom database path
python test_database.py --db-path /path/to/custom.db
```

## Expected Output

### Setup Output
```
Creating MIMIC EMR database at mimic_emr.db...
Data source: ../legacy-implementation/medical-agent/MIMIC_data/data

Creating tables...
Tables created successfully

Loading patients.csv...
  Loaded 100 rows into patients
Loading admissions.csv...
  Loaded 500 rows into admissions
Loading labevents.csv...
  Loaded 50000 rows into labevents
...

Database Verification:
============================================================
Found 15 tables:
  admissions                    :        500 rows
  d_icd_diagnoses              :      5,000 rows
  d_icd_procedures             :      1,000 rows
  d_labitems                   :        250 rows
  diagnoses_icd                :      2,500 rows
  ...
============================================================

Database setup complete!
```

### Test Output
```
======================================================================
MIMIC EMR Database Test Suite
======================================================================

Test 1: Database Structure
----------------------------------------------------------------------
  PASS: All 15 expected tables present

Test 2: Table Row Counts
----------------------------------------------------------------------
  patients                      :        100 rows
  admissions                    :        500 rows
  labevents                     :     50,000 rows
  ...
  PASS: All critical tables have data

...

======================================================================
TEST SUMMARY
======================================================================
Tests Passed: 6
Tests Failed: 0

ALL TESTS PASSED
```

## Integration with Part 2

This database is designed to be accessed via MCP Server in Part 2:

```python
# Part 2 will provide MCP server with tools like:
# - get_schema() - returns schema information
# - run_sql(query: str) - executes read-only queries
# - list_tables() - lists available tables
```

The MCP server will:
1. Connect to this SQLite database
2. Expose read-only query tools
3. Handle all database access for agents
4. Ensure agents cannot modify data

## Documentation

- **schema/SCHEMA.md** - Detailed table descriptions, column definitions, indexes, and query patterns
- **sample_queries/emr_sample_queries.sql** - 20 example queries covering common use cases
- **Code comments** - setup_database.py and test_database.py are heavily commented

## Important Notes for Agent Development

1. **Always use dictionary tables** - Join with d_labitems, d_icd_diagnoses, d_icd_procedures for human-readable output

2. **Limit results** - Use LIMIT clause to avoid overwhelming responses

3. **Handle NULLs** - Many fields can be NULL (stoptime, deathtime, etc.)

4. **Date format** - Dates are TEXT in 'YYYY-MM-DD HH:MM:SS' format

5. **Patient identifier** - Use subject_id as primary patient identifier

6. **Admission queries** - Use hadm_id to scope queries to specific admissions

7. **Read-only** - All agent access should be SELECT only (enforced by MCP server)

## Troubleshooting

### Error: Data directory not found
```
ERROR: Data directory not found: ../legacy-implementation/medical-agent/MIMIC_data/data
```
**Solution:** Run setup_database.py with `--data-dir` pointing to your MIMIC CSV location

### Warning: Missing dictionary tables
If test_database.py shows low dictionary coverage:
- Ensure d_labitems.csv, d_icd_diagnoses.csv, d_icd_procedures.csv are present
- These are CRITICAL for human-readable output

### Database locked error
SQLite can only handle one writer at a time. If you get a lock error:
- Close any open connections
- Wait a moment and retry
- Ensure no other process is accessing the database

## Next Steps

After completing Part 1, proceed to:
- **Part 2**: MCP server for SQL database access
- **Part 3**: MCP server for vector store and RL query rewriter
- **Part 4**: Worker agents (EMR + Research)
- **Part 5**: Router and planner agents
- **Part 6**: Personalization with Memento
- **Part 7**: LangSmith tracing

## License and Data

This uses MIMIC-IV demo data. Ensure you have appropriate permissions to use MIMIC data.
