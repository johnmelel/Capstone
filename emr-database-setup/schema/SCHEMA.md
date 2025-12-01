# MIMIC EMR Database Schema

This document describes the SQLite database schema for the MIMIC-IV EMR subset.

## Overview

The database contains **15 tables** organized into three categories:

1. **Core Clinical Tables** (6) - Primary patient data
2. **Dictionary Tables** (3) - Code-to-description mappings (CRITICAL for readable output)
3. **Additional Clinical Tables** (6) - Supplementary data

All dates have been transformed from MIMIC's privacy dates (2100s) to 2025-2034 range.

---

## Core Clinical Tables

### patients
Patient demographics and basic information.

**Columns:**
- `subject_id` (INTEGER, PRIMARY KEY) - Unique patient identifier
- `gender` (TEXT) - M/F
- `anchor_age` (INTEGER) - Age at anchor year
- `anchor_year` (INTEGER) - Reference year for age calculation
- `anchor_year_group` (TEXT) - Year range group
- `dod` (TEXT) - Date of death (if applicable)

**Indexes:**
- `idx_patients_gender` on (gender)

**Typical Use:** Get patient demographics, calculate current age, check survival status

---

### admissions
Hospital admission records with admission/discharge details.

**Columns:**
- `subject_id` (INTEGER, FOREIGN KEY → patients)
- `hadm_id` (INTEGER, PRIMARY KEY) - Unique hospital admission ID
- `admittime` (TEXT) - Admission timestamp
- `dischtime` (TEXT) - Discharge timestamp
- `deathtime` (TEXT) - Death timestamp (if died during admission)
- `admission_type` (TEXT) - EMERGENCY, ELECTIVE, etc.
- `admission_location` (TEXT) - Where admitted from
- `discharge_location` (TEXT) - Where discharged to
- `insurance` (TEXT) - Insurance type
- `language` (TEXT) - Primary language
- `marital_status` (TEXT)
- `race` (TEXT)
- `edregtime` (TEXT) - ED registration time
- `edouttime` (TEXT) - ED discharge time
- `hospital_expire_flag` (INTEGER) - 1 if died in hospital, 0 otherwise

**Indexes:**
- `idx_admissions_subject_id` on (subject_id)
- `idx_admissions_admittime` on (admittime)
- `idx_admissions_type` on (admission_type)

**Typical Use:** Find admissions for a patient, get admission timeline, analyze admission patterns

---

### labevents
Laboratory test results - highest query volume table.

**Columns:**
- `labevent_id` (INTEGER, PRIMARY KEY)
- `subject_id` (INTEGER, FOREIGN KEY → patients)
- `hadm_id` (INTEGER, FOREIGN KEY → admissions)
- `specimen_id` (INTEGER) - Specimen identifier
- `itemid` (INTEGER, FOREIGN KEY → d_labitems) - Lab test type
- `charttime` (TEXT) - When result was charted
- `storetime` (TEXT) - When result was stored
- `value` (TEXT) - Text value of result
- `valuenum` (REAL) - Numeric value of result
- `valueuom` (TEXT) - Unit of measurement
- `ref_range_lower` (REAL) - Lower reference range
- `ref_range_upper` (REAL) - Upper reference range
- `flag` (TEXT) - Abnormal flag
- `priority` (TEXT) - Test priority
- `comments` (TEXT)

**Indexes:**
- `idx_labevents_subject_id` on (subject_id)
- `idx_labevents_hadm_id` on (hadm_id)
- `idx_labevents_charttime` on (charttime)
- `idx_labevents_itemid` on (itemid)
- `idx_labevents_subject_charttime` on (subject_id, charttime) - Composite for temporal queries

**Typical Use:** Get recent labs for patient, track lab trends over time, identify abnormal results

**IMPORTANT:** Always JOIN with d_labitems to get human-readable test names!

---

### prescriptions
Medication prescriptions.

**Columns:**
- `subject_id` (INTEGER, FOREIGN KEY → patients)
- `hadm_id` (INTEGER, FOREIGN KEY → admissions)
- `pharmacy_id` (INTEGER)
- `starttime` (TEXT) - Prescription start time
- `stoptime` (TEXT) - Prescription stop time
- `drug_type` (TEXT) - MAIN, ADDITIVE, etc.
- `drug` (TEXT) - Drug name
- `prod_strength` (TEXT) - Product strength
- `dose_val_rx` (TEXT) - Dose value
- `dose_unit_rx` (TEXT) - Dose unit
- `route` (TEXT) - Administration route
- `doses_per_24_hrs` (REAL)

**Indexes:**
- `idx_prescriptions_subject_id` on (subject_id)
- `idx_prescriptions_hadm_id` on (hadm_id)
- `idx_prescriptions_drug` on (drug)
- `idx_prescriptions_starttime` on (starttime)

**Typical Use:** Get current medications, medication history, check for specific drugs

---

### diagnoses_icd
ICD diagnosis codes for each admission.

**Columns:**
- `subject_id` (INTEGER, FOREIGN KEY → patients)
- `hadm_id` (INTEGER, FOREIGN KEY → admissions)
- `seq_num` (INTEGER) - Sequence/priority of diagnosis
- `icd_code` (TEXT) - ICD code (9 or 10)
- `icd_version` (INTEGER) - 9 or 10

**Indexes:**
- `idx_diagnoses_icd_subject_id` on (subject_id)
- `idx_diagnoses_icd_hadm_id` on (hadm_id)
- `idx_diagnoses_icd_code` on (icd_code)

**Typical Use:** Get diagnoses for admission, find patients with specific conditions

**IMPORTANT:** Always JOIN with d_icd_diagnoses to get human-readable diagnosis descriptions!

---

### procedures_icd
ICD procedure codes for procedures performed.

**Columns:**
- `subject_id` (INTEGER, FOREIGN KEY → patients)
- `hadm_id` (INTEGER, FOREIGN KEY → admissions)
- `seq_num` (INTEGER) - Sequence number
- `chartdate` (TEXT) - Date procedure was performed
- `icd_code` (TEXT) - ICD procedure code
- `icd_version` (INTEGER) - 9 or 10

**Indexes:**
- `idx_procedures_icd_subject_id` on (subject_id)
- `idx_procedures_icd_hadm_id` on (hadm_id)
- `idx_procedures_icd_code` on (icd_code)

**Typical Use:** Get procedures performed during admission

**IMPORTANT:** Always JOIN with d_icd_procedures to get human-readable procedure descriptions!

---

## Dictionary Tables (CRITICAL)

These tables map codes to human-readable descriptions. **Always use these in JOINs** to provide meaningful agent responses.

### d_labitems
Dictionary of laboratory test definitions.

**Columns:**
- `itemid` (INTEGER, PRIMARY KEY) - Lab test identifier
- `label` (TEXT) - Human-readable test name (e.g., "Creatinine")
- `fluid` (TEXT) - Fluid type (Blood, Urine, etc.)
- `category` (TEXT) - Test category (Chemistry, Hematology, etc.)

**Indexes:**
- `idx_d_labitems_label` on (label)

**Example:**
```sql
-- BAD: Returns itemid numbers
SELECT itemid, valuenum FROM labevents WHERE subject_id = 10000032;

-- GOOD: Returns test names
SELECT d.label, l.valuenum, l.valueuom
FROM labevents l
JOIN d_labitems d ON l.itemid = d.itemid
WHERE l.subject_id = 10000032;
```

---

### d_icd_diagnoses
Dictionary of ICD diagnosis codes and descriptions.

**Columns:**
- `icd_code` (TEXT, PRIMARY KEY)
- `icd_version` (INTEGER, PRIMARY KEY) - 9 or 10
- `long_title` (TEXT) - Full diagnosis description

**Example:**
```sql
-- BAD: Returns just codes
SELECT icd_code FROM diagnoses_icd WHERE subject_id = 10000032;

-- GOOD: Returns descriptions
SELECT d.long_title, di.seq_num
FROM diagnoses_icd di
JOIN d_icd_diagnoses d ON di.icd_code = d.icd_code AND di.icd_version = d.icd_version
WHERE di.subject_id = 10000032
ORDER BY di.seq_num;
```

---

### d_icd_procedures
Dictionary of ICD procedure codes and descriptions.

**Columns:**
- `icd_code` (TEXT, PRIMARY KEY)
- `icd_version` (INTEGER, PRIMARY KEY)
- `long_title` (TEXT) - Full procedure description

---

## Additional Clinical Tables

### emar
Electronic Medication Administration Record - tracks when medications were actually given.

**Columns:**
- `subject_id` (INTEGER, FOREIGN KEY → patients)
- `hadm_id` (INTEGER, FOREIGN KEY → admissions)
- `emar_id` (TEXT, PRIMARY KEY)
- `charttime` (TEXT) - When medication was charted
- `medication` (TEXT) - Medication name
- `event_txt` (TEXT) - Event description (Given, Missed, etc.)
- `scheduletime` (TEXT)
- `storetime` (TEXT)

**Indexes:**
- `idx_emar_subject_id`, `idx_emar_medication`, `idx_emar_charttime`

---

### pharmacy
Pharmacy orders and dispensing information.

**Columns:**
- `subject_id`, `hadm_id`, `pharmacy_id` (PRIMARY KEY)
- `starttime`, `stoptime`
- `medication`, `route`, `frequency`, `status`

**Indexes:**
- `idx_pharmacy_subject_id`, `idx_pharmacy_medication`

---

### transfers
Patient transfers between care units.

**Columns:**
- `subject_id`, `hadm_id`, `transfer_id` (PRIMARY KEY)
- `eventtype` (TEXT) - admit, transfer, discharge
- `careunit` (TEXT) - Care unit name
- `intime`, `outtime` (TEXT)

**Indexes:**
- `idx_transfers_subject_id`, `idx_transfers_intime`

**Typical Use:** Track patient location over time, ICU stays

---

### microbiologyevents
Microbiology culture results and antibiotic susceptibility.

**Columns:**
- `microevent_id` (PRIMARY KEY)
- `subject_id`, `hadm_id`
- `charttime`, `spec_type_desc` (specimen type)
- `test_name`, `org_name` (organism)
- `ab_name` (antibiotic), `interpretation` (S/R/I)

**Indexes:**
- `idx_micro_subject_id`, `idx_micro_charttime`

**Typical Use:** Check for infections, antibiotic resistance

---

### drgcodes
DRG (Diagnosis Related Group) codes for severity and billing.

**Columns:**
- `subject_id`, `hadm_id`
- `drg_type`, `drg_code`, `description`
- `drg_severity`, `drg_mortality` (1-4 scale)

**Typical Use:** Understand admission complexity and severity

---

### services
Service transfers during hospitalization (Medicine, Surgery, etc.).

**Columns:**
- `subject_id`, `hadm_id`
- `transfertime`
- `prev_service`, `curr_service`

**Typical Use:** Track care team changes

---

## Query Patterns for Agents

### Pattern 1: Get Patient Overview
```sql
SELECT 
    p.subject_id,
    p.gender,
    p.anchor_age,
    a.hadm_id,
    a.admittime,
    a.admission_type
FROM patients p
LEFT JOIN admissions a ON p.subject_id = a.subject_id
WHERE p.subject_id = ?
ORDER BY a.admittime DESC;
```

### Pattern 2: Get Recent Labs (with names!)
```sql
SELECT 
    d.label as test_name,
    l.valuenum,
    l.valueuom as unit,
    l.ref_range_lower,
    l.ref_range_upper,
    l.flag,
    l.charttime
FROM labevents l
JOIN d_labitems d ON l.itemid = d.itemid
WHERE l.subject_id = ?
  AND l.charttime >= date('now', '-30 days')
ORDER BY l.charttime DESC
LIMIT 20;
```

### Pattern 3: Get Diagnoses (with descriptions!)
```sql
SELECT 
    a.admittime,
    d.long_title as diagnosis,
    di.seq_num
FROM diagnoses_icd di
JOIN d_icd_diagnoses d ON di.icd_code = d.icd_code 
    AND di.icd_version = d.icd_version
JOIN admissions a ON di.hadm_id = a.hadm_id
WHERE di.subject_id = ?
ORDER BY a.admittime DESC, di.seq_num;
```

### Pattern 4: Current Medications
```sql
SELECT 
    drug,
    dose_val_rx,
    dose_unit_rx,
    route,
    starttime,
    stoptime
FROM prescriptions
WHERE subject_id = ?
  AND (stoptime IS NULL OR stoptime >= date('now'))
ORDER BY starttime DESC;
```

### Pattern 5: Abnormal Labs
```sql
SELECT 
    d.label,
    l.valuenum,
    l.valueuom,
    l.ref_range_lower,
    l.ref_range_upper,
    l.charttime
FROM labevents l
JOIN d_labitems d ON l.itemid = d.itemid
WHERE l.subject_id = ?
  AND l.flag IS NOT NULL
  AND l.charttime >= date('now', '-7 days')
ORDER BY l.charttime DESC;
```

---

## Key Points for Agent Development

1. **Always JOIN with dictionary tables** (d_labitems, d_icd_diagnoses, d_icd_procedures) to get human-readable output

2. **Use proper date filtering** - Dates are stored as TEXT in 'YYYY-MM-DD HH:MM:SS' format

3. **Check for NULL values** - Many fields can be NULL (stoptime, deathtime, etc.)

4. **Limit result sets** - Use LIMIT clause to avoid overwhelming responses

5. **Use indexes** - The database has comprehensive indexes on frequently queried columns

6. **Foreign key relationships** - Always respect the relationships:
   - admissions → patients (via subject_id)
   - labevents → admissions (via hadm_id)
   - labevents → d_labitems (via itemid)
   - diagnoses_icd → d_icd_diagnoses (via icd_code + icd_version)

7. **Read-only access** - This database should only be queried with SELECT statements

---

## Example Agent Queries

**Q: "What were the last lab results for patient 10000032?"**

```sql
SELECT 
    d.label as test,
    l.valuenum as value,
    l.valueuom as unit,
    l.charttime as date
FROM labevents l
JOIN d_labitems d ON l.itemid = d.itemid
WHERE l.subject_id = 10000032
ORDER BY l.charttime DESC
LIMIT 10;
```

**Q: "What are patient 10000032's diagnoses?"**

```sql
SELECT 
    d.long_title as diagnosis,
    a.admittime as admission_date
FROM diagnoses_icd di
JOIN d_icd_diagnoses d ON di.icd_code = d.icd_code 
    AND di.icd_version = d.icd_version
JOIN admissions a ON di.hadm_id = a.hadm_id
WHERE di.subject_id = 10000032
ORDER BY a.admittime DESC, di.seq_num;
```

**Q: "Is patient 10000032 on any medications?"**

```sql
SELECT 
    drug,
    dose_val_rx || ' ' || dose_unit_rx as dose,
    route,
    starttime
FROM prescriptions
WHERE subject_id = 10000032
  AND (stoptime IS NULL OR stoptime >= date('now'))
ORDER BY starttime DESC;
```
