-- Sample EMR Queries for MIMIC Database
-- These queries demonstrate common patterns for EMR agent use cases

-- ============================================================================
-- PATIENT DEMOGRAPHICS
-- ============================================================================

-- Query 1: Get patient basic info
SELECT 
    p.subject_id,
    p.gender,
    p.anchor_age as age_at_anchor,
    p.anchor_year,
    p.dod as date_of_death,
    COUNT(DISTINCT a.hadm_id) as total_admissions
FROM patients p
LEFT JOIN admissions a ON p.subject_id = a.subject_id
WHERE p.subject_id = 10000032
GROUP BY p.subject_id;

-- Query 2: Get all admissions for a patient with timeline
SELECT 
    hadm_id,
    admittime,
    dischtime,
    admission_type,
    admission_location,
    discharge_location,
    hospital_expire_flag,
    ROUND((julianday(dischtime) - julianday(admittime)), 1) as length_of_stay_days
FROM admissions
WHERE subject_id = 10000032
ORDER BY admittime DESC;

-- ============================================================================
-- LABORATORY RESULTS (WITH HUMAN-READABLE NAMES)
-- ============================================================================

-- Query 3: Recent lab results with test names
SELECT 
    d.label as test_name,
    d.fluid,
    d.category,
    l.valuenum as value,
    l.valueuom as unit,
    l.ref_range_lower,
    l.ref_range_upper,
    l.flag as abnormal_flag,
    l.charttime
FROM labevents l
JOIN d_labitems d ON l.itemid = d.itemid
WHERE l.subject_id = 10000032
  AND l.valuenum IS NOT NULL
ORDER BY l.charttime DESC
LIMIT 20;

-- Query 4: Get specific lab test (e.g., Creatinine) over time
SELECT 
    d.label as test_name,
    l.valuenum as value,
    l.valueuom as unit,
    l.charttime,
    CASE 
        WHEN l.valuenum > l.ref_range_upper THEN 'HIGH'
        WHEN l.valuenum < l.ref_range_lower THEN 'LOW'
        ELSE 'NORMAL'
    END as status
FROM labevents l
JOIN d_labitems d ON l.itemid = d.itemid
WHERE l.subject_id = 10000032
  AND d.label LIKE '%Creatinine%'
  AND l.valuenum IS NOT NULL
ORDER BY l.charttime DESC
LIMIT 10;

-- Query 5: All abnormal labs in last 7 days
SELECT 
    d.label as test_name,
    l.valuenum as value,
    l.valueuom as unit,
    l.ref_range_lower,
    l.ref_range_upper,
    l.flag,
    l.charttime
FROM labevents l
JOIN d_labitems d ON l.itemid = d.itemid
WHERE l.subject_id = 10000032
  AND l.flag IS NOT NULL
  AND l.charttime >= date('now', '-7 days')
ORDER BY l.charttime DESC;

-- Query 6: Lab results by category (e.g., Chemistry)
SELECT 
    d.label as test_name,
    d.category,
    l.valuenum as value,
    l.valueuom as unit,
    l.charttime
FROM labevents l
JOIN d_labitems d ON l.itemid = d.itemid
WHERE l.subject_id = 10000032
  AND d.category = 'Chemistry'
  AND l.valuenum IS NOT NULL
ORDER BY l.charttime DESC
LIMIT 15;

-- ============================================================================
-- DIAGNOSES (WITH HUMAN-READABLE DESCRIPTIONS)
-- ============================================================================

-- Query 7: All diagnoses with descriptions
SELECT 
    a.admittime as admission_date,
    d.long_title as diagnosis,
    di.seq_num as sequence,
    di.icd_code,
    di.icd_version
FROM diagnoses_icd di
JOIN d_icd_diagnoses d ON di.icd_code = d.icd_code 
    AND di.icd_version = d.icd_version
JOIN admissions a ON di.hadm_id = a.hadm_id
WHERE di.subject_id = 10000032
ORDER BY a.admittime DESC, di.seq_num;

-- Query 8: Primary diagnoses (seq_num = 1) across admissions
SELECT 
    a.admittime,
    a.dischtime,
    d.long_title as primary_diagnosis,
    a.admission_type
FROM diagnoses_icd di
JOIN d_icd_diagnoses d ON di.icd_code = d.icd_code 
    AND di.icd_version = d.icd_version
JOIN admissions a ON di.hadm_id = a.hadm_id
WHERE di.subject_id = 10000032
  AND di.seq_num = 1
ORDER BY a.admittime DESC;

-- Query 9: Check for specific condition (e.g., diabetes)
SELECT DISTINCT
    d.long_title as diagnosis,
    di.icd_code,
    a.admittime
FROM diagnoses_icd di
JOIN d_icd_diagnoses d ON di.icd_code = d.icd_code 
    AND di.icd_version = d.icd_version
JOIN admissions a ON di.hadm_id = a.hadm_id
WHERE di.subject_id = 10000032
  AND d.long_title LIKE '%Diabetes%'
ORDER BY a.admittime DESC;

-- ============================================================================
-- MEDICATIONS
-- ============================================================================

-- Query 10: Current medications (active or recently stopped)
SELECT 
    drug,
    dose_val_rx,
    dose_unit_rx,
    route,
    doses_per_24_hrs,
    starttime,
    stoptime,
    CASE 
        WHEN stoptime IS NULL THEN 'Active'
        WHEN stoptime >= date('now') THEN 'Active'
        ELSE 'Stopped'
    END as status
FROM prescriptions
WHERE subject_id = 10000032
  AND (stoptime IS NULL OR stoptime >= date('now', '-30 days'))
ORDER BY starttime DESC;

-- Query 11: Medication history for specific drug class
SELECT 
    drug,
    dose_val_rx || ' ' || dose_unit_rx as dose,
    route,
    starttime,
    stoptime
FROM prescriptions
WHERE subject_id = 10000032
  AND drug LIKE '%Insulin%'
ORDER BY starttime DESC;

-- Query 12: All medications during specific admission
SELECT 
    p.drug,
    p.dose_val_rx,
    p.dose_unit_rx,
    p.route,
    p.starttime,
    p.stoptime
FROM prescriptions p
JOIN admissions a ON p.hadm_id = a.hadm_id
WHERE p.subject_id = 10000032
  AND a.hadm_id = 20000001
ORDER BY p.starttime;

-- ============================================================================
-- PROCEDURES
-- ============================================================================

-- Query 13: Procedures with descriptions
SELECT 
    d.long_title as procedure,
    pi.chartdate,
    pi.icd_code,
    a.admittime
FROM procedures_icd pi
JOIN d_icd_procedures d ON pi.icd_code = d.icd_code 
    AND pi.icd_version = d.icd_version
JOIN admissions a ON pi.hadm_id = a.hadm_id
WHERE pi.subject_id = 10000032
ORDER BY pi.chartdate DESC;

-- ============================================================================
-- COMPREHENSIVE PATIENT SUMMARY
-- ============================================================================

-- Query 14: Patient summary with latest admission info
SELECT 
    p.subject_id,
    p.gender,
    p.anchor_age,
    a.hadm_id as latest_admission_id,
    a.admittime as latest_admission,
    a.dischtime as latest_discharge,
    a.admission_type,
    COUNT(DISTINCT l.labevent_id) as lab_count,
    COUNT(DISTINCT pr.pharmacy_id) as medication_count,
    COUNT(DISTINCT di.icd_code) as diagnosis_count
FROM patients p
LEFT JOIN admissions a ON p.subject_id = a.subject_id
LEFT JOIN labevents l ON a.hadm_id = l.hadm_id
LEFT JOIN prescriptions pr ON a.hadm_id = pr.hadm_id
LEFT JOIN diagnoses_icd di ON a.hadm_id = di.hadm_id
WHERE p.subject_id = 10000032
  AND a.hadm_id = (
      SELECT hadm_id 
      FROM admissions 
      WHERE subject_id = 10000032 
      ORDER BY admittime DESC 
      LIMIT 1
  )
GROUP BY p.subject_id, a.hadm_id;

-- ============================================================================
-- CARE TRAJECTORY
-- ============================================================================

-- Query 15: Patient location history (transfers)
SELECT 
    eventtype,
    careunit,
    intime,
    outtime,
    ROUND((julianday(outtime) - julianday(intime)) * 24, 1) as hours_in_unit
FROM transfers
WHERE subject_id = 10000032
ORDER BY intime;

-- Query 16: Service transfers during admissions
SELECT 
    s.transfertime,
    s.prev_service,
    s.curr_service,
    a.admittime,
    a.admission_type
FROM services s
JOIN admissions a ON s.hadm_id = a.hadm_id
WHERE s.subject_id = 10000032
ORDER BY s.transfertime;

-- ============================================================================
-- MICROBIOLOGY (INFECTIONS)
-- ============================================================================

-- Query 17: Positive cultures and organisms
SELECT 
    charttime,
    spec_type_desc as specimen_type,
    test_name,
    org_name as organism,
    ab_name as antibiotic,
    interpretation
FROM microbiologyevents
WHERE subject_id = 10000032
  AND org_name IS NOT NULL
ORDER BY charttime DESC;

-- ============================================================================
-- SEVERITY AND COMPLEXITY
-- ============================================================================

-- Query 18: DRG codes for admissions (severity indicators)
SELECT 
    a.admittime,
    a.dischtime,
    d.drg_type,
    d.drg_code,
    d.description,
    d.drg_severity,
    d.drg_mortality
FROM drgcodes d
JOIN admissions a ON d.hadm_id = a.hadm_id
WHERE d.subject_id = 10000032
ORDER BY a.admittime DESC;

-- ============================================================================
-- TEMPORAL QUERIES
-- ============================================================================

-- Query 19: Recent activity (last 30 days)
SELECT 
    'Lab' as event_type,
    d.label as description,
    l.charttime as event_time
FROM labevents l
JOIN d_labitems d ON l.itemid = d.itemid
WHERE l.subject_id = 10000032
  AND l.charttime >= date('now', '-30 days')

UNION ALL

SELECT 
    'Prescription',
    drug,
    starttime
FROM prescriptions
WHERE subject_id = 10000032
  AND starttime >= date('now', '-30 days')

ORDER BY event_time DESC
LIMIT 50;

-- Query 20: Lab trends - Compare first and last values
WITH lab_trends AS (
    SELECT 
        d.label as test_name,
        l.valuenum as value,
        l.valueuom as unit,
        l.charttime,
        ROW_NUMBER() OVER (PARTITION BY d.label ORDER BY l.charttime DESC) as rn_latest,
        ROW_NUMBER() OVER (PARTITION BY d.label ORDER BY l.charttime ASC) as rn_earliest
    FROM labevents l
    JOIN d_labitems d ON l.itemid = d.itemid
    WHERE l.subject_id = 10000032
      AND l.valuenum IS NOT NULL
)
SELECT 
    test_name,
    MAX(CASE WHEN rn_earliest = 1 THEN value END) as first_value,
    MAX(CASE WHEN rn_earliest = 1 THEN charttime END) as first_date,
    MAX(CASE WHEN rn_latest = 1 THEN value END) as latest_value,
    MAX(CASE WHEN rn_latest = 1 THEN charttime END) as latest_date,
    MAX(CASE WHEN rn_latest = 1 THEN value END) - 
    MAX(CASE WHEN rn_earliest = 1 THEN value END) as change,
    unit
FROM lab_trends
WHERE rn_latest = 1 OR rn_earliest = 1
GROUP BY test_name, unit
HAVING COUNT(DISTINCT charttime) > 1
ORDER BY test_name;

-- ============================================================================
-- NOTES FOR AGENTS
-- ============================================================================

-- IMPORTANT REMINDERS:
-- 1. Always JOIN with dictionary tables (d_labitems, d_icd_diagnoses, d_icd_procedures)
-- 2. Use LIMIT to avoid overwhelming results
-- 3. Check for NULL values in optional fields (stoptime, deathtime, etc.)
-- 4. Dates are TEXT in 'YYYY-MM-DD HH:MM:SS' format
-- 5. Use subject_id as primary patient identifier
-- 6. Use hadm_id for admission-specific queries
