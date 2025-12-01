"""
Simple test to verify expanded MIMIC database with 2025 dates
"""
import sqlite3
from pathlib import Path

def test_expanded_database():
    """Test the expanded database functionality"""
    db_path = Path(__file__).parent / 'mimic.db'
    
    if not db_path.exists():
        print("FAIL: Database file not found")
        return
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    print("Testing Expanded MIMIC Database")
    print("=" * 50)
    
    # Test table existence and row counts
    tables = [
        'patients', 'admissions', 'prescriptions', 'emar', 'pharmacy',
        'labevents', 'diagnoses_icd', 'procedures_icd', 'transfers', 'microbiologyevents'
    ]
    
    for table in tables:
        try:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            status = "PASS" if count > 0 else "WARN"
            print(f"{status}: {table}: {count:,} rows")
        except sqlite3.OperationalError as e:
            print(f"FAIL: {table}: {e}")
    
    print("\nSample Lab Results with Date Transformation:")
    print("-" * 50)
    
    # Test lab events with 2025 dates
    cursor.execute("""
        SELECT 
            subject_id,
            charttime,
            value,
            valueuom,
            flag
        FROM labevents 
        WHERE charttime IS NOT NULL 
        AND charttime LIKE '2025%'
        ORDER BY charttime DESC 
        LIMIT 5
    """)
    
    lab_results = cursor.fetchall()
    if lab_results:
        print("PASS: Date transformation successful! Found 2025 dates:")
        for result in lab_results:
            print(f"  Patient {result[0]}: {result[1]} - {result[2]} {result[3]} ({result[4] or 'normal'})")
    else:
        print("WARN: No 2025 dates found in lab results")
    
    print("\nSample Diagnoses:")
    print("-" * 30)
    
    # Test diagnoses
    cursor.execute("""
        SELECT DISTINCT
            icd_code,
            COUNT(*) as patient_count
        FROM diagnoses_icd 
        WHERE icd_code IS NOT NULL
        GROUP BY icd_code
        ORDER BY patient_count DESC
        LIMIT 5
    """)
    
    diagnoses = cursor.fetchall()
    if diagnoses:
        print("PASS: Diagnosis data available:")
        for dx in diagnoses:
            print(f"  {dx[0]}: {dx[1]} patients")
    
    conn.close()
    print("\nDatabase expansion test complete!")

if __name__ == "__main__":
    test_expanded_database()
