"""
Test and validate MIMIC EMR database
Verifies schema, data integrity, and query functionality
"""
import sqlite3
from pathlib import Path
import sys


class DatabaseTester:
    def __init__(self, db_path='mimic_emr.db'):
        self.db_path = Path(db_path)
        self.tests_passed = 0
        self.tests_failed = 0
        
    def run_all_tests(self):
        """Run comprehensive database tests"""
        print("=" * 70)
        print("MIMIC EMR Database Test Suite")
        print("=" * 70)
        print()
        
        if not self.db_path.exists():
            print(f"ERROR: Database not found at {self.db_path}")
            print("Please run setup_database.py first")
            return False
        
        print(f"Testing database: {self.db_path.absolute()}")
        print()
        
        # Run all test categories
        self.test_database_structure()
        self.test_table_counts()
        self.test_dictionary_tables()
        self.test_foreign_keys()
        self.test_indexes()
        self.test_sample_queries()
        
        # Summary
        print()
        print("=" * 70)
        print("TEST SUMMARY")
        print("=" * 70)
        print(f"Tests Passed: {self.tests_passed}")
        print(f"Tests Failed: {self.tests_failed}")
        
        if self.tests_failed == 0:
            print("\nALL TESTS PASSED")
            return True
        else:
            print(f"\n{self.tests_failed} TEST(S) FAILED")
            return False
    
    def test_database_structure(self):
        """Test that all expected tables exist"""
        print("Test 1: Database Structure")
        print("-" * 70)
        
        expected_tables = [
            'patients', 'admissions', 'labevents', 'prescriptions',
            'diagnoses_icd', 'procedures_icd', 'd_labitems',
            'd_icd_diagnoses', 'd_icd_procedures', 'emar', 'pharmacy',
            'transfers', 'microbiologyevents', 'drgcodes', 'services'
        ]
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        actual_tables = [row[0] for row in cursor.fetchall()]
        
        missing_tables = set(expected_tables) - set(actual_tables)
        extra_tables = set(actual_tables) - set(expected_tables)
        
        if not missing_tables and not extra_tables:
            print(f"  PASS: All {len(expected_tables)} expected tables present")
            self.tests_passed += 1
        else:
            if missing_tables:
                print(f"  FAIL: Missing tables: {missing_tables}")
            if extra_tables:
                print(f"  INFO: Extra tables: {extra_tables}")
            self.tests_failed += 1
        
        conn.close()
        print()
    
    def test_table_counts(self):
        """Test that tables have data"""
        print("Test 2: Table Row Counts")
        print("-" * 70)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Critical tables that must have data
        critical_tables = [
            'patients', 'admissions', 'labevents', 'prescriptions',
            'diagnoses_icd', 'd_labitems', 'd_icd_diagnoses'
        ]
        
        all_passed = True
        for table in critical_tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            
            if count > 0:
                print(f"  {table:30s}: {count:>10,} rows")
            else:
                print(f"  FAIL: {table} has no data")
                all_passed = False
        
        if all_passed:
            print("  PASS: All critical tables have data")
            self.tests_passed += 1
        else:
            self.tests_failed += 1
        
        conn.close()
        print()
    
    def test_dictionary_tables(self):
        """Test that dictionary tables are properly populated"""
        print("Test 3: Dictionary Table Integrity")
        print("-" * 70)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        tests = []
        
        # Test 3a: d_labitems coverage
        cursor.execute("""
            SELECT COUNT(DISTINCT l.itemid) as used_items,
                   COUNT(DISTINCT d.itemid) as defined_items
            FROM labevents l
            LEFT JOIN d_labitems d ON l.itemid = d.itemid
        """)
        used, defined = cursor.fetchone()
        coverage = (defined / used * 100) if used > 0 else 0
        
        print(f"  d_labitems coverage: {defined}/{used} ({coverage:.1f}%)")
        tests.append(coverage > 90)  # At least 90% coverage
        
        # Test 3b: d_icd_diagnoses coverage
        cursor.execute("""
            SELECT COUNT(DISTINCT di.icd_code) as used_codes,
                   COUNT(DISTINCT d.icd_code) as defined_codes
            FROM diagnoses_icd di
            LEFT JOIN d_icd_diagnoses d ON di.icd_code = d.icd_code 
                AND di.icd_version = d.icd_version
        """)
        used, defined = cursor.fetchone()
        coverage = (defined / used * 100) if used > 0 else 0
        
        print(f"  d_icd_diagnoses coverage: {defined}/{used} ({coverage:.1f}%)")
        tests.append(coverage > 90)
        
        # Test 3c: Dictionary entries have descriptions
        cursor.execute("SELECT COUNT(*) FROM d_labitems WHERE label IS NULL")
        null_labels = cursor.fetchone()[0]
        print(f"  d_labitems with NULL labels: {null_labels}")
        tests.append(null_labels == 0)
        
        if all(tests):
            print("  PASS: Dictionary tables properly populated")
            self.tests_passed += 1
        else:
            print("  FAIL: Dictionary table issues detected")
            self.tests_failed += 1
        
        conn.close()
        print()
    
    def test_foreign_keys(self):
        """Test foreign key relationships"""
        print("Test 4: Foreign Key Integrity")
        print("-" * 70)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        tests = []
        
        # Test 4a: All admissions have valid patient references
        cursor.execute("""
            SELECT COUNT(*) 
            FROM admissions a
            LEFT JOIN patients p ON a.subject_id = p.subject_id
            WHERE p.subject_id IS NULL
        """)
        orphaned = cursor.fetchone()[0]
        print(f"  Admissions without patient: {orphaned}")
        tests.append(orphaned == 0)
        
        # Test 4b: All labevents have valid patient references
        cursor.execute("""
            SELECT COUNT(*) 
            FROM labevents l
            LEFT JOIN patients p ON l.subject_id = p.subject_id
            WHERE p.subject_id IS NULL
            LIMIT 1000
        """)
        orphaned = cursor.fetchone()[0]
        print(f"  Lab events without patient (sample): {orphaned}")
        tests.append(orphaned == 0)
        
        if all(tests):
            print("  PASS: Foreign key integrity verified")
            self.tests_passed += 1
        else:
            print("  FAIL: Foreign key integrity issues")
            self.tests_failed += 1
        
        conn.close()
        print()
    
    def test_indexes(self):
        """Test that indexes exist"""
        print("Test 5: Index Coverage")
        print("-" * 70)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check for key indexes
        expected_indexes = [
            'idx_labevents_subject_id',
            'idx_labevents_charttime',
            'idx_prescriptions_drug',
            'idx_admissions_admittime',
            'd_labitems_label'
        ]
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='index'")
        actual_indexes = [row[0] for row in cursor.fetchall()]
        
        found = 0
        for idx in expected_indexes:
            if idx in actual_indexes:
                found += 1
        
        print(f"  Key indexes found: {found}/{len(expected_indexes)}")
        
        if found >= len(expected_indexes) * 0.8:  # At least 80% coverage
            print("  PASS: Index coverage adequate")
            self.tests_passed += 1
        else:
            print("  FAIL: Missing critical indexes")
            self.tests_failed += 1
        
        conn.close()
        print()
    
    def test_sample_queries(self):
        """Test that sample queries execute successfully"""
        print("Test 6: Sample Query Execution")
        print("-" * 70)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get a sample patient ID
        cursor.execute("SELECT subject_id FROM patients LIMIT 1")
        result = cursor.fetchone()
        if not result:
            print("  FAIL: No patient data to test queries")
            self.tests_failed += 1
            conn.close()
            return
        
        sample_patient = result[0]
        tests = []
        
        # Query 1: Patient demographics
        try:
            cursor.execute("""
                SELECT p.subject_id, p.gender, COUNT(a.hadm_id) as admissions
                FROM patients p
                LEFT JOIN admissions a ON p.subject_id = a.subject_id
                WHERE p.subject_id = ?
                GROUP BY p.subject_id
            """, (sample_patient,))
            result = cursor.fetchone()
            print(f"  Patient query: {result}")
            tests.append(True)
        except Exception as e:
            print(f"  FAIL: Patient query error: {e}")
            tests.append(False)
        
        # Query 2: Labs with dictionary join
        try:
            cursor.execute("""
                SELECT d.label, l.valuenum, l.valueuom
                FROM labevents l
                JOIN d_labitems d ON l.itemid = d.itemid
                WHERE l.subject_id = ?
                  AND l.valuenum IS NOT NULL
                ORDER BY l.charttime DESC
                LIMIT 5
            """, (sample_patient,))
            results = cursor.fetchall()
            print(f"  Lab query returned: {len(results)} results")
            tests.append(True)
        except Exception as e:
            print(f"  FAIL: Lab query error: {e}")
            tests.append(False)
        
        # Query 3: Diagnoses with dictionary join
        try:
            cursor.execute("""
                SELECT d.long_title
                FROM diagnoses_icd di
                JOIN d_icd_diagnoses d ON di.icd_code = d.icd_code 
                    AND di.icd_version = d.icd_version
                WHERE di.subject_id = ?
                LIMIT 5
            """, (sample_patient,))
            results = cursor.fetchall()
            print(f"  Diagnosis query returned: {len(results)} results")
            tests.append(True)
        except Exception as e:
            print(f"  FAIL: Diagnosis query error: {e}")
            tests.append(False)
        
        if all(tests):
            print("  PASS: All sample queries executed successfully")
            self.tests_passed += 1
        else:
            print("  FAIL: Some queries failed")
            self.tests_failed += 1
        
        conn.close()
        print()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Test MIMIC EMR database')
    parser.add_argument('--db-path', default='mimic_emr.db', 
                       help='Path to database file')
    
    args = parser.parse_args()
    
    tester = DatabaseTester(args.db_path)
    success = tester.run_all_tests()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
