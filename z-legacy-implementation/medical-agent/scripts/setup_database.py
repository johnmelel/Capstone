"""
Setup SQLite database with MIMIC-IV data from CSVs
Updated to load all available tables and transform dates to 2025
"""
import sqlite3
import pandas as pd
import os
from pathlib import Path
from datetime import datetime, timedelta
import re


class MIMICDatabaseSetup:
    def __init__(self, database_path=None):
        if database_path is None:
            # Default to mimic.db in current directory
            self.database_path = Path(__file__).parent.parent / 'mimic.db'
        else:
            self.database_path = Path(database_path)
        self.data_dir = Path(__file__).parent.parent / 'MIMIC_data' / 'data'
        
    def create_database(self):
        """Create database and tables"""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        # Create all tables
        self.create_patients_table(cursor)
        self.create_admissions_table(cursor)
        self.create_prescriptions_table(cursor)
        self.create_emar_table(cursor)
        self.create_pharmacy_table(cursor)
        self.create_labevents_table(cursor)
        self.create_diagnoses_icd_table(cursor)
        self.create_procedures_icd_table(cursor)
        self.create_transfers_table(cursor)
        self.create_microbiologyevents_table(cursor)
        
        conn.commit()
        conn.close()
        
    def create_patients_table(self, cursor):
        """Create patients table"""
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS patients (
                subject_id INTEGER PRIMARY KEY,
                gender TEXT NOT NULL,
                anchor_age INTEGER NOT NULL,
                anchor_year INTEGER NOT NULL,
                anchor_year_group TEXT NOT NULL,
                dod TEXT
            )
        """)
        
    def create_admissions_table(self, cursor):
        """Create admissions table"""
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS admissions (
                subject_id INTEGER NOT NULL,
                hadm_id INTEGER PRIMARY KEY,
                admittime TEXT NOT NULL,
                dischtime TEXT,
                deathtime TEXT,
                admission_type TEXT NOT NULL,
                admit_provider_id TEXT,
                admission_location TEXT,
                discharge_location TEXT,
                insurance TEXT,
                language TEXT,
                marital_status TEXT,
                race TEXT,
                edregtime TEXT,
                edouttime TEXT,
                hospital_expire_flag INTEGER
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_admissions_subject_id ON admissions(subject_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_admissions_admittime ON admissions(admittime)")
        
    def create_prescriptions_table(self, cursor):
        """Create prescriptions table"""
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS prescriptions (
                subject_id INTEGER NOT NULL,
                hadm_id INTEGER NOT NULL,
                pharmacy_id INTEGER,
                poe_id TEXT,
                poe_seq INTEGER,
                order_provider_id TEXT,
                starttime TEXT,
                stoptime TEXT,
                drug_type TEXT,
                drug TEXT,
                formulary_drug_cd TEXT,
                gsn TEXT,
                ndc TEXT,
                prod_strength TEXT,
                form_rx TEXT,
                dose_val_rx TEXT,
                dose_unit_rx TEXT,
                form_val_disp TEXT,
                form_unit_disp TEXT,
                doses_per_24_hrs REAL,
                route TEXT
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_prescriptions_subject_id ON prescriptions(subject_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_prescriptions_drug ON prescriptions(drug)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_prescriptions_starttime ON prescriptions(starttime)")
        
    def create_emar_table(self, cursor):
        """Create emar table"""
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS emar (
                subject_id INTEGER NOT NULL,
                hadm_id INTEGER,
                emar_id TEXT PRIMARY KEY,
                emar_seq INTEGER,
                poe_id TEXT,
                pharmacy_id INTEGER,
                enter_provider_id TEXT,
                charttime TEXT,
                medication TEXT,
                event_txt TEXT,
                scheduletime TEXT,
                storetime TEXT
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_emar_subject_id ON emar(subject_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_emar_medication ON emar(medication)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_emar_charttime ON emar(charttime)")
        
    def create_pharmacy_table(self, cursor):
        """Create pharmacy table"""
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pharmacy (
                subject_id INTEGER NOT NULL,
                hadm_id INTEGER,
                pharmacy_id INTEGER PRIMARY KEY,
                poe_id TEXT,
                starttime TEXT,
                stoptime TEXT,
                medication TEXT,
                proc_type TEXT,
                status TEXT,
                route TEXT,
                frequency TEXT
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_pharmacy_subject_id ON pharmacy(subject_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_pharmacy_medication ON pharmacy(medication)")
        
    def create_labevents_table(self, cursor):
        """Create labevents table"""
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS labevents (
                labevent_id INTEGER NOT NULL,
                subject_id INTEGER NOT NULL,
                hadm_id INTEGER,
                specimen_id INTEGER NOT NULL,
                itemid INTEGER NOT NULL,
                order_provider_id TEXT,
                charttime TEXT,
                storetime TEXT,
                value TEXT,
                valuenum REAL,
                valueuom TEXT,
                ref_range_lower REAL,
                ref_range_upper REAL,
                flag TEXT,
                priority TEXT,
                comments TEXT
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_labevents_subject_id ON labevents(subject_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_labevents_charttime ON labevents(charttime)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_labevents_itemid ON labevents(itemid)")
        
    def create_diagnoses_icd_table(self, cursor):
        """Create diagnoses_icd table"""
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS diagnoses_icd (
                subject_id INTEGER NOT NULL,
                hadm_id INTEGER NOT NULL,
                seq_num INTEGER,
                icd_code TEXT,
                icd_version INTEGER
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_diagnoses_icd_subject_id ON diagnoses_icd(subject_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_diagnoses_icd_code ON diagnoses_icd(icd_code)")
        
    def create_procedures_icd_table(self, cursor):
        """Create procedures_icd table"""
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS procedures_icd (
                subject_id INTEGER NOT NULL,
                hadm_id INTEGER NOT NULL,
                seq_num INTEGER,
                chartdate TEXT,
                icd_code TEXT,
                icd_version INTEGER
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_procedures_icd_subject_id ON procedures_icd(subject_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_procedures_icd_code ON procedures_icd(icd_code)")
        
    def create_transfers_table(self, cursor):
        """Create transfers table"""
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS transfers (
                subject_id INTEGER NOT NULL,
                hadm_id INTEGER,
                transfer_id INTEGER,
                eventtype TEXT,
                careunit TEXT,
                intime TEXT,
                outtime TEXT
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_transfers_subject_id ON transfers(subject_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_transfers_intime ON transfers(intime)")
        
    def create_microbiologyevents_table(self, cursor):
        """Create microbiologyevents table"""
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS microbiologyevents (
                microevent_id INTEGER NOT NULL,
                subject_id INTEGER NOT NULL,
                hadm_id INTEGER,
                micro_specimen_id INTEGER,
                order_provider_id TEXT,
                chartdate TEXT,
                charttime TEXT,
                spec_itemid INTEGER,
                spec_type_desc TEXT,
                test_name TEXT,
                org_itemid INTEGER,
                org_name TEXT,
                isolate_num INTEGER,
                quantity TEXT,
                ab_itemid INTEGER,
                ab_name TEXT,
                dilution_text TEXT,
                dilution_comparison TEXT,
                dilution_value REAL,
                interpretation TEXT,
                comments TEXT
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_micro_subject_id ON microbiologyevents(subject_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_micro_charttime ON microbiologyevents(charttime)")
        
    def transform_date(self, date_str):
        """Transform date from MIMIC's privacy dates (2100s) to 2025"""
        if pd.isna(date_str) or date_str is None or date_str == '':
            return None
        
        try:
            # Parse the date string
            if isinstance(date_str, str):
                # Common MIMIC date formats
                for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d', '%Y-%m-%d %H:%M:%S.%f']:
                    try:
                        dt = datetime.strptime(date_str, fmt)
                        break
                    except ValueError:
                        continue
                else:
                    # If no format matched, return original
                    return date_str
            else:
                return date_str
            
            # Transform year: MIMIC dates are in 2100s, shift to 2025
            if dt.year >= 2100:
                # Calculate years difference from 2100 and shift to 2025 base
                years_from_2100 = dt.year - 2100
                # Map to 2025 + a smaller offset to keep dates realistic
                new_year = 2025 + (years_from_2100 % 10)  # Keep within 2025-2034
                transformed_dt = dt.replace(year=new_year)
                return transformed_dt.strftime('%Y-%m-%d %H:%M:%S')
            else:
                return date_str
                
        except Exception as e:
            # If transformation fails, return original
            return date_str
        
    def load_csv_data(self):
        """Load CSV data into SQLite tables"""
        conn = sqlite3.connect(self.database_path)
        
        # Define tables to load with their date columns
        tables_to_load = {
            'patients.csv': ['dod'],
            'admissions.csv': ['admittime', 'dischtime', 'deathtime', 'edregtime', 'edouttime'],
            'prescriptions.csv': ['starttime', 'stoptime'],
            'emar.csv': ['charttime', 'scheduletime', 'storetime'],
            'pharmacy.csv': ['starttime', 'stoptime'],
            'labevents.csv': ['charttime', 'storetime'],
            'diagnoses_icd.csv': [],
            'procedures_icd.csv': ['chartdate'],
            'transfers.csv': ['intime', 'outtime'],
            'microbiologyevents.csv': ['chartdate', 'charttime']
        }
        
        for csv_file, date_columns in tables_to_load.items():
            csv_path = self.data_dir / csv_file
            if csv_path.exists():
                print(f"Loading {csv_file}...")
                table_name = csv_file.replace('.csv', '')
                self.load_table_from_csv(conn, table_name, csv_path, date_columns)
            else:
                print(f"Warning: {csv_file} not found")
                
        conn.close()
        
    def load_table_from_csv(self, conn, table_name: str, csv_path: Path, date_columns: list):
        """Load specific table from CSV with date transformation"""
        try:
            # Read CSV in smaller chunks for large files like labevents
            chunk_size = 5000 if 'labevents' in str(csv_path) else 10000
            first_chunk = True
            total_rows = 0
            
            # Clear existing data first
            cursor = conn.cursor()
            cursor.execute(f"DELETE FROM {table_name}")
            conn.commit()
            
            for chunk in pd.read_csv(csv_path, chunksize=chunk_size, low_memory=False):
                # Transform date columns
                for date_col in date_columns:
                    if date_col in chunk.columns:
                        chunk[date_col] = chunk[date_col].apply(self.transform_date)
                
                # Clean data - replace NaN with None
                chunk = chunk.where(pd.notnull(chunk), None)
                
                # Convert to list of tuples for insertion
                data = [tuple(row) for row in chunk.values]
                
                # Insert data
                placeholders = ', '.join(['?'] * len(chunk.columns))
                query = f"INSERT INTO {table_name} VALUES ({placeholders})"
                
                cursor.executemany(query, data)
                conn.commit()
                
                total_rows += len(data)
                print(f"  Loaded {len(data)} rows (total: {total_rows})")
                
        except Exception as e:
            print(f"Error loading {table_name}: {e}")
            
    def setup(self):
        """Complete database setup"""
        print(f"Creating database at {self.database_path}...")
        self.create_database()
        print("Loading CSV data with date transformation...")
        self.load_csv_data()
        print("Database setup complete!")


if __name__ == "__main__":
    # Default setup
    setup = MIMICDatabaseSetup()
    setup.setup()
