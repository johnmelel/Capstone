"""
Enhanced SQLite Database Setup for MIMIC-IV EMR Data
Loads comprehensive set of tables including critical dictionary tables
for human-readable agent responses
"""
import sqlite3
import pandas as pd
import os
from pathlib import Path
from datetime import datetime
import sys


class MIMICDatabaseSetup:
    def __init__(self, database_path=None, data_dir=None):
        if database_path is None:
            self.database_path = Path(__file__).parent / 'mimic_emr.db'
        else:
            self.database_path = Path(database_path)
            
        if data_dir is None:
            # Default to legacy MIMIC data location
            self.data_dir = Path(__file__).parent.parent / 'legacy-implementation' / 'medical-agent' / 'MIMIC_data' / 'data'
        else:
            self.data_dir = Path(data_dir)
        
    def create_database(self):
        """Create database with all tables"""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        print("Creating tables...")
        
        # Core clinical tables
        self.create_patients_table(cursor)
        self.create_admissions_table(cursor)
        self.create_labevents_table(cursor)
        self.create_prescriptions_table(cursor)
        self.create_diagnoses_icd_table(cursor)
        self.create_procedures_icd_table(cursor)
        
        # Dictionary tables (CRITICAL for human-readable output)
        self.create_d_labitems_table(cursor)
        self.create_d_icd_diagnoses_table(cursor)
        self.create_d_icd_procedures_table(cursor)
        
        # Additional useful tables
        self.create_emar_table(cursor)
        self.create_pharmacy_table(cursor)
        self.create_transfers_table(cursor)
        self.create_microbiologyevents_table(cursor)
        self.create_drgcodes_table(cursor)
        self.create_services_table(cursor)
        
        conn.commit()
        conn.close()
        print("Tables created successfully")
        
    def create_patients_table(self, cursor):
        """Core demographics table"""
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
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_patients_gender ON patients(gender)")
        
    def create_admissions_table(self, cursor):
        """Hospital admissions with temporal and categorical indexes"""
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
                hospital_expire_flag INTEGER,
                FOREIGN KEY (subject_id) REFERENCES patients(subject_id)
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_admissions_subject_id ON admissions(subject_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_admissions_admittime ON admissions(admittime)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_admissions_type ON admissions(admission_type)")
        
    def create_labevents_table(self, cursor):
        """Lab results - high query volume table"""
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS labevents (
                labevent_id INTEGER PRIMARY KEY,
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
                comments TEXT,
                FOREIGN KEY (subject_id) REFERENCES patients(subject_id),
                FOREIGN KEY (hadm_id) REFERENCES admissions(hadm_id),
                FOREIGN KEY (itemid) REFERENCES d_labitems(itemid)
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_labevents_subject_id ON labevents(subject_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_labevents_hadm_id ON labevents(hadm_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_labevents_charttime ON labevents(charttime)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_labevents_itemid ON labevents(itemid)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_labevents_subject_charttime ON labevents(subject_id, charttime)")
        
    def create_prescriptions_table(self, cursor):
        """Medication prescriptions"""
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
                route TEXT,
                FOREIGN KEY (subject_id) REFERENCES patients(subject_id),
                FOREIGN KEY (hadm_id) REFERENCES admissions(hadm_id)
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_prescriptions_subject_id ON prescriptions(subject_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_prescriptions_hadm_id ON prescriptions(hadm_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_prescriptions_drug ON prescriptions(drug)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_prescriptions_starttime ON prescriptions(starttime)")
        
    def create_diagnoses_icd_table(self, cursor):
        """ICD diagnosis codes"""
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS diagnoses_icd (
                subject_id INTEGER NOT NULL,
                hadm_id INTEGER NOT NULL,
                seq_num INTEGER,
                icd_code TEXT,
                icd_version INTEGER,
                FOREIGN KEY (subject_id) REFERENCES patients(subject_id),
                FOREIGN KEY (hadm_id) REFERENCES admissions(hadm_id)
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_diagnoses_icd_subject_id ON diagnoses_icd(subject_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_diagnoses_icd_hadm_id ON diagnoses_icd(hadm_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_diagnoses_icd_code ON diagnoses_icd(icd_code)")
        
    def create_procedures_icd_table(self, cursor):
        """ICD procedure codes"""
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS procedures_icd (
                subject_id INTEGER NOT NULL,
                hadm_id INTEGER NOT NULL,
                seq_num INTEGER,
                chartdate TEXT,
                icd_code TEXT,
                icd_version INTEGER,
                FOREIGN KEY (subject_id) REFERENCES patients(subject_id),
                FOREIGN KEY (hadm_id) REFERENCES admissions(hadm_id)
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_procedures_icd_subject_id ON procedures_icd(subject_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_procedures_icd_hadm_id ON procedures_icd(hadm_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_procedures_icd_code ON procedures_icd(icd_code)")
        
    def create_d_labitems_table(self, cursor):
        """CRITICAL: Lab test dictionary for human-readable lab names"""
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS d_labitems (
                itemid INTEGER PRIMARY KEY,
                label TEXT,
                fluid TEXT,
                category TEXT
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_d_labitems_label ON d_labitems(label)")
        
    def create_d_icd_diagnoses_table(self, cursor):
        """CRITICAL: ICD diagnosis dictionary for human-readable diagnoses"""
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS d_icd_diagnoses (
                icd_code TEXT,
                icd_version INTEGER,
                long_title TEXT,
                PRIMARY KEY (icd_code, icd_version)
            )
        """)
        
    def create_d_icd_procedures_table(self, cursor):
        """CRITICAL: ICD procedure dictionary for human-readable procedures"""
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS d_icd_procedures (
                icd_code TEXT,
                icd_version INTEGER,
                long_title TEXT,
                PRIMARY KEY (icd_code, icd_version)
            )
        """)
        
    def create_emar_table(self, cursor):
        """Electronic Medication Administration Record"""
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
                storetime TEXT,
                FOREIGN KEY (subject_id) REFERENCES patients(subject_id),
                FOREIGN KEY (hadm_id) REFERENCES admissions(hadm_id)
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_emar_subject_id ON emar(subject_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_emar_medication ON emar(medication)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_emar_charttime ON emar(charttime)")
        
    def create_pharmacy_table(self, cursor):
        """Pharmacy orders"""
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
                frequency TEXT,
                FOREIGN KEY (subject_id) REFERENCES patients(subject_id),
                FOREIGN KEY (hadm_id) REFERENCES admissions(hadm_id)
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_pharmacy_subject_id ON pharmacy(subject_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_pharmacy_medication ON pharmacy(medication)")
        
    def create_transfers_table(self, cursor):
        """Patient transfers between units"""
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS transfers (
                subject_id INTEGER NOT NULL,
                hadm_id INTEGER,
                transfer_id INTEGER PRIMARY KEY,
                eventtype TEXT,
                careunit TEXT,
                intime TEXT,
                outtime TEXT,
                FOREIGN KEY (subject_id) REFERENCES patients(subject_id),
                FOREIGN KEY (hadm_id) REFERENCES admissions(hadm_id)
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_transfers_subject_id ON transfers(subject_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_transfers_intime ON transfers(intime)")
        
    def create_microbiologyevents_table(self, cursor):
        """Microbiology culture results"""
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS microbiologyevents (
                microevent_id INTEGER PRIMARY KEY,
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
                comments TEXT,
                FOREIGN KEY (subject_id) REFERENCES patients(subject_id),
                FOREIGN KEY (hadm_id) REFERENCES admissions(hadm_id)
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_micro_subject_id ON microbiologyevents(subject_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_micro_charttime ON microbiologyevents(charttime)")
        
    def create_drgcodes_table(self, cursor):
        """DRG codes for severity and billing"""
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS drgcodes (
                subject_id INTEGER NOT NULL,
                hadm_id INTEGER NOT NULL,
                drg_type TEXT,
                drg_code TEXT,
                description TEXT,
                drg_severity INTEGER,
                drg_mortality INTEGER,
                FOREIGN KEY (subject_id) REFERENCES patients(subject_id),
                FOREIGN KEY (hadm_id) REFERENCES admissions(hadm_id)
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_drgcodes_subject_id ON drgcodes(subject_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_drgcodes_hadm_id ON drgcodes(hadm_id)")
        
    def create_services_table(self, cursor):
        """Service transfers during admission"""
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS services (
                subject_id INTEGER NOT NULL,
                hadm_id INTEGER NOT NULL,
                transfertime TEXT,
                prev_service TEXT,
                curr_service TEXT,
                FOREIGN KEY (subject_id) REFERENCES patients(subject_id),
                FOREIGN KEY (hadm_id) REFERENCES admissions(hadm_id)
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_services_subject_id ON services(subject_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_services_hadm_id ON services(hadm_id)")
        
    def transform_date(self, date_str):
        """Transform date from MIMIC's privacy dates (2100s) to 2025"""
        if pd.isna(date_str) or date_str is None or date_str == '':
            return None
        
        try:
            if isinstance(date_str, str):
                # Try common MIMIC date formats
                for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d', '%Y-%m-%d %H:%M:%S.%f']:
                    try:
                        dt = datetime.strptime(date_str, fmt)
                        break
                    except ValueError:
                        continue
                else:
                    return date_str
            else:
                return date_str
            
            # Transform year: MIMIC dates are in 2100s, shift to 2025
            if dt.year >= 2100:
                years_from_2100 = dt.year - 2100
                new_year = 2025 + (years_from_2100 % 10)  # Keep within 2025-2034
                transformed_dt = dt.replace(year=new_year)
                return transformed_dt.strftime('%Y-%m-%d %H:%M:%S')
            else:
                return date_str
                
        except Exception as e:
            return date_str
        
    def load_csv_data(self):
        """Load CSV data into SQLite tables"""
        if not self.data_dir.exists():
            print(f"ERROR: Data directory not found: {self.data_dir}")
            print("Please ensure MIMIC data CSVs are available")
            return False
            
        conn = sqlite3.connect(self.database_path)
        
        # Define tables to load with their date columns
        tables_config = {
            # Core tables
            'patients.csv': (['dod'], 'patients'),
            'admissions.csv': (['admittime', 'dischtime', 'deathtime', 'edregtime', 'edouttime'], 'admissions'),
            'labevents.csv': (['charttime', 'storetime'], 'labevents'),
            'prescriptions.csv': (['starttime', 'stoptime'], 'prescriptions'),
            'diagnoses_icd.csv': ([], 'diagnoses_icd'),
            'procedures_icd.csv': (['chartdate'], 'procedures_icd'),
            
            # Dictionary tables (CRITICAL)
            'd_labitems.csv': ([], 'd_labitems'),
            'd_icd_diagnoses.csv': ([], 'd_icd_diagnoses'),
            'd_icd_procedures.csv': ([], 'd_icd_procedures'),
            
            # Additional tables
            'emar.csv': (['charttime', 'scheduletime', 'storetime'], 'emar'),
            'pharmacy.csv': (['starttime', 'stoptime'], 'pharmacy'),
            'transfers.csv': (['intime', 'outtime'], 'transfers'),
            'microbiologyevents.csv': (['chartdate', 'charttime'], 'microbiologyevents'),
            'drgcodes.csv': ([], 'drgcodes'),
            'services.csv': (['transfertime'], 'services'),
        }
        
        for csv_file, (date_columns, table_name) in tables_config.items():
            csv_path = self.data_dir / csv_file
            if csv_path.exists():
                print(f"Loading {csv_file}...")
                success = self.load_table_from_csv(conn, table_name, csv_path, date_columns)
                if not success:
                    print(f"  Warning: Failed to load {csv_file}")
            else:
                print(f"  Skipping {csv_file} (not found)")
                
        conn.close()
        return True
        
    def load_table_from_csv(self, conn, table_name: str, csv_path: Path, date_columns: list):
        """Load specific table from CSV with date transformation"""
        try:
            # Read CSV in chunks for large files
            chunk_size = 5000 if 'labevents' in str(csv_path) else 10000
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
                
            print(f"  Loaded {total_rows} rows into {table_name}")
            return True
            
        except Exception as e:
            print(f"  ERROR loading {table_name}: {e}")
            return False
            
    def verify_database(self):
        """Verify database was created successfully"""
        if not self.database_path.exists():
            print("ERROR: Database file was not created")
            return False
            
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        print("\nDatabase Verification:")
        print("=" * 60)
        
        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        tables = [row[0] for row in cursor.fetchall()]
        
        print(f"Found {len(tables)} tables:")
        for table in tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            print(f"  {table:30s}: {count:>10,} rows")
        
        conn.close()
        print("=" * 60)
        return True
            
    def setup(self):
        """Complete database setup"""
        print(f"Creating MIMIC EMR database at {self.database_path}...")
        print(f"Data source: {self.data_dir}")
        print()
        
        self.create_database()
        print()
        
        success = self.load_csv_data()
        if not success:
            print("\nWARNING: Some data loading issues occurred")
        
        print()
        self.verify_database()
        print("\nDatabase setup complete!")
        print(f"Database location: {self.database_path.absolute()}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Setup MIMIC EMR SQLite database')
    parser.add_argument('--db-path', help='Path to database file (default: ./mimic_emr.db)')
    parser.add_argument('--data-dir', help='Path to MIMIC CSV data directory')
    
    args = parser.parse_args()
    
    setup = MIMICDatabaseSetup(
        database_path=args.db_path,
        data_dir=args.data_dir
    )
    setup.setup()


if __name__ == "__main__":
    main()
