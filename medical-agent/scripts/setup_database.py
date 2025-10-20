"""
Setup MySQL database with MIMIC-IV data from CSVs
"""
import mysql.connector
import pandas as pd
import os
from pathlib import Path


class MIMICDatabaseSetup:
    def __init__(self, host='localhost', user='root', password='password', database='mimic'):
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.data_dir = Path(__file__).parent.parent / 'MIMIC_data' / 'data'
        
    def create_database(self):
        """Create database and tables"""
        # Connect without database first
        conn = mysql.connector.connect(
            host=self.host,
            user=self.user,
            password=self.password
        )
        cursor = conn.cursor()
        
        # Create database
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {self.database}")
        cursor.execute(f"USE {self.database}")
        
        # Create key tables for our use case
        self.create_patients_table(cursor)
        self.create_admissions_table(cursor)
        self.create_prescriptions_table(cursor)
        self.create_emar_table(cursor)
        self.create_pharmacy_table(cursor)
        
        conn.commit()
        conn.close()
        
    def create_patients_table(self, cursor):
        """Create patients table"""
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS patients (
                subject_id INT PRIMARY KEY,
                gender VARCHAR(1) NOT NULL,
                anchor_age INT NOT NULL,
                anchor_year INT NOT NULL,
                anchor_year_group VARCHAR(255) NOT NULL,
                dod TIMESTAMP NULL
            )
        """)
        
    def create_admissions_table(self, cursor):
        """Create admissions table"""
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS admissions (
                subject_id INT NOT NULL,
                hadm_id INT PRIMARY KEY,
                admittime TIMESTAMP NOT NULL,
                dischtime TIMESTAMP,
                deathtime TIMESTAMP,
                admission_type VARCHAR(40) NOT NULL,
                admission_location VARCHAR(60),
                discharge_location VARCHAR(60),
                insurance VARCHAR(255),
                language VARCHAR(10),
                marital_status VARCHAR(30),
                race VARCHAR(80),
                hospital_expire_flag SMALLINT,
                INDEX idx_subject_id (subject_id),
                INDEX idx_admittime (admittime)
            )
        """)
        
    def create_prescriptions_table(self, cursor):
        """Create prescriptions table"""
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS prescriptions (
                subject_id INT NOT NULL,
                hadm_id INT NOT NULL,
                pharmacy_id INT,
                starttime TIMESTAMP,
                stoptime TIMESTAMP,
                drug_type VARCHAR(100),
                drug VARCHAR(255),
                prod_strength VARCHAR(120),
                dose_val_rx VARCHAR(120),
                dose_unit_rx VARCHAR(50),
                form_val_disp VARCHAR(50),
                form_unit_disp VARCHAR(50),
                route VARCHAR(60),
                INDEX idx_subject_id (subject_id),
                INDEX idx_hadm_id (hadm_id),
                INDEX idx_drug (drug),
                INDEX idx_starttime (starttime)
            )
        """)
        
    def create_emar_table(self, cursor):
        """Create emar table"""
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS emar (
                subject_id INT NOT NULL,
                hadm_id INT,
                emar_id VARCHAR(25) PRIMARY KEY,
                emar_seq INT,
                poe_id VARCHAR(25),
                pharmacy_id INT,
                charttime TIMESTAMP,
                medication VARCHAR(255),
                event_txt VARCHAR(100),
                scheduletime TIMESTAMP,
                storetime TIMESTAMP,
                INDEX idx_subject_id (subject_id),
                INDEX idx_hadm_id (hadm_id),
                INDEX idx_medication (medication),
                INDEX idx_charttime (charttime)
            )
        """)
        
    def create_pharmacy_table(self, cursor):
        """Create pharmacy table"""
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pharmacy (
                subject_id INT NOT NULL,
                hadm_id INT,
                pharmacy_id INT PRIMARY KEY,
                poe_id VARCHAR(25),
                starttime TIMESTAMP,
                stoptime TIMESTAMP,
                medication VARCHAR(255),
                proc_type VARCHAR(50),
                status VARCHAR(50),
                route VARCHAR(120),
                frequency VARCHAR(120),
                INDEX idx_subject_id (subject_id),
                INDEX idx_medication (medication),
                INDEX idx_starttime (starttime)
            )
        """)
        
    def load_csv_data(self):
        """Load CSV data into MySQL tables"""
        conn = mysql.connector.connect(
            host=self.host,
            user=self.user,
            password=self.password,
            database=self.database
        )
        
        # Load key tables
        tables_to_load = [
            'patients.csv',
            'admissions.csv', 
            'prescriptions.csv',
            'emar.csv',
            'pharmacy.csv'
        ]
        
        for csv_file in tables_to_load:
            csv_path = self.data_dir / csv_file
            if csv_path.exists():
                print(f"Loading {csv_file}...")
                table_name = csv_file.replace('.csv', '')
                self.load_table_from_csv(conn, table_name, csv_path)
            else:
                print(f"Warning: {csv_file} not found")
                
        conn.close()
        
    def load_table_from_csv(self, conn, table_name: str, csv_path: Path):
        """Load specific table from CSV"""
        try:
            # Read CSV in chunks to handle large files
            chunk_size = 10000
            first_chunk = True
            
            for chunk in pd.read_csv(csv_path, chunksize=chunk_size, low_memory=False):
                # Clean data
                chunk = chunk.where(pd.notnull(chunk), None)
                
                # Convert to list of tuples for insertion
                data = [tuple(row) for row in chunk.values]
                
                if first_chunk:
                    # Clear existing data
                    cursor = conn.cursor()
                    cursor.execute(f"DELETE FROM {table_name}")
                    first_chunk = False
                
                # Insert data
                placeholders = ', '.join(['%s'] * len(chunk.columns))
                query = f"INSERT INTO {table_name} VALUES ({placeholders})"
                
                cursor = conn.cursor()
                cursor.executemany(query, data)
                conn.commit()
                cursor.close()
                
                print(f"  Loaded {len(data)} rows")
                
        except Exception as e:
            print(f"Error loading {table_name}: {e}")
            
    def setup(self):
        """Complete database setup"""
        print("Creating database and tables...")
        self.create_database()
        print("Loading CSV data...")
        self.load_csv_data()
        print("Database setup complete!")


if __name__ == "__main__":
    # Default setup - adjust credentials as needed
    setup = MIMICDatabaseSetup(
        host='localhost',
        user='root', 
        password='password',
        database='mimic'
    )
    setup.setup()
