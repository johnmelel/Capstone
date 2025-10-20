"""
Minimal MCP-SQL Server Implementation
Provides SQL query tool for structured data access using SQLite
"""
import sqlite3
import json
import pandas as pd
import os
from typing import Dict, Any, List, Optional
from pathlib import Path
from .a2a_messages import MCPRequest, MCPResponse


class MCPSQLServer:
    """Minimal MCP server for SQL operations using SQLite"""
    
    def __init__(self, db_path='data/mimic.db'):
        self.db_path = db_path
        self.ensure_database_exists()
        
    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> MCPResponse:
        """Handle MCP tool calls"""
        if tool_name == "sql.query":
            return self._execute_sql_query(arguments.get("sql", ""))
        else:
            return MCPResponse(
                success=False,
                error=f"Unknown tool: {tool_name}"
            )
    
    def _execute_sql_query(self, sql: str) -> MCPResponse:
        """Execute SQL query and return results"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row  # Makes rows accessible like dictionaries
            cursor = conn.cursor()
            
            cursor.execute(sql)
            
            if sql.strip().upper().startswith('SELECT'):
                results = [dict(row) for row in cursor.fetchall()]
                return MCPResponse(
                    success=True,
                    result={
                        'rows': results,
                        'row_count': len(results)
                    }
                )
            else:
                conn.commit()
                return MCPResponse(
                    success=True,
                    result={
                        'affected_rows': cursor.rowcount
                    }
                )
                
        except sqlite3.Error as e:
            return MCPResponse(
                success=False,
                error=f"Database error: {str(e)}"
            )
        except Exception as e:
            return MCPResponse(
                success=False,
                error=f"SQL execution error: {str(e)}"
            )
        finally:
            if 'conn' in locals():
                cursor.close()
                conn.close()
    
    def ensure_database_exists(self):
        """Create and populate database from MIMIC CSV files if it doesn't exist"""
        if os.path.exists(self.db_path):
            return  # Database already exists
        
        # Create directory if needed
        os.makedirs(os.path.dirname(self.db_path) if os.path.dirname(self.db_path) else '.', exist_ok=True)
        
        print(f"Creating SQLite database at {self.db_path}...")
        print("Loading MIMIC data from CSVs...")
        
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Load key MIMIC tables from CSVs
            csv_tables = [
                ('patients', 'MIMIC_data/data/patients.csv'),
                ('prescriptions', 'MIMIC_data/data/prescriptions.csv'),
                ('admissions', 'MIMIC_data/data/admissions.csv'),
                ('emar', 'MIMIC_data/data/emar.csv')
            ]
            
            for table_name, csv_path in csv_tables:
                if os.path.exists(csv_path):
                    print(f"  Loading {table_name} from {csv_path}...")
                    df = pd.read_csv(csv_path)
                    df.to_sql(table_name, conn, if_exists='replace', index=False)
                    print(f"  Loaded {len(df)} rows into {table_name}")
                else:
                    print(f"  Warning: {csv_path} not found, skipping {table_name}")
            
            conn.close()
            print(f"Database created successfully at {self.db_path}")
            
        except Exception as e:
            print(f"Error creating database: {e}")
            # Remove partial database file
            if os.path.exists(self.db_path):
                os.remove(self.db_path)
            raise
    
    def get_available_tools(self) -> List[str]:
        """Return list of available tools"""
        return ["sql.query"]
