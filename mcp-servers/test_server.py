#!/usr/bin/env python3
"""
Comprehensive Test Suite for Unified MCP Server using pytest
Run with: pytest test_server.py -v
"""
import pytest
import pytest_asyncio
import asyncio
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from server import UnifiedMCPServer


@pytest.fixture(scope="module")
def event_loop():
    """Create an event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="module")
async def server():
    """Initialize server once for all tests"""
    server_instance = UnifiedMCPServer()
    return server_instance


class TestEMRDatabaseTools:
    """Test suite for EMR database access tools"""
    
    @pytest.mark.asyncio
    async def test_list_tables(self, server):
        """Test 1: List all database tables"""
        result = await server.list_tables()
        
        assert 'tables' in result
        assert 'count' in result
        assert result['count'] == 15, f"Expected 15 tables, got {result['count']}"
        
        # Verify key tables exist
        tables = result['tables']
        assert 'patients' in tables
        assert 'd_labitems' in tables
        assert 'd_icd_diagnoses' in tables
        assert 'labevents' in tables
        
        print(f"\n✓ Found {result['count']} tables: {', '.join(tables[:5])}...")
    
    @pytest.mark.asyncio
    async def test_get_schema_patients(self, server):
        """Test 2: Get schema for patients table"""
        result = await server.get_schema('patients')
        
        assert 'table_name' in result
        assert result['table_name'] == 'patients'
        assert 'columns' in result
        assert 'indexes' in result
        
        columns = result['columns']
        assert len(columns) == 6, f"Expected 6 columns, got {len(columns)}"
        
        # Verify key columns exist
        column_names = [col['name'] for col in columns]
        assert 'subject_id' in column_names
        assert 'gender' in column_names
        assert 'anchor_age' in column_names
        
        print(f"\n✓ Patients table has {len(columns)} columns: {', '.join(column_names)}")
    
    @pytest.mark.asyncio
    async def test_run_sql_count_patients(self, server):
        """Test 3: Count patients using SQL"""
        query = "SELECT COUNT(*) as patient_count FROM patients"
        result = await server.run_sql(query)
        
        assert 'query' in result
        assert 'rows' in result
        assert 'row_count' in result
        assert result['row_count'] == 1
        
        count = result['rows'][0]['patient_count']
        assert count == 100, f"Expected 100 patients, got {count}"
        
        print(f"\n✓ Database contains {count} patients")
    
    @pytest.mark.asyncio
    async def test_run_sql_with_dictionary_join(self, server):
        """Test 4: Query with JOIN to get human-readable lab test names"""
        query = """
        SELECT 
            d.label as test_name,
            l.valuenum as value,
            l.valueuom as unit
        FROM labevents l
        JOIN d_labitems d ON l.itemid = d.itemid
        WHERE l.subject_id = 10000032
        ORDER BY l.charttime DESC
        LIMIT 5
        """
        result = await server.run_sql(query)
        
        assert result['row_count'] > 0, "No lab results found"
        
        # Verify we got human-readable test names, not codes
        for row in result['rows']:
            assert row['test_name'] is not None
            assert isinstance(row['test_name'], str)
            # Test names should be readable, not numeric codes
            assert not row['test_name'].isdigit()
        
        print(f"\n✓ Retrieved {result['row_count']} lab results with readable test names")
        print(f"  Example: {result['rows'][0]['test_name']}")
    
    @pytest.mark.asyncio
    async def test_run_sql_rejects_insert(self, server):
        """Test 5: Verify INSERT queries are rejected"""
        query = "INSERT INTO patients (subject_id) VALUES (99999)"
        
        with pytest.raises(ValueError) as exc_info:
            await server.run_sql(query)
        
        assert "Only SELECT queries are allowed" in str(exc_info.value)
        
        print(f"\n✓ INSERT query properly rejected: {exc_info.value}")
    
    @pytest.mark.asyncio
    async def test_run_sql_rejects_update(self, server):
        """Test 6: Verify UPDATE queries are rejected"""
        query = "UPDATE patients SET gender='M' WHERE subject_id=10000032"
        
        with pytest.raises(ValueError) as exc_info:
            await server.run_sql(query)
        
        assert "Only SELECT queries are allowed" in str(exc_info.value)
        
        print(f"\n✓ UPDATE query properly rejected")
    
    @pytest.mark.asyncio
    async def test_run_sql_rejects_delete(self, server):
        """Test 7: Verify DELETE queries are rejected"""
        query = "DELETE FROM patients WHERE subject_id=10000032"
        
        with pytest.raises(ValueError) as exc_info:
            await server.run_sql(query)
        
        assert "Only SELECT queries are allowed" in str(exc_info.value)
        
        print(f"\n✓ DELETE query properly rejected")


class TestVectorSearchTools:
    """Test suite for vector search tools"""
    
    @pytest.mark.asyncio
    async def test_semantic_search_basic(self, server):
        """Test 8: Basic semantic search"""
        if not server.milvus_connected:
            pytest.skip("Milvus not connected - skipping vector search tests")
        
        query = "amphetamine monitoring"
        result = await server.semantic_search(query, top_k=3)
        
        assert 'query' in result
        assert result['query'] == query
        assert 'documents' in result
        assert 'total_found' in result
        
        docs = result['documents']
        assert len(docs) > 0, "No documents found"
        assert len(docs) <= 3, "Too many documents returned"
        
        # Verify document structure
        for doc in docs:
            assert 'text' in doc
            assert 'source' in doc
            assert 'relevance' in doc
            assert 'doc_id' in doc
            assert 0.0 <= doc['relevance'] <= 1.0, "Relevance score out of range"
        
        print(f"\n✓ Semantic search returned {len(docs)} documents")
        print(f"  Top result relevance: {docs[0]['relevance']:.4f}")
        print(f"  Source: {docs[0]['source']}")
    
    @pytest.mark.asyncio
    async def test_semantic_search_with_context(self, server):
        """Test 9: Semantic search with patient context"""
        if not server.milvus_connected:
            pytest.skip("Milvus not connected - skipping vector search tests")
        
        query = "patient on amphetamine cardiovascular monitoring requirements"
        result = await server.semantic_search(query, top_k=5)
        
        docs = result['documents']
        assert len(docs) > 0, "No documents found"
        assert len(docs) <= 5, "Too many documents returned"
        
        # Verify relevance scores are sorted descending
        relevances = [doc['relevance'] for doc in docs]
        assert relevances == sorted(relevances, reverse=True), "Results not sorted by relevance"
        
        print(f"\n✓ Contextual search returned {len(docs)} documents")
        print(f"  Relevance range: {docs[-1]['relevance']:.4f} to {docs[0]['relevance']:.4f}")
    
    @pytest.mark.asyncio
    async def test_semantic_search_top_k_parameter(self, server):
        """Test 10: Verify top_k parameter works correctly"""
        if not server.milvus_connected:
            pytest.skip("Milvus not connected - skipping vector search tests")
        
        query = "diabetes treatment"
        
        # Test with different top_k values
        for k in [1, 3, 5]:
            result = await server.semantic_search(query, top_k=k)
            docs = result['documents']
            assert len(docs) <= k, f"Expected at most {k} documents, got {len(docs)}"
        
        print(f"\n✓ top_k parameter working correctly")


class TestIntegration:
    """Integration tests combining multiple tools"""
    
    @pytest.mark.asyncio
    async def test_get_patient_data_and_search_literature(self, server):
        """Test 11: Integration test - get patient meds and search for monitoring"""
        # Step 1: Get patient's medications
        sql_query = """
        SELECT DISTINCT drug
        FROM prescriptions
        WHERE subject_id = 10000032
        LIMIT 1
        """
        sql_result = await server.run_sql(sql_query)
        
        assert sql_result['row_count'] > 0, "No medications found"
        drug_name = sql_result['rows'][0]['drug']
        
        print(f"\n✓ Found patient medication: {drug_name}")
        
        # Step 2: Search literature for monitoring guidelines (if Milvus available)
        if server.milvus_connected:
            search_query = f"{drug_name} monitoring guidelines"
            search_result = await server.semantic_search(search_query, top_k=2)
            
            assert len(search_result['documents']) > 0
            print(f"✓ Found {len(search_result['documents'])} monitoring guidelines")
        else:
            print("  (Skipped literature search - Milvus not connected)")
    
    @pytest.mark.asyncio
    async def test_schema_then_query(self, server):
        """Test 12: Integration test - use schema info to construct query"""
        # Step 1: Get schema to understand table structure
        schema_result = await server.get_schema('labevents')
        columns = [col['name'] for col in schema_result['columns']]
        
        assert 'subject_id' in columns
        assert 'itemid' in columns
        assert 'valuenum' in columns
        
        print(f"\n✓ Retrieved labevents schema with {len(columns)} columns")
        
        # Step 2: Use schema knowledge to construct valid query
        query = """
        SELECT COUNT(DISTINCT subject_id) as patient_count
        FROM labevents
        """
        query_result = await server.run_sql(query)
        
        patient_count = query_result['rows'][0]['patient_count']
        assert patient_count > 0
        
        print(f"✓ Found lab data for {patient_count} patients")


if __name__ == "__main__":
    # Run with pytest
    pytest.main([__file__, "-v", "--tb=short"])
