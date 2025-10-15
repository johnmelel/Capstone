```bash
# 1. Make sure you're in the right folder
cd ~/path/to/Data_Ingestion_VDB

# 4. Clean old database
rm -rf vector_db/

# 5. Run!
python run_pipeline.py --test

# Go to your home directory
cd ~

# Then navigate to your project
cd Documents/GitHub/Capstone/Data_Ingestion_VDB

# Verify you're in the right place
pwd
ls


# 1. Basic statistics and metadata quality
python3 analyze_embeddings.py

# 2. Test search quality
python3 test_search_quality.py

# 3. Test multimodal (text→image, image→text)
python3 test_multimodal.py
```