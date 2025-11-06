When running the embedding code, ensure your directory is `Data_Ingestion_VDB`
```bash
yourName@Mac Data_Ingestion_VDB %
```

# Runnables

Workflow:
1. `run_pipleine.py`
2. `strategy_test.py` (e.g., `strategy_5_research_test.py`) 

Observation: Do not run `strategy.py` (e.g., `strategy_5_research_test.py`) directly

---

For main processing pipeline:
```bash
yourName@Mac Data_Ingestion_VDB % python run_pipleine.py
```

This file should parse and embed all data, according to the correct parsing strategy. This one file calls for all other .py files, so you do not need to run them separatedly. 

If you do want to visualize the resulting chunks in order to assess the parsing and chunking strategy, run each `strategy_/.../_test.py` file separatedly with this directory path:

```bash
yourName@Mac Data_Ingestion_VDB % python -m src.parsers.strategy_5_research_test
```

# Other Additional Information

```bash
# Go to your home directory
cd ~/path/to/Data_Ingestion_VDB
cd Documents/GitHub/Capstone/Data_Ingestion_VDB
cd ~                                  

# Clean old database
rm -rf vector_db/                     

# Verify you're in the right place
pwd
ls
```

```bash
gcloud auth list # verify authentication (which account is active)
gcloud config get-value project # return: adsp-34002-ip09-team-2
gcloud projects list # a list of all GCP projects john was part of?? or Uchicago? either way, our project ID should be there. 
        # adsp-34002-ip09-team-2          Team-2                          336671232225
gcloud config set project adsp-34002-ip09-team-2 # set this as our project



# BEGIN
gcloud compute instances start milvus-server --zone=us-central1-a # START the server
gcloud compute instances stop milvus-server --zone=us-central1-a  # STOP the server
gcloud compute instances describe milvus-server --zone=us-central1-a --format="get(networkInterfaces[0].accessConfigs[0].natIP)" # CHECK EXTERNAL API
gcloud compute instances describe milvus-server --zone=us-central1-a --format="get(status)" # Check STATUS
gcloud compute ssh milvus-server --zone=us-central1-a # Check if docker container is running

# IF IT DOESN'T WORK:
gcloud compute ssh milvus-server --zone=us-central1-a   # 1) SSH into server
docker ps                                               # 2) Check if docker container is running
docker start milvus-standalone                          # 3) Start milvus container



# DELETE instance 
gcloud compute instances delete milvus-server --zone=us-central1-a


# Resize
gcloud compute instances set-machine-type milvus-server \
  --machine-type=e2-standard-4 \
  --zone=us-central1-a

```


Small team (you):
- e2-standard-2 (2 CPU, 8GB RAM)
- 50GB pd-standard
- Cost: ~$40/month

Medium team:
- e2-standard-4 (4 CPU, 16GB RAM)
- 100GB pd-ssd
- Cost: ~$120/month

Large enterprise:
- n2-standard-8 (8 CPU, 32GB RAM)
- 500GB pd-ssd
- + Kubernetes cluster for high availability
- Cost: $500+/month

SO, I created mivlus under these configurations. i can change later
```bash
gcloud compute instances create milvus-server \
--zone=us-central1-a \
  --machine-type=e2-standard-2 \
  --boot-disk-size=50GB \
  --boot-disk-type=pd-standard \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --tags=milvus-server \
  --network=vpc-1 \
  --subnet=subnet-1
```
NAME           ZONE           MACHINE_TYPE   PREEMPTIBLE  INTERNAL_IP  EXTERNAL_IP  STATUS
milvus-server  us-central1-a  e2-standard-2               10.0.0.4     136.115.151.65  RUNNING

Instance internal IP is 10.0.0.4
Instance external IP is 34.134.169.84
