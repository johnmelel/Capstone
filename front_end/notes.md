```bash
brunamedeiros@Mac api % source ../../venv/bin/activate
(venv) brunamedeiros@Mac api % uvicorn front_end.api.main:app --reload --port 8000
```


```bash

# terminal window 1
cd ~/Documents/GitHub/Capstone
source venv/bin/activate

# terminal window 2 (workers)
cd ~/Documents/GitHub/Capstone/medical-agent
source ../venv/bin/activate
./scripts/start_workers.sh

# terminal window 3 (fast API backend - connects react frontent to workers)
cd ~/Documents/GitHub/Capstone
source venv/bin/activate
python3 -m uvicorn front_end.api.main:app --reload --port 8000
        # fastAPI server starts on port 8000
        # connects to both workers (8001 and 8002)
        # initializes the orchestrator

# terminal window 4 (react frontend)
cd ~/Documents/GitHub/Capstone/front_end/client
npm run dev
```