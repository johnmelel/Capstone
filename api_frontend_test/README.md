# Mock Frontend Test API

A simple mock service to test the Iris AI frontend independently.

## Behavior

| Input | Output |
|-------|--------|
| Any text | `the user said: "your text"` |
| Image sent | `image received` |
| Text = "image" | Returns an image from `./images/` folder |

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Add test images to the `images/` subfolder (optional, for image output testing)

3. Run the service:
```bash
python main.py
```
Or:
```bash
uvicorn main:app --reload --port 8000
```

## Running with Frontend

**Terminal 1 - Start this mock API:**
```bash
cd api_frontend_test
python main.py
```

**Terminal 2 - Start the frontend:**
```bash
cd front_end/client
npm install
npm run dev
```

Then open http://localhost:5173 in your browser.

## API Endpoints

- `POST /chat` - Main chat endpoint
  - Request: `{ "query": "string", "image": "base64_string (optional)" }`
  - Response: `{ "answer": "string", "image": "base64_string (optional)", ... }`

- `GET /health` - Health check
- `GET /` - Service info
