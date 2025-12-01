import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Any
import base64

app = FastAPI(title="Mock Medical Agent API - Test Service")

# Configure CORS (same as original API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(SCRIPT_DIR, "images")


class ChatRequest(BaseModel):
    query: str
    image: Optional[str] = None  # Base64 encoded image (optional)


class ChatResponse(BaseModel):
    answer: str
    structured_source: Optional[Any] = None
    unstructured_source: Optional[Any] = None
    reasoning_steps: Optional[int] = None
    error: Optional[str] = None
    image: Optional[str] = None  # Base64 encoded image for output


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Mock chat endpoint that:
    - For text input: returns 'the user said: "input text"'
    - For image input: returns 'image received'
    - For text "image": returns an image from the images folder
    """
    
    # Check if an image was sent
    if request.image:
        return ChatResponse(
            answer="image received",
            reasoning_steps=1,
            structured_source={"type": "image_input"},
            unstructured_source=None
        )
    
    # Check if user is requesting an image
    if request.query.strip().lower() == "image":
        # Find an image in the images folder
        image_base64 = None
        if os.path.exists(IMAGES_DIR):
            image_files = [f for f in os.listdir(IMAGES_DIR) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp'))]
            if image_files:
                image_path = os.path.join(IMAGES_DIR, image_files[0])
                with open(image_path, "rb") as img_file:
                    image_data = img_file.read()
                    image_base64 = base64.b64encode(image_data).decode('utf-8')
                    # Determine mime type
                    ext = image_files[0].lower().split('.')[-1]
                    mime_type = {
                        'png': 'image/png',
                        'jpg': 'image/jpeg',
                        'jpeg': 'image/jpeg',
                        'gif': 'image/gif',
                        'webp': 'image/webp'
                    }.get(ext, 'image/png')
                    image_base64 = f"data:{mime_type};base64,{image_base64}"
        
        if image_base64:
            return ChatResponse(
                answer="Here is the requested image:",
                reasoning_steps=1,
                structured_source=None,
                unstructured_source={"type": "image_output"},
                image=image_base64
            )
        else:
            return ChatResponse(
                answer="No images found in the images folder. Please add an image to `api_frontend_test/images/`",
                reasoning_steps=1,
                error="No images available"
            )
    
    # Default: echo back the text
    response_text = f'the user said: "{request.query}"'
    
    return ChatResponse(
        answer=response_text,
        reasoning_steps=2,
        structured_source={"echo": True, "original_query": request.query},
        unstructured_source={"mock_service": True}
    )


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "ok", "orchestrator_initialized": True, "mock_service": True}


@app.get("/")
async def root():
    """Root endpoint with service info"""
    return {
        "service": "Mock Medical Agent API",
        "version": "1.0.0",
        "endpoints": {
            "POST /chat": "Send a chat message",
            "GET /health": "Health check"
        },
        "behavior": {
            "text_input": 'Returns: the user said: "your text"',
            "image_input": 'Returns: image received',
            "query_image": 'Returns an image from the images folder'
        }
    }


if __name__ == "__main__":
    import uvicorn
    print("Starting Mock Medical Agent API on http://localhost:8000")
    print("Endpoints:")
    print("  POST /chat - Send chat messages")
    print("  GET /health - Health check")
    print("")
    print("Behavior:")
    print('  - Text input: Returns "the user said: <your text>"')
    print('  - Image input: Returns "image received"')
    print('  - Query "image": Returns an image from ./images/ folder')
    uvicorn.run(app, host="0.0.0.0", port=8000)
