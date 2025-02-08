from fastapi import FastAPI, APIRouter, UploadFile, Header, Response
from typing import Optional
import uuid
from api.orchestrator import PipelineOrchestrator
from fastapi.middleware.cors import CORSMiddleware

import soundfile as sf
from io import BytesIO
import numpy as np
import time

app = FastAPI()  # Define FastAPI app
router = APIRouter()
orchestrator = PipelineOrchestrator()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[],  # Or use ["*"] for all
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)


@router.post("/process")
async def process_voice(
    file: UploadFile,
    x_session_id: Optional[str] = Header(None)
):
    # Generate session ID if not provided
    session_id = x_session_id or str(uuid.uuid4())
    print(f"process: session id: {session_id}")

    audio_bytes = await file.read()
    audio_buffer = BytesIO(audio_bytes)
    print(f"bytes: {audio_buffer}")
    start = time.time()
    
    # Process the audio and get response
    result = await orchestrator.process_audio(
        audio=audio_buffer,
        session_id=session_id
    )
    
    end = time.time()
    print(f"total time taken: {end - start}")
    # Return both audio response and conversation history
    response = result['audio']
    response.headers['X-Session-ID'] = session_id
    response.headers['X-Conversation-History'] = str(result['conversation'])
    return response

@router.post("/new-conversation")
async def start_new_conversation(
    x_session_id: Optional[str] = Header(None)
):
    session_id = x_session_id or str(uuid.uuid4())
    orchestrator.start_new_conversation(session_id)
    print(f"new-conversation: session_id: {session_id}")
    return {"session_id": session_id}

# Include the router in the FastAPI app
app.include_router(router)

# Run the server correctly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=5000)
