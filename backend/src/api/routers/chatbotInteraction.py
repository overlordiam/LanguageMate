from fastapi import FastAPI, APIRouter, UploadFile
from api.orchestrator import PipelineOrchestrator
from fastapi.middleware.cors import CORSMiddleware

import soundfile as sf
from io import BytesIO
import numpy as np


app = FastAPI()  # Define FastAPI app
router = APIRouter()
orchestrator = PipelineOrchestrator()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Or use ["*"] for all
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)


@router.post("/process")
async def process_voice(file: UploadFile):
    
    audio_bytes = await file.read()
    audio_buffer = BytesIO(audio_bytes)
    print(f"bytes: {audio_buffer}")
    
    # audio = await file.read()
    result = await orchestrator.process_audio(audio_buffer)
    return result


# Include the router in the FastAPI app
app.include_router(router)

# Run the server correctly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=5000)
