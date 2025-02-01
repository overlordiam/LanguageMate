from fastapi import APIRouter, UploadFile
import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

from api.orchestrator import PipelineOrchestrator

router = APIRouter()
orchestrator = PipelineOrchestrator()

@router.post("/process")
async def process_voice(file: UploadFile):
    audio = await file.read()
    result = await orchestrator.process_audio(audio)
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(router, host="127.0.0.1", port=8000)