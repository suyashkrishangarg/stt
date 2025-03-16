import os
import tempfile
import logging
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union
import soundfile as sf
import numpy as np
import time
import json
from datetime import datetime
from faster_whisper import WhisperModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Whisper API",
    description="OpenAI-compatible API for audio transcription using faster-whisper",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load environment variables
MODEL_SIZE = os.getenv("MODEL_SIZE", "tiny")
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "int8")
DEVICE = os.getenv("DEVICE", "cpu")
BEAM_SIZE = int(os.getenv("BEAM_SIZE", "5"))
ENABLE_API_KEY = os.getenv("ENABLE_API_KEY", "false").lower() == "true"
API_KEY = os.getenv("API_KEY", "")

# Initialize model
logger.info(f"Loading whisper model: {MODEL_SIZE} on {DEVICE} with compute type {COMPUTE_TYPE}")
model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)

# Models for request and response
class TranscriptionRequest(BaseModel):
    file: UploadFile
    model: Optional[str] = None
    prompt: Optional[str] = None
    response_format: Optional[str] = "json"
    temperature: Optional[float] = 0.0
    language: Optional[str] = None

class TranscriptionResponse(BaseModel):
    text: str

class JSONTranscriptionResponse(BaseModel):
    text: str

class VerboseJSONTranscriptionResponse(BaseModel):
    task: str
    language: str
    duration: float
    segments: List[Dict[str, Any]]
    text: str

class SRTTranscriptionResponse(BaseModel):
    text: str

def verify_api_key(authorization: Optional[str] = Header(None)):
    if not ENABLE_API_KEY:
        return True
    
    if not authorization:
        raise HTTPException(status_code=401, detail="API key required")
    
    # Extract Bearer token if present
    if authorization.startswith("Bearer "):
        token = authorization.split("Bearer ")[1]
    else:
        token = authorization
        
    if token != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    return True

@app.post("/v1/audio/transcriptions", response_model=Union[TranscriptionResponse, JSONTranscriptionResponse, VerboseJSONTranscriptionResponse, SRTTranscriptionResponse])
async def transcribe_audio(
    file: UploadFile = File(...),
    model: str = Form(None),
    prompt: Optional[str] = Form(None),
    response_format: Optional[str] = Form("json"),
    temperature: Optional[float] = Form(0.0),
    language: Optional[str] = Form(None),
    _: bool = Depends(verify_api_key)
):
    start_time = time.time()
    
    # Validate file
    if not file.filename or not (file.filename.endswith(('.mp3', '.mp4', '.mpeg', '.mpga', '.m4a', '.wav', '.webm'))):
        raise HTTPException(status_code=400, detail="Invalid file format. Supported formats: mp3, mp4, mpeg, mpga, m4a, wav, webm")

    try:
        # Save uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
            temp_file.write(await file.read())
            temp_file_path = temp_file.name

        # Setup transcription options
        beam_size = BEAM_SIZE
        
        language_options = {}
        if language:
            language_options["language"] = language
            
        initial_prompt = prompt if prompt else None
        
        # Perform transcription
        logger.info(f"Starting transcription for file: {file.filename} using model: {model or MODEL_SIZE}")
        
        segments, info = model.transcribe(
            temp_file_path,
            beam_size=beam_size,
            temperature=temperature,
            initial_prompt=initial_prompt,
            **language_options
        )
        
        # Process segments
        segments_list = []
        full_text = ""
        
        for segment in segments:
            segment_dict = {
                "id": segment.id,
                "seek": segment.seek,
                "start": segment.start,
                "end": segment.end,
                "text": segment.text,
                "tokens": segment.tokens,
                "temperature": segment.temperature,
                "avg_logprob": segment.avg_logprob,
                "compression_ratio": segment.compression_ratio,
                "no_speech_prob": segment.no_speech_prob
            }
            segments_list.append(segment_dict)
            full_text += segment.text
            
        # Format response according to response_format
        if response_format == "text":
            return {"text": full_text.strip()}
        
        elif response_format == "json":
            return {"text": full_text.strip()}
        
        elif response_format == "verbose_json":
            return {
                "task": "transcribe",
                "language": info.language,
                "duration": info.duration,
                "segments": segments_list,
                "text": full_text.strip()
            }
        
        elif response_format == "srt":
            srt_content = ""
            for i, segment in enumerate(segments_list, 1):
                start_time_str = format_timestamp(segment["start"])
                end_time_str = format_timestamp(segment["end"])
                srt_content += f"{i}\n{start_time_str} --> {end_time_str}\n{segment['text'].strip()}\n\n"
            return {"text": srt_content}
        
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported response format: {response_format}")
            
    except Exception as e:
        logger.error(f"Error during transcription: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error during transcription: {str(e)}")
        
    finally:
        # Clean up temporary file
        if 'temp_file_path' in locals():
            try:
                os.unlink(temp_file_path)
            except Exception as e:
                logger.error(f"Error removing temporary file: {str(e)}")
        
        end_time = time.time()
        logger.info(f"Transcription completed in {end_time - start_time:.2f} seconds")

def format_timestamp(seconds):
    """Format timestamp from seconds to SRT format: HH:MM:SS,mmm"""
    hours = int(seconds // 3600)
    seconds %= 3600
    minutes = int(seconds // 60)
    seconds %= 60
    milliseconds = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{int(seconds):02d},{milliseconds:03d}"

@app.get("/health")
def health_check():
    return {"status": "healthy", "model": MODEL_SIZE, "device": DEVICE}

@app.get("/")
def root():
    return {
        "service": "Whisper API",
        "version": "1.0.0",
        "endpoints": {
            "/v1/audio/transcriptions": "OpenAI-compatible audio transcription endpoint",
            "/health": "Health check endpoint"
        }
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    
    uvicorn.run("app:app", host=host, port=port, reload=True) 
