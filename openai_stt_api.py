import os
import sys
import io
import base64
import tempfile
import json
import asyncio
import numpy as np
from time import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import pyaudio
import webrtcvad
import collections
from faster_whisper import WhisperModel, BatchedInferencePipeline
from fastapi import FastAPI, HTTPException, Request, Response, Depends, UploadFile, File, Form, BackgroundTasks, Header
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from starlette.status import HTTP_401_UNAUTHORIZED
import uvicorn
from pydantic import BaseModel, Field
import logging

# Add current directory to path for local imports
current_dir = os.getcwd()
sys.path.append(current_dir)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("openai_stt_api")

# Authentication settings
API_KEY_NAME = "X-Api-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# Create FastAPI app
app = FastAPI(
    title="OpenAI-Compatible STT API",
    description="An API that mimics OpenAI's Speech-to-Text API functionality using faster-whisper",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins in development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Whisper model
WHISPER_MODEL_SIZE = "tiny"  # Can be "tiny", "base", "small", "medium", "large"
model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
batched_model = BatchedInferencePipeline(model=model)
print(f"==> Whisper model '{WHISPER_MODEL_SIZE}' loaded")

# Audio parameters
SAMPLE_RATE = 16000  # Sample rate in Hz
FRAME_DURATION_MS = 30  # Duration of each frame in milliseconds
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)  # Number of samples per frame
BYTES_PER_FRAME = FRAME_SIZE * 2  # 16-bit audio: 2 bytes per sample

# Initialize Voice Activity Detection (VAD)
vad = webrtcvad.Vad(3)  # Aggressiveness mode 3

# Models for request and response
class TranscriptionRequest(BaseModel):
    file: Optional[str] = Field(None, description="Base64-encoded audio file. Either this or a file upload is required.")
    model: str = Field("whisper-1", description="ID of the model to use. Only whisper-1 is currently available.")
    prompt: Optional[str] = Field(None, description="An optional text to guide the model's style")
    response_format: Optional[str] = Field("json", description="The format of the transcript output")
    temperature: Optional[float] = Field(0, description="The sampling temperature")
    language: Optional[str] = Field(None, description="The language of the input audio")

class TranslationRequest(BaseModel):
    file: Optional[str] = Field(None, description="Base64-encoded audio file. Either this or a file upload is required.")
    model: str = Field("whisper-1", description="ID of the model to use. Only whisper-1 is currently available.")
    prompt: Optional[str] = Field(None, description="An optional text to guide the model's style")
    response_format: Optional[str] = Field("json", description="The format of the transcript output")
    temperature: Optional[float] = Field(0, description="The sampling temperature")

# Audio processing functions
def process_audio_bytes(audio_bytes: bytes, language: Optional[str] = None, prompt: Optional[str] = None, temperature: float = 0):
    """
    Process audio bytes using the Whisper model.
    """
    # Convert byte data to a float32 numpy array normalized to [-1, 1]
    audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    
    # Transcribe using Whisper
    segments, info = batched_model.transcribe(
        audio_np,
        beam_size=5,
        language=language if language else 'en',
        no_speech_threshold=0,
        temperature=temperature,
        initial_prompt=prompt
    )
    
    # Collect all text segments
    transcript = ""
    segments_data = []
    for segment in segments:
        transcript += segment.text.lstrip() + " "
        segments_data.append({
            "id": segment.id,
            "start": segment.start,
            "end": segment.end,
            "text": segment.text.strip()
        })
    
    return {
        "text": transcript.strip(),
        "segments": segments_data,
        "language": info.language
    }

def save_base64_audio(base64_data: str):
    """
    Save base64-encoded audio to a temporary file.
    """
    # Detect and remove data URI scheme prefix if present
    if "base64," in base64_data:
        base64_data = base64_data.split("base64,")[1]
    
    # Decode base64 data
    audio_bytes = base64.b64decode(base64_data)
    
    # Create a temporary file
    fd, temp_file_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    
    # Write audio bytes to the temporary file
    with open(temp_file_path, "wb") as f:
        f.write(audio_bytes)
    
    return temp_file_path

def save_uploaded_file(file: UploadFile):
    """
    Save an uploaded file to a temporary location.
    """
    # Create a temporary file
    fd, temp_file_path = tempfile.mkstemp(suffix=os.path.splitext(file.filename)[1])
    os.close(fd)
    
    # Write content to the temporary file
    with open(temp_file_path, "wb") as f:
        f.write(file.file.read())
    
    return temp_file_path

def cleanup_temp_file(file_path: str):
    """
    Remove a temporary file.
    """
    try:
        os.remove(file_path)
    except Exception as e:
        print(f"Error removing temporary file: {e}")

# Authentication function
async def get_api_key(
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    authorization: Optional[str] = Header(None)
):
    """
    Get API key from either X-API-Key header or Authorization header.
    This ensures compatibility with both direct HTTP clients and the OpenAI client.
    """
    api_key = None
    
    # Try to get API key from X-API-Key header
    if x_api_key:
        api_key = x_api_key
    
    # If not found, try to get from Authorization header
    elif authorization:
        if authorization.startswith("Bearer "):
            api_key = authorization.replace("Bearer ", "")
    
    # If still no API key, raise an error
    if not api_key:
        raise HTTPException(
            status_code=HTTP_401_UNAUTHORIZED,
            detail="API key is missing",
        )
    
    # For simplicity, any non-empty key is accepted in this example
    return api_key

# Streaming transcription function
async def stream_audio_from_microphone():
    """
    Stream audio from the microphone and transcribe it in real-time.
    """
    # Initialize PyAudio
    pa = pyaudio.PyAudio()
    stream = pa.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=FRAME_SIZE
    )
    
    # Setup for smoothing VAD decisions
    buffer_duration_sec = 0.3
    num_padding_frames = int(buffer_duration_sec / (FRAME_DURATION_MS / 1000))
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    recorded_frames = []
    triggered = False
    
    sliding_window_seconds = 1
    sliding_window_frames = int(sliding_window_seconds * SAMPLE_RATE / FRAME_SIZE)
    
    try:
        # Initial delay to let the stream initialize
        await asyncio.sleep(0.1)
        
        yield json.dumps({"status": "listening"}) + "\n"
        
        while True:
            frame = stream.read(FRAME_SIZE, exception_on_overflow=False)
            is_speech = vad.is_speech(frame, SAMPLE_RATE)
            
            if not triggered:
                ring_buffer.append((frame, is_speech))
                # Trigger recording if >50% of frames in the buffer are speech
                num_voiced = sum(1 for _, speech in ring_buffer if speech)
                if num_voiced > 0.5 * ring_buffer.maxlen:
                    triggered = True
                    # Include buffered frames to capture the beginning of speech
                    recorded_frames.extend(f for f, _ in ring_buffer if speech)
                    ring_buffer.clear()
                    yield json.dumps({"status": "started"}) + "\n"
            else:
                recorded_frames.append(frame)
                ring_buffer.append((frame, is_speech))
                
                # Process the sliding window if we have enough frames
                if len(recorded_frames) >= sliding_window_frames:
                    window_frames = recorded_frames[-sliding_window_frames:]
                    audio_data = b"".join(window_frames)
                    audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                    
                    # Transcribe the sliding window
                    segments, _ = batched_model.transcribe(
                        audio_np,
                        beam_size=5,
                        language='en',
                        no_speech_threshold=0,
                        temperature=0
                    )
                    
                    transcript = ""
                    for segment in segments:
                        transcript += segment.text.lstrip() + " "
                    
                    if transcript.strip():
                        yield json.dumps({"partial": transcript.strip()}) + "\n"
                
                # Stop recording if >80% of recent frames are silence
                num_unvoiced = sum(1 for _, speech in ring_buffer if not speech)
                if num_unvoiced > 0.8 * ring_buffer.maxlen:
                    break
            
            # Small delay to prevent CPU overload
            await asyncio.sleep(0.01)
        
        # Final transcription of the entire recording
        if recorded_frames:
            audio_data = b"".join(recorded_frames)
            result = process_audio_bytes(audio_data)
            yield json.dumps({"text": result["text"]}) + "\n"
            yield json.dumps({"status": "complete"}) + "\n"
        else:
            yield json.dumps({"error": "No speech detected"}) + "\n"
    
    finally:
        # Clean up resources
        stream.stop_stream()
        stream.close()
        pa.terminate()

# API endpoints
@app.post("/v1/audio/transcriptions")
async def create_transcription(
    background_tasks: BackgroundTasks,
    file: Optional[UploadFile] = File(None),
    request: Optional[TranscriptionRequest] = None,
    model: str = Form("whisper-1", include_in_schema=False),
    prompt: Optional[str] = Form(None, include_in_schema=False),
    response_format: str = Form("json", include_in_schema=False),
    temperature: float = Form(0, include_in_schema=False),
    language: Optional[str] = Form(None, include_in_schema=False),
    api_key: str = Depends(get_api_key)
):
    """
    Transcribes audio into the input language.
    Supports both JSON request body and multipart form data.
    """
    try:
        logger.info("Received transcription request")
        temp_file_path = None
        
        # Get parameters from either form data or JSON body
        actual_model = model
        actual_prompt = prompt
        actual_response_format = response_format
        actual_temperature = temperature
        actual_language = language
        
        # If we have a JSON request, use those values instead
        if request:
            logger.info("Processing JSON request")
            if request.file:
                # Handle base64-encoded audio
                logger.info("Processing base64-encoded audio")
                temp_file_path = save_base64_audio(request.file)
                with open(temp_file_path, "rb") as f:
                    audio_bytes = f.read()
                logger.info(f"Saved base64 audio to {temp_file_path}, size: {len(audio_bytes)} bytes")
            actual_model = request.model
            actual_prompt = request.prompt
            actual_response_format = request.response_format or "json"
            actual_temperature = request.temperature
            actual_language = request.language
        elif file:
            logger.info(f"Processing file upload: {file.filename}, content-type: {file.content_type}")
            # Handle file upload as before
            temp_file_path = save_uploaded_file(file)
            with open(temp_file_path, "rb") as f:
                audio_bytes = f.read()
            logger.info(f"Saved uploaded file to {temp_file_path}, size: {len(audio_bytes)} bytes")
        else:
            logger.error("No audio file provided in request")
            raise HTTPException(status_code=400, detail="No audio file provided")
        
        # Schedule cleanup of temporary file
        if temp_file_path:
            background_tasks.add_task(cleanup_temp_file, temp_file_path)
        
        # Process the audio
        logger.info(f"Processing audio with params: lang={actual_language}, temp={actual_temperature}")
        result = process_audio_bytes(
            audio_bytes=audio_bytes,
            language=actual_language,
            prompt=actual_prompt,
            temperature=actual_temperature
        )
        
        logger.info(f"Transcription complete, detected language: {result['language']}, text length: {len(result['text'])}")
        
        # Format the response based on response_format
        if actual_response_format == "json":
            logger.info("Returning JSON response")
            return {
                "text": result["text"]
            }
        elif actual_response_format == "verbose_json":
            logger.info("Returning verbose JSON response")
            return {
                "task": "transcribe",
                "language": result["language"],
                "duration": 0,  # We don't calculate this currently
                "segments": result["segments"],
                "text": result["text"]
            }
        elif actual_response_format == "text":
            logger.info("Returning text response")
            return Response(content=result["text"], media_type="text/plain")
        elif actual_response_format == "srt":
            # Generate SRT format
            logger.info("Returning SRT response")
            srt_content = ""
            for i, segment in enumerate(result["segments"]):
                srt_content += f"{i+1}\n"
                start_time = format_timestamp(segment["start"])
                end_time = format_timestamp(segment["end"])
                srt_content += f"{start_time} --> {end_time}\n"
                srt_content += f"{segment['text']}\n\n"
            return Response(content=srt_content, media_type="text/plain")
        elif actual_response_format == "vtt":
            # Generate VTT format
            logger.info("Returning VTT response")
            vtt_content = "WEBVTT\n\n"
            for i, segment in enumerate(result["segments"]):
                start_time = format_timestamp(segment["start"], vtt=True)
                end_time = format_timestamp(segment["end"], vtt=True)
                vtt_content += f"{start_time} --> {end_time}\n"
                vtt_content += f"{segment['text']}\n\n"
            return Response(content=vtt_content, media_type="text/plain")
        else:
            logger.error(f"Unsupported response format: {actual_response_format}")
            raise HTTPException(status_code=400, detail=f"Unsupported response format: {actual_response_format}")
    
    except Exception as e:
        logger.error(f"Error processing transcription request: {str(e)}", exc_info=True)
        # If there was an exception and we have a temp file, clean it up
        if temp_file_path:
            background_tasks.add_task(cleanup_temp_file, temp_file_path)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/audio/translations")
async def create_translation(
    background_tasks: BackgroundTasks,
    file: Optional[UploadFile] = File(None),
    request: Optional[TranslationRequest] = None,
    model: str = Form("whisper-1", include_in_schema=False),
    prompt: Optional[str] = Form(None, include_in_schema=False),
    response_format: str = Form("json", include_in_schema=False),
    temperature: float = Form(0, include_in_schema=False),
    api_key: str = Depends(get_api_key)
):
    """
    Translates audio into English.
    Supports both JSON request body and multipart form data.
    """
    try:
        temp_file_path = None
        
        # Get parameters from either form data or JSON body
        actual_model = model
        actual_prompt = prompt
        actual_response_format = response_format
        actual_temperature = temperature
        
        # If we have a JSON request, use those values instead
        if request:
            if request.file:
                # Handle base64-encoded audio
                temp_file_path = save_base64_audio(request.file)
                with open(temp_file_path, "rb") as f:
                    audio_bytes = f.read()
            actual_model = request.model
            actual_prompt = request.prompt
            actual_response_format = request.response_format or "json"
            actual_temperature = request.temperature
        elif file:
            # Handle file upload as before
            temp_file_path = save_uploaded_file(file)
            with open(temp_file_path, "rb") as f:
                audio_bytes = f.read()
        else:
            raise HTTPException(status_code=400, detail="No audio file provided")
        
        # Schedule cleanup of temporary file
        if temp_file_path:
            background_tasks.add_task(cleanup_temp_file, temp_file_path)
        
        # Process the audio and force English output
        result = process_audio_bytes(
            audio_bytes=audio_bytes,
            language="en",  # Force English output for translation
            prompt=actual_prompt,
            temperature=actual_temperature
        )
        
        # Format the response based on response_format
        if actual_response_format == "json":
            return {
                "text": result["text"]
            }
        elif actual_response_format == "verbose_json":
            return {
                "task": "translate",
                "language": result["language"],
                "duration": 0,  # We don't calculate this currently
                "segments": result["segments"],
                "text": result["text"]
            }
        elif actual_response_format == "text":
            return Response(content=result["text"], media_type="text/plain")
        elif actual_response_format == "srt":
            # Generate SRT format
            srt_content = ""
            for i, segment in enumerate(result["segments"]):
                srt_content += f"{i+1}\n"
                start_time = format_timestamp(segment["start"])
                end_time = format_timestamp(segment["end"])
                srt_content += f"{start_time} --> {end_time}\n"
                srt_content += f"{segment['text']}\n\n"
            return Response(content=srt_content, media_type="text/plain")
        elif actual_response_format == "vtt":
            # Generate VTT format
            vtt_content = "WEBVTT\n\n"
            for i, segment in enumerate(result["segments"]):
                start_time = format_timestamp(segment["start"], vtt=True)
                end_time = format_timestamp(segment["end"], vtt=True)
                vtt_content += f"{start_time} --> {end_time}\n"
                vtt_content += f"{segment['text']}\n\n"
            return Response(content=vtt_content, media_type="text/plain")
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported response format: {actual_response_format}")
    
    except Exception as e:
        # If there was an exception and we have a temp file, clean it up
        if temp_file_path:
            background_tasks.add_task(cleanup_temp_file, temp_file_path)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/audio/transcriptions/streaming")
async def stream_transcription(api_key: str = Depends(get_api_key)):
    """
    Stream audio from microphone and transcribe in real-time.
    This is a custom endpoint not in the original OpenAI API.
    """
    return StreamingResponse(
        stream_audio_from_microphone(),
        media_type="text/event-stream"
    )

@app.get("/v1/models")
async def list_models(api_key: str = Depends(get_api_key)):
    """
    List available models.
    This endpoint mimics the OpenAI models endpoint.
    """
    # Currently only supporting the whisper-1 model
    return {
        "data": [
            {
                "id": "whisper-1",
                "object": "model",
                "created": int(time()),
                "owned_by": "openai",
                "permission": [],
                "root": "whisper-1",
                "parent": None
            }
        ],
        "object": "list"
    }

@app.get("/status")
async def status():
    """
    Check if the API is running.
    """
    return {"status": "ok", "version": "1.0.0"}

# Helper functions
def format_timestamp(seconds, vtt=False):
    """
    Format a timestamp for SRT or VTT format.
    """
    hours = int(seconds / 3600)
    seconds %= 3600
    minutes = int(seconds / 60)
    seconds %= 60
    milliseconds = int((seconds - int(seconds)) * 1000)
    seconds = int(seconds)
    
    if vtt:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"
    else:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

# Main function to run the API
def main():
    """Run the API server."""
    uvicorn.run(app, host="0.0.0.0", port=8001)

if __name__ == "__main__":
    main() 