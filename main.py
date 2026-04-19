import os
import io
import torch
import torchaudio
from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.responses import Response

app = FastAPI(title="Silero Voice API")

# Оптимизация для бесплатных тарифов (CPU, 1 поток)
torch.set_num_threads(1)
os.environ["OMP_NUM_THREADS"] = "1"

# Загрузка моделей при старте
@app.on_event("startup")
def load_models():
    global tts_model, stt_model, tts_speakers
    
    # TTS
tts_model, example_text = torch.hub.load(
    repo_or_dir="snakers4/silero-models",
    model="silero_tts",
    language="ru",
    speaker="kseniya",
    trust_repo=True  # <-- обязательно
	)
    tts_model.to("cpu")
    tts_speakers = example_text.speakers
    tts_model.to("cpu")
    tts_speakers = example_text.speakers
    
    # STT
    stt_model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-models",
        model="silero_stt"
    )
    stt_model.to("cpu")

@app.post("/tts")
def text_to_speech(text: str = Form(...), speaker: str = Form("kseniya")):
    if speaker not in tts_speakers:
        raise HTTPException(400, f"Доступные голоса: {', '.join(tts_speakers)}")
    
    audio = tts_model.apply_tts(text=text, speaker=speaker, sample_rate=48000)
    buffer = io.BytesIO()
    torchaudio.save(buffer, audio, 48000, format="wav")
    return Response(content=buffer.getvalue(), media_type="audio/wav")

@app.post("/stt")
async def speech_to_text(file: UploadFile):
    audio_bytes = await file.read()
    buffer = io.BytesIO(audio_bytes)
    waveform, sample_rate = torchaudio.load(buffer)
    
    # Приводим к 16kHz (требуется для STT)
    if sample_rate != 16000:
        transform = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = transform(waveform)
        
    text = stt_model(waveform.squeeze())
    return {"text": text}
